from data_generator import ThermalNetworkDataGenerator
from thermal_optimizer import ThermalNetworkOptimizer
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import csv

def ensure_directories():
    """Create output directories if they don't exist"""
    directories = ['images', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def read_instances(filename='instances.txt'):
    """Read instances from the text file"""
    instances = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.strip() and not line.strip().startswith('#'):
                # Parse the line
                nums = [int(x) for x in line.strip().split(',')]
                if len(nums) == 4:
                    instances.append({
                        'num_facilities': nums[0],
                        'num_substations': nums[1],
                        'num_customers': nums[2],
                        'grid_size': nums[3]
                    })
    return instances

def run_optimization(instance_num, params):
    """
    Run optimization for a thermal network instance
    """
    print(f"\nSolving Instance {instance_num}")
    print("="*50)
    print(f"Configuration: {params}")
    
    # Initialize generator
    generator = ThermalNetworkDataGenerator(
        num_facilities=params['num_facilities'],
        num_substations=params['num_substations'],
        num_customers=params['num_customers'],
        grid_size=params['grid_size']
    )
    
    # Create and save network visualization
    plt = generator.visualize_network()
    plt.savefig(f'images/network_layout_instance_{instance_num}.png')
    plt.close()
    
    # Initialize optimizer
    optimizer = ThermalNetworkOptimizer(generator)
    
    # Solve optimization problem
    start_time = time.time()
    solution = optimizer.solve_single_stage()
    solve_time = time.time() - start_time
    
    result_summary = {
        'Instance': instance_num,
        'Facilities': params['num_facilities'],
        'Substations': params['num_substations'],
        'Customers': params['num_customers'],
        'Grid Size': params['grid_size'],
        'Solution Time': solve_time,
        'Status': 'Optimal' if solution else 'Infeasible/Error'
    }
    
    if solution:
        result_summary.update({
            'Total Cost': solution['objective_value'],
            'Open Substations': len(solution['opened_substations']),
            'Opened Substations': ', '.join(solution['opened_substations'])
        })
        
        # Save flows to CSV
        flows_data = []
        for (i, j, s), flow in solution['flows']['facility_to_substation'].items():
            if flow > 1e-6:  # Only include non-zero flows
                flows_data.append({
                    'From': i,
                    'To': j,
                    'Season': s,
                    'Flow': flow,
                    'Type': 'Facility to Substation'
                })
        
        for (j, k, s), flow in solution['flows']['substation_to_customer'].items():
            if flow > 1e-6:  # Only include non-zero flows
                flows_data.append({
                    'From': j,
                    'To': k,
                    'Season': s,
                    'Flow': flow,
                    'Type': 'Substation to Customer'
                })
        
        flows_df = pd.DataFrame(flows_data)
        flows_df.to_csv(f'results/flows_instance_{instance_num}.csv', index=False)
        
        # Save assignments
        assignments_df = pd.DataFrame([
            {'Customer': k, 'Assigned_Substation': v}
            for k, v in solution['assignments'].items()
        ])
        assignments_df.to_csv(f'results/assignments_instance_{instance_num}.csv', index=False)
        
    return result_summary

def main():
    """Main function to run all instances"""
    # Ensure output directories exist
    ensure_directories()
    
    # Read instances
    try:
        instances = read_instances()
    except FileNotFoundError:
        print("Error: instances.txt file not found!")
        print("Please create an instances.txt file with the problem configurations.")
        return
    
    if not instances:
        print("Error: No valid instances found in instances.txt!")
        return
    
    # Store results for all instances
    results = []
    
    # Process each instance
    for i, instance_params in enumerate(instances, 1):
        result = run_optimization(i, instance_params)
        results.append(result)
        
        # Print instance summary
        print(f"\nInstance {i} Summary:")
        print(f"Status: {result['Status']}")
        print(f"Solution Time: {result['Solution Time']:.2f} seconds")
        if 'Total Cost' in result:
            print(f"Total Cost: {result['Total Cost']:,.2f}")
    
    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/summary_all_instances.csv', index=False)
    print("\nAll results have been saved to the 'results' directory")
    print("All network layouts have been saved to the 'images' directory")

if __name__ == "__main__":
    main()