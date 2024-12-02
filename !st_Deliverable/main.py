from data_generator import ThermalNetworkDataGenerator
from random_data_generator import RandomizedThermalNetworkGenerator
from thermal_optimizer import ThermalNetworkOptimizer
from branch_and_bound_solver import BranchAndBoundSolver
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import argparse
import shutil

def cleanup_directories():
    """Remove existing images and results directories"""
    directories = ['images', 'results']
    for directory in directories:
        if os.path.exists(directory):
            print(f"Removing existing {directory} directory...")
            shutil.rmtree(directory)

def ensure_directories():
    """Create output directories if they don't exist"""
    directories = ['images', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created new {directory} directory")

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

def run_optimization(instance_num, params, use_custom_bnb=False):
    """Run optimization for a thermal network instance"""
    print(f"\nSolving Instance {instance_num}")
    print("="*50)
    print(f"Configuration: {params}")
    print(f"Solver: {'Custom Branch & Bound' if use_custom_bnb else 'Gurobi Branch & Bound'}")
    
    # Initialize generator
    generator = ThermalNetworkDataGenerator(
        num_facilities=params['num_facilities'],
        num_substations=params['num_substations'],
        num_customers=params['num_customers'],
        grid_size=params['grid_size']
    )

    # generator = RandomizedThermalNetworkGenerator( 
    #     num_facilities=params['num_facilities'],
    #     num_substations=params['num_substations'],
    #     num_customers=params['num_customers'],
    #     grid_size=params['grid_size']
    # )
    
    # Create and save network visualization
    plt = generator.visualize_network()
    plt.savefig(f'images/network_layout_instance_{instance_num}.png')
    plt.close()
    
    # Initialize optimizer
    optimizer = ThermalNetworkOptimizer(generator)
    
    # Solve using either custom B&B or Gurobi
    start_time = time.time()
    if use_custom_bnb:
        bnb_solver = BranchAndBoundSolver(optimizer)
        solution = bnb_solver.solve()
    else:
        solution = optimizer.solve_single_stage()
    solve_time = time.time() - start_time
    
    result_summary = {
        'Instance': instance_num,
        'Solver': 'Custom B&B' if use_custom_bnb else 'Gurobi B&B',
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
        
        if use_custom_bnb and 'branch_and_bound_stats' in solution:
            result_summary['Nodes Explored'] = solution['branch_and_bound_stats']['nodes_explored']
        
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
        solver_str = 'custom_bnb' if use_custom_bnb else 'gurobi'
        flows_df.to_csv(f'results/flows_instance_{instance_num}_{solver_str}.csv', index=False)
        
        # Save assignments
        assignments_df = pd.DataFrame([
            {'Customer': k, 'Assigned_Substation': v}
            for k, v in solution['assignments'].items()
        ])
        assignments_df.to_csv(f'results/assignments_instance_{instance_num}_{solver_str}.csv', index=False)
        
    return result_summary

def main():
    # Add command line argument parser
    parser = argparse.ArgumentParser(description='Solve thermal network optimization.')
    parser.add_argument('--solver', choices=['gurobi', 'custom'], default='gurobi',
                       help='Choose solver: gurobi (default) or custom branch & bound')
    args = parser.parse_args()
    
    # Use custom B&B if specified
    use_custom_bnb = (args.solver == 'custom')
    
    # Clean up existing directories
    cleanup_directories()
    
    # Create fresh output directories
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
        try:
            result = run_optimization(i, instance_params, use_custom_bnb)
            results.append(result)
            
            # Print instance summary
            print(f"\nInstance {i} Summary:")
            print(f"Solver: {result['Solver']}")
            print(f"Status: {result['Status']}")
            print(f"Solution Time: {result['Solution Time']:.2f} seconds")
            if 'Total Cost' in result:
                print(f"Total Cost: {result['Total Cost']:,.2f}")
            if 'Nodes Explored' in result:
                print(f"Nodes Explored: {result['Nodes Explored']}")
        except Exception as e:
            print(f"Error processing instance {i}: {str(e)}")
            continue
    
    # Save summary results
    if results:
        results_df = pd.DataFrame(results)
        solver_str = 'custom_bnb' if use_custom_bnb else 'gurobi'
        results_df.to_csv(f'results/summary_all_instances_{solver_str}.csv', index=False)
        print("\nAll results have been saved to the 'results' directory")
        print("All network layouts have been saved to the 'images' directory")

if __name__ == "__main__":
    main()