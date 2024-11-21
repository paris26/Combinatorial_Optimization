from data_generator import ThermalNetworkDataGenerator
from thermal_optimizer import ThermalNetworkOptimizer
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import csv

def ensure_directories():
    """Create required output directories"""
    directories = ['images', 'results', 'metrics', 'analysis']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def read_instances(filename='instances.txt'):
    """Read problem instances from file"""
    instances = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#'):
                nums = [int(x) for x in line.strip().split(',')]
                if len(nums) == 6:
                    instances.append({
                        'num_facilities': nums[0],
                        'num_substations': nums[1],
                        'num_customers': nums[2],
                        'grid_size': nums[3],
                        'min_facility_per_sub': nums[4],
                        'max_facility_per_sub': nums[5]
                    })
    return instances

def save_metric_choices(instance_num: int, solution: dict):
    """Save the chosen distance metrics"""
    metrics_df = pd.DataFrame()
    
    # Facility to substation metrics
    f_s_metrics = [(f"{src}->{dst}", metric) 
                   for (src, dst), metric in solution['distance_metrics']['facility_substation'].items()]
    
    # Substation to customer metrics
    s_c_metrics = [(f"{src}->{dst}", metric) 
                   for (src, dst), metric in solution['distance_metrics']['substation_customer'].items()]
    
    all_metrics = f_s_metrics + s_c_metrics
    metrics_df = pd.DataFrame(all_metrics, columns=['Connection', 'Chosen_Metric'])
    metrics_df.to_csv(f'metrics/distance_metrics_instance_{instance_num}.csv', index=False)

def save_facility_connections(instance_num: int, solution: dict):
    """Save facility connection patterns"""
    connections_df = pd.DataFrame([
        {
            'Facility': i,
            'Substation': j,
            'Connected': connected,
            'Distance_Metric': solution['distance_metrics']['facility_substation'][(i,j)]
        }
        for (i,j), connected in solution['facility_connections']['connections'].items()
        if connected
    ])
    connections_df.to_csv(f'analysis/facility_connections_instance_{instance_num}.csv', index=False)

    # Save facilities per substation summary
    facilities_per_sub_df = pd.DataFrame([
        {
            'Substation': j,
            'Num_Facilities': count,
            'Is_Open': j in solution['opened_substations']
        }
        for j, count in solution['facility_connections']['facilities_per_sub'].items()
    ])
    facilities_per_sub_df.to_csv(f'analysis/facilities_per_sub_instance_{instance_num}.csv', index=False)

def run_optimization(instance_num, params):
    """Run optimization for single instance"""
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
    
    # Initialize optimizer and solve
    optimizer = ThermalNetworkOptimizer(generator)
    
    start_time = time.time()
    solution = optimizer.solve_single_stage(
        min_facility_per_sub=params['min_facility_per_sub'],
        max_facility_per_sub=params['max_facility_per_sub']
    )
    solve_time = time.time() - start_time
    
    result_summary = {
        'Instance': instance_num,
        'Facilities': params['num_facilities'],
        'Substations': params['num_substations'],
        'Customers': params['num_customers'],
        'Grid Size': params['grid_size'],
        'Min Facilities per Sub': params['min_facility_per_sub'],
        'Max Facilities per Sub': params['max_facility_per_sub'],
        'Solution Time': solve_time,
        'Status': 'Optimal' if solution else 'Infeasible/Error'
    }
    
    if solution:
        result_summary.update({
            'Total Cost': solution['objective_value'],
            'Open Substations': len(solution['opened_substations']),
            'Opened Substations': ', '.join(solution['opened_substations']),
            'Avg Facilities per Sub': sum(solution['facility_connections']['facilities_per_sub'].values()) / 
                                    len(solution['opened_substations'])
        })
        
        # Save detailed results
        save_metric_choices(instance_num, solution)
        save_facility_connections(instance_num, solution)
        
        # Save flows to CSV
        flows_data = []
        for (i, j, s), flow in solution['flows']['facility_to_substation'].items():
            if flow > 1e-6:
                metric = solution['distance_metrics']['facility_substation'][(i,j)]
                flows_data.append({
                    'From': i,
                    'To': j,
                    'Season': s,
                    'Flow': flow,
                    'Type': 'Facility to Substation',
                    'Distance_Metric': metric
                })
        
        for (j, k, s), flow in solution['flows']['substation_to_customer'].items():
            if flow > 1e-6:
                metric = solution['distance_metrics']['substation_customer'][(j,k)]
                flows_data.append({
                    'From': j,
                    'To': k,
                    'Season': s,
                    'Flow': flow,
                    'Type': 'Substation to Customer',
                    'Distance_Metric': metric
                })
        
        flows_df = pd.DataFrame(flows_data)
        flows_df.to_csv(f'results/flows_instance_{instance_num}.csv', index=False)
        
        # Save assignments with chosen metrics
        assignments_data = [
            {
                'Customer': k,
                'Assigned_Substation': v,
                'Distance_Metric': solution['distance_metrics']['substation_customer'][(v,k)],
                'Num_Facilities_Serving': solution['facility_connections']['facilities_per_sub'][v]
            }
            for k, v in solution['assignments'].items()
        ]
        assignments_df = pd.DataFrame(assignments_data)
        assignments_df.to_csv(f'results/assignments_instance_{instance_num}.csv', index=False)
        
    return result_summary

def main():
    """Main execution function"""
    ensure_directories()
    
    try:
        instances = read_instances()
    except FileNotFoundError:
        print("Error: instances.txt file not found!")
        return
    
    if not instances:
        print("Error: No valid instances found in instances.txt!")
        return
    
    results = []
    
    for i, instance_params in enumerate(instances, 1):
        result = run_optimization(i, instance_params)
        results.append(result)
        
        print(f"\nInstance {i} Summary:")
        print(f"Status: {result['Status']}")
        print(f"Solution Time: {result['Solution Time']:.2f} seconds")
        if 'Total Cost' in result:
            print(f"Total Cost: {result['Total Cost']:,.2f}")
            print(f"Average Facilities per Substation: {result['Avg Facilities per Sub']:.2f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/summary_all_instances.csv', index=False)
    print("\nAll results saved to respective directories")

if __name__ == "__main__":
    main()