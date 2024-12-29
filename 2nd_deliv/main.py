from data_generator import ThermalNetworkDataGenerator
from thermal_optimizer import ThermalNetworkOptimizer
from optimized_branch_and_bound_solver import OptimizedBranchAndBoundSolver
from branch_and_bound_solver import BranchAndBoundSolver
import pandas as pd
import time
import os
import argparse
import shutil
# Add these imports at the top of main.py
import os
import warnings
# Suppress Qt warnings
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# Suppress data generator warnings if desired
warnings.filterwarnings('ignore', category=UserWarning)

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


def run_optimization(instance_num, params, solver_type='gurobi'):
    """
    Run optimization for a thermal network instance
    Args:
        instance_num: Instance number
        params: Instance parameters
        solver_type: One of ['gurobi', 'custom', 'optimized']
    """
    print(f"\nSolving Instance {instance_num}")
    print("=" * 50)
    print(f"Configuration: {params}")
    print(f"Solver: {solver_type.capitalize()} Solver")

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

    # Solve using selected solver
    start_time = time.time()
    if solver_type == 'optimized':
        bnb_solver = OptimizedBranchAndBoundSolver(optimizer)
        solution = bnb_solver.solve()
    elif solver_type == 'custom':
        bnb_solver = BranchAndBoundSolver(optimizer)
        solution = bnb_solver.solve()
    else:  # gurobi
        solution = optimizer.solve_single_stage()
    solve_time = time.time() - start_time

    result_summary = {
        'Instance': instance_num,
        'Solver': f'{solver_type.capitalize()} Solver',
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

        if solver_type in ['custom', 'optimized'] and 'branch_and_bound_stats' in solution:
            result_summary.update(solution['branch_and_bound_stats'])
            if 'search_statistics' in solution['branch_and_bound_stats']:
                stats = solution['branch_and_bound_stats']['search_statistics']
                result_summary.update({
                    'Total Nodes': stats['total_nodes'],
                    'Pruned Nodes': stats['pruned_nodes'],
                    'Integer Solutions': stats['integer_solutions'],
                    'Best Bound': stats['best_bound']
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
        flows_df.to_csv(f'results/flows_instance_{instance_num}_{solver_type}.csv', index=False)

        # Save assignments
        assignments_df = pd.DataFrame([
            {'Customer': k, 'Assigned_Substation': v}
            for k, v in solution['assignments'].items()
        ])
        assignments_df.to_csv(f'results/assignments_instance_{instance_num}_{solver_type}.csv', index=False)

    return result_summary


def main():
    # Add command line argument parser
    parser = argparse.ArgumentParser(description='Solve thermal network optimization.')
    parser.add_argument('--solver', choices=['gurobi', 'custom', 'optimized'], default='gurobi',
                        help='Choose solver: gurobi (default), custom branch & bound, or optimized branch & bound')
    args = parser.parse_args()

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
        try:
            result = run_optimization(i, instance_params, solver_type=args.solver)
            results.append(result)

            # Print instance summary
            print(f"\nInstance {i} Summary:")
            print(f"Solver: {result['Solver']}")
            print(f"Status: {result['Status']}")
            print(f"Solution Time: {result['Solution Time']:.2f} seconds")
            if 'Total Cost' in result:
                print(f"Total Cost: {result['Total Cost']:,.2f}")
            if 'Total Nodes' in result:
                print(f"Total Nodes: {result['Total Nodes']}")
                print(f"Pruned Nodes: {result['Pruned Nodes']}")
                print(f"Integer Solutions: {result['Integer Solutions']}")
                print(f"Best Bound: {result['Best Bound']:.2f}")
        except Exception as e:
            print(f"Error processing instance {i}: {str(e)}")
            continue

    # Save summary results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/summary_all_instances_{args.solver}.csv', index=False)
        print("\nAll results have been saved to the 'results' directory")
        print("All network layouts have been saved to the 'images' directory")


if __name__ == "__main__":
    main()