from data_generator import ThermalNetworkDataGenerator
from thermal_optimizer import ThermalNetworkOptimizer
import matplotlib.pyplot as plt
import pandas as pd
import time

def run_optimization(num_facilities=2, 
                    num_substations=3, 
                    num_customers=5, 
                    grid_size=100):
    """
    Run optimization for a thermal network instance
    """
    print("Initializing data generation...")
    generator = ThermalNetworkDataGenerator(
        num_facilities=num_facilities,
        num_substations=num_substations,
        num_customers=num_customers,
        grid_size=grid_size
    )
    
    print("\nCreating network visualization...")
    plt = generator.visualize_network()
    plt.savefig('network_layout.png')
    plt.close()
    
    print("\nInitializing optimizer...")
    optimizer = ThermalNetworkOptimizer(generator)
    
    # Solve optimization problem
    print("\nSolving optimization problem...")
    start_time = time.time()
    solution = optimizer.solve_single_stage()
    solve_time = time.time() - start_time
    
    if solution:
        # Print results
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        
        print(f"\nTotal Cost: {solution['objective_value']:,.2f}")
        print(f"Solution Time: {solve_time:.2f} seconds")
        print(f"\nOpened Substations: {', '.join(solution['opened_substations'])}")
        
        print("\nCustomer Assignments:")
        for city, substation in solution['assignments'].items():
            print(f"{city} → {substation}")
        
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
        flows_df.to_csv('optimization_flows.csv', index=False)
        print("\nFlow details have been saved to 'optimization_flows.csv'")
        
    else:
        print("Failed to find optimal solution!")

def run_multiple_instances():
    """
    Run optimization for multiple problem instances
    """
    instances = [
        (2, 3, 5),   # Small instance
        (3, 4, 8),   # Medium instance
        (4, 6, 12),  # Large instance
    ]
    
    results = []
    for num_facilities, num_substations, num_customers in instances:
        print(f"\nSolving instance: {num_facilities} facilities, {num_substations} substations, {num_customers} customers")
        
        generator = ThermalNetworkDataGenerator(
            num_facilities=num_facilities,
            num_substations=num_substations,
            num_customers=num_customers
        )
        
        optimizer = ThermalNetworkOptimizer(generator)
        
        start_time = time.time()
        solution = optimizer.solve_single_stage()
        solve_time = time.time() - start_time
        
        if solution:
            results.append({
                'Facilities': num_facilities,
                'Substations': num_substations,
                'Customers': num_customers,
                'Total Cost': solution['objective_value'],
                'Solution Time': solve_time,
                'Open Substations': len(solution['opened_substations'])
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('multiple_instances_results.csv', index=False)
    print("\nResults have been saved to 'multiple_instances_results.csv'")

if __name__ == "__main__":
    # Run a single instance
    print("Solving single instance...")
    run_optimization(
        num_facilities=2,
        num_substations=3,
        num_customers=5,
        grid_size=100
    )
    
    # Run multiple instances
    print("\nSolving multiple instances...")
    run_multiple_instances()