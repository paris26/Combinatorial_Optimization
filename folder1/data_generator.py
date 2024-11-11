# data_generator.py

import random
import numpy as np
import json
import os

def validate_range(input_str, data_type=float):
    """Validate and return a range as a tuple (lower, upper)."""
    try:
        lower, upper = map(data_type, input_str.strip().split())
        if lower > upper:
            raise ValueError("Lower bound must be less than upper bound.")
        return lower, upper
    except Exception as e:
        raise ValueError(f"Invalid input range: {e}")

def get_user_input():
    """Prompt the user for all necessary inputs and validate them."""
    num_instances = int(input("Enter the number of problem instances to generate: "))
    num_facilities_range = validate_range(input("Enter the range for number of facilities (e.g., '3 5'): "), int)
    num_substations_range = validate_range(input("Enter the range for number of substations (e.g., '3 5'): "), int)
    num_customers_range = validate_range(input("Enter the range for number of customers (e.g., '5 8'): "), int)
    production_cost_range = validate_range(input("Enter the range for production cost (e.g., '20 40'): "))
    production_capacity_range = validate_range(input("Enter the range for production capacity (e.g., '1500 2000'): "))
    fixed_cost_range = validate_range(input("Enter the range for fixed costs for substations (e.g., '5000 8000'): "))
    substation_capacity_range = validate_range(input("Enter the range for substation capacity (e.g., '1000 1500'): "))
    demand_range = validate_range(input("Enter the range for base demand (e.g., '100 200'): "))
    lambda_value = float(input("Enter the heat loss rate per unit distance (e.g., '0.01'): "))
    if lambda_value < 0:
        raise ValueError("Lambda value must be non-negative.")
    return (num_instances, num_facilities_range, num_substations_range, num_customers_range,
            production_cost_range, production_capacity_range, fixed_cost_range,
            substation_capacity_range, demand_range, lambda_value)

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def generate_instances_from_file(input_file):
    """Generate problem instances based on configurations from an input file."""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue  # Skip comments and empty lines

        parts = line.split(',')
        if len(parts) < 5:
            print(f"Invalid line format: {line}")
            continue

        # Extract parameters
        instance_num = int(parts[0])
        num_facilities = int(parts[1])
        num_substations = int(parts[2])
        num_customers = int(parts[3])
        difficulty = parts[4].lower()

        # Additional parameters can be extracted as needed

        # Generate instance
        print(f"Generating instance {instance_num} with difficulty '{difficulty}'")

        # Set parameter ranges based on difficulty
        if difficulty == 'easy':
            # Parameters for an easy instance
            production_cost_range = (20, 40)
            production_capacity_range = (2000, 3000)
            fixed_cost_range = (5000, 8000)
            substation_capacity_range = (1500, 2000)
            demand_range = (100, 200)
            lambda_value = 0.01
        elif difficulty == 'medium':
            # Parameters for a medium instance
            production_cost_range = (30, 50)
            production_capacity_range = (1500, 2500)
            fixed_cost_range = (6000, 9000)
            substation_capacity_range = (1200, 1800)
            demand_range = (150, 250)
            lambda_value = 0.02
        elif difficulty == 'hard':
            # Parameters for a hard instance
            production_cost_range = (40, 60)
            production_capacity_range = (1000, 2000)
            fixed_cost_range = (7000, 10000)
            substation_capacity_range = (1000, 1500)
            demand_range = (200, 300)
            lambda_value = 0.03
        elif difficulty == 'infeasible':
            # Parameters designed to create an infeasible instance
            production_cost_range = (20, 40)
            production_capacity_range = (500, 800)
            fixed_cost_range = (5000, 8000)
            substation_capacity_range = (500, 800)
            demand_range = (1000, 1500)  # High demand
            lambda_value = 0.01
        else:
            print(f"Unknown difficulty level: {difficulty}")
            continue

        # Generate the problem instance using the existing logic

        data_directory = f'scenario_{instance_num}'
        os.makedirs(data_directory, exist_ok=True)

        # Sets
        I = [f'Plant{i+1}' for i in range(num_facilities)]
        J = [f'Sub{j+1}' for j in range(num_substations)]
        K = [f'City{k+1}' for k in range(num_customers)]
        S = ['Winter', 'Summer', 'Fall', 'Spring']

        # Generate production costs PC[i][s]
        production_cost_data = {i: {} for i in I}
        for i in I:
            base_cost = random.uniform(*production_cost_range)
            seasonal_variation = {s: random.uniform(0.9, 1.1) for s in S}
            for s in S:
                production_cost_data[i][s] = base_cost * seasonal_variation[s]

        # Generate production capacities Cap[i][s]
        production_capacity_data = {i: {} for i in I}
        for i in I:
            base_capacity = random.uniform(*production_capacity_range)
            seasonal_variation = {s: random.uniform(0.9, 1.0) for s in S}
            for s in S:
                production_capacity_data[i][s] = base_capacity * seasonal_variation[s]

        # Generate fixed costs FC_j for substations
        fixed_cost_data = {}
        for j in J:
            fixed_cost_data[j] = random.uniform(*fixed_cost_range)

        # Generate substation capacities SC[j][s]
        substation_capacity_data = {j: {} for j in J}
        for j in J:
            base_capacity = random.uniform(*substation_capacity_range)
            for s in S:
                substation_capacity_data[j][s] = base_capacity * random.uniform(0.95, 1.05)

        # Generate demands D[k][s]
        demand_data = {k: {} for k in K}
        for k in K:
            base_demand = random.uniform(*demand_range)
            seasonal_variation = {
                'Winter': random.uniform(1.1, 1.3),
                'Summer': random.uniform(0.7, 0.9),
                'Fall': random.uniform(0.9, 1.1),
                'Spring': random.uniform(0.9, 1.1),
            }
            for s in S:
                demand_data[k][s] = base_demand * seasonal_variation[s]

        # Generate distances between facilities and substations d_ij
        distance_ij_data = {i: {} for i in I}
        for i in I:
            for j in J:
                distance_ij_data[i][j] = random.uniform(5, 15)

        # Generate distances between substations and customers d_jk
        distance_jk_data = {j: {} for j in J}
        for j in J:
            for k in K:
                distance_jk_data[j][k] = random.uniform(1, 10)

        # Calculate heat loss coefficients alpha_ij and beta_jk
        alpha_data = {i: {} for i in I}
        for i in I:
            for j in J:
                d = distance_ij_data[i][j]
                alpha_data[i][j] = np.exp(-lambda_value * d)

        beta_data = {j: {} for j in J}
        for j in J:
            for k in K:
                d = distance_jk_data[j][k]
                beta_data[j][k] = np.exp(-lambda_value * d)

        # Calculate Big M values for constraints
        M_value = {j: {} for j in J}
        for j in J:
            for k in K:
                beta_jk = beta_data[j][k]
                max_demand = max(demand_data[k][s] for s in S)
                M_value[j][k] = max_demand / (beta_jk * 0.9)

        # Prepare data for JSON serialization
        data = {
            'I': I,
            'J': J,
            'K': K,
            'S': S,
            'PC': production_cost_data,
            'Cap': production_capacity_data,
            'FC': fixed_cost_data,
            'SC': substation_capacity_data,
            'D': demand_data,
            'alpha': alpha_data,
            'beta': beta_data,
            'M': M_value,
        }

        # Save data to a JSON file
        filename = f'problem_instance_{instance_num}.json'
        filepath = os.path.join(data_directory, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data for Problem Instance {instance_num} saved to {filepath}\n")

def main():
    # Set a random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)

    # Get user input
    try:
        (num_instances, num_facilities_range, num_substations_range, num_customers_range,
        production_cost_range, production_capacity_range, fixed_cost_range,
        substation_capacity_range, demand_range, lambda_value) = get_user_input()
    except ValueError as e:
        print(e)
        return

    # Generate data for each instance
    for instance_num in range(1, num_instances + 1):
        data_directory = f'scenario_{instance_num}'
        os.makedirs(data_directory, exist_ok=True)

        # Generate random counts for entities
        num_facilities = random.randint(*num_facilities_range)
        num_substations = random.randint(*num_substations_range)
        num_customers = random.randint(*num_customers_range)

        # Generate sets
        I = [f'Plant{i+1}' for i in range(num_facilities)]
        J = [f'Sub{j+1}' for j in range(num_substations)]
        K = [f'City{k+1}' for k in range(num_customers)]
        S = ['Winter', 'Summer', 'Fall', 'Spring']

        # Generate production costs PC[i][s]
        production_cost_data = {i: {} for i in I}
        for i in I:
            base_cost = random.uniform(*production_cost_range)
            seasonal_variation = {s: random.uniform(0.9, 1.1) for s in S}
            for s in S:
                production_cost_data[i][s] = base_cost * seasonal_variation[s]

        # Generate production capacities Cap[i][s]
        production_capacity_data = {i: {} for i in I}
        for i in I:
            base_capacity = random.uniform(*production_capacity_range)
            seasonal_variation = {s: random.uniform(0.9, 1.0) for s in S}
            for s in S:
                production_capacity_data[i][s] = base_capacity * seasonal_variation[s]

        # Generate fixed costs FC_j for substations
        fixed_cost_data = {}
        for j in J:
            fixed_cost_data[j] = random.uniform(*fixed_cost_range)

        # Generate substation capacities SC[j][s]
        substation_capacity_data = {j: {} for j in J}
        for j in J:
            base_capacity = random.uniform(*substation_capacity_range)
            for s in S:
                substation_capacity_data[j][s] = base_capacity * random.uniform(0.95, 1.05)

        # Generate demands D[k][s]
        demand_data = {k: {} for k in K}
        for k in K:
            base_demand = random.uniform(*demand_range)
            seasonal_variation = {
                'Winter': random.uniform(1.1, 1.3),
                'Summer': random.uniform(0.7, 0.9),
                'Fall': random.uniform(0.9, 1.1),
                'Spring': random.uniform(0.9, 1.1),
            }
            for s in S:
                demand_data[k][s] = base_demand * seasonal_variation[s]

        # Generate distances between facilities and substations d_ij
        distance_ij_data = {i: {} for i in I}
        for i in I:
            for j in J:
                distance_ij_data[i][j] = random.uniform(5, 15)

        # Generate distances between substations and customers d_jk
        distance_jk_data = {j: {} for j in J}
        for j in J:
            for k in K:
                distance_jk_data[j][k] = random.uniform(1, 10)

        # Calculate heat loss coefficients alpha_ij and beta_jk
        alpha_data = {i: {} for i in I}
        for i in I:
            for j in J:
                d = distance_ij_data[i][j]
                alpha_data[i][j] = np.exp(-lambda_value * d)

        beta_data = {j: {} for j in J}
        for j in J:
            for k in K:
                d = distance_jk_data[j][k]
                beta_data[j][k] = np.exp(-lambda_value * d)

        # Calculate Big M values for constraints
        M_value = {j: {} for j in J}
        for j in J:
            for k in K:
                beta_jk = beta_data[j][k]
                max_demand = max(demand_data[k][s] for s in S)
                M_value[j][k] = max_demand / (beta_jk * 0.9)

        # Prepare data for JSON serialization
        data = {
            'I': I,
            'J': J,
            'K': K,
            'S': S,
            'PC': production_cost_data,
            'Cap': production_capacity_data,
            'FC': fixed_cost_data,
            'SC': substation_capacity_data,
            'D': demand_data,
            'alpha': alpha_data,
            'beta': beta_data,
            'M': M_value,
        }

        # Save data to a JSON file
        filename = f'problem_instance_{instance_num}.json'
        filepath = os.path.join(data_directory, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data for Problem Instance {instance_num} saved to {filepath}\n")

if __name__ == "__main__":
    # To generate instances based on user input:
    # main()

    # To generate instances from a configuration file:
    generate_instances_from_file('input_configurations.txt')
