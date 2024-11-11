# data_generator.py

import random
import numpy as np
import json
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# PARAMETERS FOR DATA GENERATION

num_instances = 5       # Number of problem instances to generate
num_facilities = 3      # Number of production facilities
num_substations = 5     # Number of substations
num_customers = 8       # Number of customers
seasons = ['Winter', 'Summer', 'Fall', 'Spring']  # Seasons

# Directory to save data files
data_directory = 'problem_instances'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

for instance_num in range(1, num_instances + 1):
    print(f"Generating data for Problem Instance {instance_num}")

    # Generate sets
    I = [f'Plant{i+1}' for i in range(num_facilities)]
    J = [f'Sub{j+1}' for j in range(num_substations)]
    K = [f'City{k+1}' for k in range(num_customers)]
    S = seasons

    # Generate production costs PC_{i,s}
    production_cost_data = {}
    for i in I:
        base_cost = random.uniform(20, 40)  # Lower base production cost
        seasonal_variation = {
            'Winter': random.uniform(0.9, 1.1),
            'Summer': random.uniform(0.9, 1.1),
            'Fall': random.uniform(0.9, 1.1),
            'Spring': random.uniform(0.9, 1.1),
        }
        for s in S:
            production_cost_data[(i, s)] = base_cost * seasonal_variation[s]

    # Generate production capacities Cap_{i,s}
    production_capacity_data = {}
    for i in I:
        base_capacity = random.uniform(1500, 2000)  # Increased base capacity
        seasonal_variation = {
            'Winter': random.uniform(0.9, 1.0),
            'Summer': random.uniform(0.9, 1.0),
            'Fall': random.uniform(0.9, 1.0),
            'Spring': random.uniform(0.9, 1.0),
        }
        for s in S:
            production_capacity_data[(i, s)] = base_capacity * seasonal_variation[s]

    # Generate fixed costs FC_j for substations
    fixed_cost_data = {}
    for j in J:
        fixed_cost_data[j] = random.uniform(5000, 8000)  # Reduced fixed costs

    # Generate substation capacities SC_{j,s}
    substation_capacity_data = {}
    for j in J:
        base_capacity = random.uniform(1000, 1500)  # Increased base capacity
        for s in S:
            substation_capacity_data[(j, s)] = base_capacity * random.uniform(0.95, 1.05)

    # Generate demands D_{k,s}
    demand_data = {}
    for k in K:
        base_demand = random.uniform(100, 200)  # Reduced base demand
        seasonal_variation = {
            'Winter': random.uniform(1.1, 1.3),   # Higher demand in Winter
            'Summer': random.uniform(0.7, 0.9),   # Lower demand in Summer
            'Fall': random.uniform(0.9, 1.1),
            'Spring': random.uniform(0.9, 1.1),
        }
        for s in S:
            demand_data[(k, s)] = base_demand * seasonal_variation[s]

    # Generate distances between facilities and substations d_{ij}
    distance_ij_data = {}
    for i in I:
        for j in J:
            distance_ij_data[(i, j)] = random.uniform(5, 15)  # Reduced distances

    # Generate distances between substations and customers d_{jk}
    distance_jk_data = {}
    for j in J:
        for k in K:
            distance_jk_data[(j, k)] = random.uniform(1, 10)  # Reduced distances

    # Heat loss rate per unit distance (lambda)
    lambda_value = 0.01  # Reduced lambda to decrease heat loss impact

    # Calculate heat loss coefficients alpha_{ij} and beta_{jk}
    alpha_data = {}
    for i in I:
        for j in J:
            d = distance_ij_data[(i, j)]
            alpha_data[(i, j)] = np.exp(-lambda_value * d)

    beta_data = {}
    for j in J:
        for k in K:
            d = distance_jk_data[(j, k)]
            beta_data[(j, k)] = np.exp(-lambda_value * d)

    # Calculate Big M values for constraints
    M_value = {}
    for j in J:
        for k in K:
            beta_jk = beta_data[(j, k)]
            max_demand = max(demand_data[(k, s)] for s in S)
            M_value[(j, k)] = max_demand / (beta_jk * 0.9)  # Added safety factor

    # Convert tuple keys to strings for JSON serialization
    def convert_keys_to_strings(data_dict):
        return {str(key): value for key, value in data_dict.items()}

    # Prepare data for JSON serialization
    data = {
        'I': I,
        'J': J,
        'K': K,
        'S': S,
        'PC': convert_keys_to_strings(production_cost_data),
        'Cap': convert_keys_to_strings(production_capacity_data),
        'FC': fixed_cost_data,
        'SC': convert_keys_to_strings(substation_capacity_data),
        'D': convert_keys_to_strings(demand_data),
        'alpha': convert_keys_to_strings(alpha_data),
        'beta': convert_keys_to_strings(beta_data),
        'M': convert_keys_to_strings(M_value),
    }

    # Save data to a JSON file
    filename = f'problem_instance_{instance_num}.json'
    filepath = os.path.join(data_directory, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data for Problem Instance {instance_num} saved to {filepath}\n")
