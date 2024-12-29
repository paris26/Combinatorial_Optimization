import random
import math

def generate_instance_set(num_instances=10, seed=None):
    """
    Generate a set of thermal network instances with increasing complexity.
    
    Args:
        num_instances (int): Number of instances to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        list: List of dictionaries containing instance parameters
    """
    if seed is not None:
        random.seed(seed)
    
    instances = []
    
    # Define progression factors (observed from test files)
    facility_base = random.randint(5, 7)
    substation_multiplier = random.uniform(1.4, 1.8)
    customer_per_substation = random.uniform(2.5, 3.5)
    
    # Generate instances with increasing complexity
    for i in range(num_instances):
        # Calculate base values with some randomization
        facilities = facility_base + math.floor(i * 0.3)
        substations = math.floor(facilities * substation_multiplier * random.uniform(0.9, 1.1))
        customers = math.floor(substations * customer_per_substation * random.uniform(0.95, 1.05))
        
        # Add some noise to prevent too regular progression
        facilities += random.randint(-1, 1)
        substations += random.randint(-2, 2)
        customers += random.randint(-3, 3)
        
        # Ensure minimum values
        facilities = max(4, facilities)
        substations = max(6, substations)
        customers = max(15, customers)
        
        # Keep grid size constant as in test files
        grid_size = 100
        
        instances.append({
            'facilities': facilities,
            'substations': substations,
            'customers': customers,
            'grid_size': grid_size
        })
    
    return instances

def write_instances_file(instances, filename='instances.txt'):
    """
    Write instances to a file in the format matching test files.
    
    Args:
        instances (list): List of instance dictionaries
        filename (str): Output filename
    """
    difficulty_levels = [
        "Very Easy - Small network, many substations relative to customers",
        "Easy - Balanced small network",
        "Easy-Medium - Moderate size, good ratio",
        "Medium - Balanced network configuration",
        "Medium-Hard - Larger network, limited substations",
        "Hard - Large network, needs careful optimization",
        "Hard-Plus - Complex network configuration",
        "Very Hard - Large network, tight constraints",
        "Challenging - Complex network, critical optimization",
        "Very Challenging - Maximum complexity network"
    ]
    
    with open(filename, 'w') as f:
        # Write header
        f.write("# Instance Format: num_facilities,num_substations,num_customers,grid_size\n")
        f.write("# Difficulty increases with each instance\n")
        
        # Write instances
        for i, instance in enumerate(instances):
            difficulty = difficulty_levels[i] if i < len(difficulty_levels) else "Complex Instance"
            f.write(f"# {difficulty}\n")
            f.write(f"{instance['facilities']},{instance['substations']},"
                   f"{instance['customers']},{instance['grid_size']}\n")

def main():
    """Generate and save a new set of instances."""
    # Generate instances with random seed for reproducibility
    random_seed = random.randint(1, 10000)
    instances = generate_instance_set(num_instances=10, seed=random_seed)
    
    # Save to file
    write_instances_file(instances)
    print(f"Generated new instances with seed {random_seed}")
    print("Instances have been saved to 'instances.txt'")

if __name__ == "__main__":
    main()