import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import math

class ThermalNetworkDataGenerator:
    def __init__(self, 
                 num_facilities: int,
                 num_substations: int,
                 num_customers: int,
                 grid_size: int = 100,
                 seed: int = 42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        num_facilities: int
            Number of production facilities
        num_substations: int
            Number of potential substation locations
        num_customers: int
            Number of customers
        grid_size: int
            Size of the square grid for location generation
        seed: int
            Random seed for reproducibility
        """
        self.num_facilities = num_facilities
        self.num_substations = num_substations
        self.num_customers = num_customers
        self.grid_size = grid_size
        self.seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate locations
        self.locations = self._generate_locations()
        
        # Calculate distances
        self.distances = self._calculate_distances()

    def _generate_locations(self) -> Dict[str, Tuple[float, float]]:
        """Generate random locations for facilities, substations, and customers."""
        locations = {}
        
        # Generate facility locations (preferably on the edges of the grid)
        for i in range(self.num_facilities):
            if random.random() < 0.5:
                x = random.choice([0, self.grid_size])
                y = random.uniform(0, self.grid_size)
            else:
                x = random.uniform(0, self.grid_size)
                y = random.choice([0, self.grid_size])
            locations[f'Plant{i+1}'] = (x, y)
        
        # Generate substation locations
        for j in range(self.num_substations):
            while True:
                x = random.uniform(0, self.grid_size)
                y = random.uniform(0, self.grid_size)
                new_loc = (x, y)
                # Ensure minimum distance from other substations
                if all(math.dist(new_loc, loc) > self.grid_size/math.sqrt(self.num_substations) 
                       for loc in locations.values()):
                    locations[f'Sub{j+1}'] = new_loc
                    break
        
        # Generate customer locations (cluster them somewhat)
        num_clusters = min(3, self.num_customers)
        cluster_centers = [
            (random.uniform(20, self.grid_size-20),
             random.uniform(20, self.grid_size-20))
            for _ in range(num_clusters)
        ]
        
        for k in range(self.num_customers):
            center = random.choice(cluster_centers)
            x = np.random.normal(center[0], self.grid_size/10)
            y = np.random.normal(center[1], self.grid_size/10)
            x = max(0, min(self.grid_size, x))
            y = max(0, min(self.grid_size, y))
            locations[f'City{k+1}'] = (x, y)
            
        return locations

    def _calculate_distances(self) -> Dict[str, Dict[str, float]]:
        """Calculate distances between all points."""
        distances = {}
        for name1, loc1 in self.locations.items():
            distances[name1] = {}
            for name2, loc2 in self.locations.items():
                distances[name1][name2] = math.dist(loc1, loc2)
        return distances

    def generate_production_costs(self) -> Dict[Tuple[str, str], float]:
        """Generate production costs based on distances and base costs."""
        production_cost = {}
        base_cost = random.uniform(40, 60)
        
        for i in range(self.num_facilities):
            facility = f'Plant{i+1}'
            for j in range(self.num_substations):
                substation = f'Sub{j+1}'
                # Cost increases with distance
                distance_factor = self.distances[facility][substation] / self.grid_size
                production_cost[(facility, substation)] = base_cost * (1 + 0.5 * distance_factor)
        
        return production_cost

    def generate_fixed_costs(self) -> Dict[str, float]:
        """Generate fixed costs for substations."""
        return {f'Sub{j+1}': random.uniform(800, 1500) 
                for j in range(self.num_substations)}

    def generate_capacities(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Generate capacities for facilities and substations."""
        # Facility capacities
        total_winter_demand = self.num_customers * 300  # Approximate peak demand
        facility_capacity = {}
        for i in range(self.num_facilities):
            facility_capacity[f'Plant{i+1}'] = (total_winter_demand / 
                                              self.num_facilities) * random.uniform(1.2, 1.5)

        # Substation capacities
        substation_capacity = {}
        for j in range(self.num_substations):
            substation_capacity[f'Sub{j+1}'] = (total_winter_demand / 
                                              self.num_substations) * random.uniform(0.8, 1.2)
            
        return facility_capacity, substation_capacity

    def generate_demand(self) -> Dict[Tuple[str, str], float]:
        """Generate seasonal demand for customers."""
        demand = {}
        season_factors = {
            'Winter': 1.0,
            'Summer': 0.2,
            'Fall': 0.6,
            'Spring': 0.6
        }
        
        for k in range(self.num_customers):
            base_demand = random.uniform(200, 300)
            for season in self.seasons:
                demand[(f'City{k+1}', season)] = base_demand * season_factors[season]
        
        return demand

    def generate_heat_loss_coefficients(self) -> Tuple[Dict[Tuple[str, str], float], 
                                                      Dict[Tuple[str, str], float]]:
        """Generate heat loss coefficients based on distances."""
        alpha = {}  # Facility to Substation
        beta = {}   # Substation to Customer
        
        # Base heat loss coefficient
        lambda_coef = 0.001
        
        # Facility to Substation coefficients
        for i in range(self.num_facilities):
            for j in range(self.num_substations):
                facility = f'Plant{i+1}'
                substation = f'Sub{j+1}'
                distance = self.distances[facility][substation]
                alpha[(facility, substation)] = math.exp(-lambda_coef * distance)
        
        # Substation to Customer coefficients
        for j in range(self.num_substations):
            for k in range(self.num_customers):
                substation = f'Sub{j+1}'
                customer = f'City{k+1}'
                distance = self.distances[substation][customer]
                beta[(substation, customer)] = math.exp(-lambda_coef * distance)
        
        return alpha, beta

    def generate_all_data(self) -> Dict:
        """Generate all necessary data for the model."""
        facility_capacity, substation_capacity = self.generate_capacities()
        alpha, beta = self.generate_heat_loss_coefficients()
        
        data = {
            'production_cost': self.generate_production_costs(),
            'fixed_cost': self.generate_fixed_costs(),
            'facility_capacity': facility_capacity,
            'substation_capacity': substation_capacity,
            'demand': self.generate_demand(),
            'alpha': alpha,
            'beta': beta,
            'locations': self.locations
        }
        
        return data

    def visualize_network(self):
        """Visualize the network using matplotlib."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot facilities
        facilities_x = [loc[0] for name, loc in self.locations.items() if 'Plant' in name]
        facilities_y = [loc[1] for name, loc in self.locations.items() if 'Plant' in name]
        plt.scatter(facilities_x, facilities_y, c='red', s=100, label='Facilities')
        
        # Plot substations
        substations_x = [loc[0] for name, loc in self.locations.items() if 'Sub' in name]
        substations_y = [loc[1] for name, loc in self.locations.items() if 'Sub' in name]
        plt.scatter(substations_x, substations_y, c='blue', s=100, label='Substations')
        
        # Plot customers
        customers_x = [loc[0] for name, loc in self.locations.items() if 'City' in name]
        customers_y = [loc[1] for name, loc in self.locations.items() if 'City' in name]
        plt.scatter(customers_x, customers_y, c='green', s=100, label='Customers')
        
        plt.grid(True)
        plt.legend()
        plt.title('Thermal Network Layout')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        return plt

# Example usage:
if __name__ == "__main__":
    # Create generator
    generator = ThermalNetworkDataGenerator(
        num_facilities=2,
        num_substations=3,
        num_customers=5,
        grid_size=100
    )
    
    # Generate data
    data = generator.generate_all_data()
    
    # Print sample of generated data
    print("\nSample of generated data:")
    print("\nProduction Costs (first 2):")
    for i, (k, v) in enumerate(data['production_cost'].items()):
        if i < 2:
            print(f"{k}: {v:.2f}")
            
    print("\nFixed Costs (first 2):")
    for i, (k, v) in enumerate(data['fixed_cost'].items()):
        if i < 2:
            print(f"{k}: {v:.2f}")
    
    # Visualize network
    plt = generator.visualize_network()
    plt.show()