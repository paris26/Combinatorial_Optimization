import random
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import warnings

class RandomizedThermalNetworkGenerator:
    def __init__(self, 
                 num_facilities: int,
                 num_substations: int,
                 num_customers: int,
                 grid_size: int = 100,
                 seed: Optional[int] = None):
        """
        Initialize a fully randomized thermal network generator.
        
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
        seed: Optional[int]
            Random seed for reproducibility
        """
        self.num_facilities = num_facilities
        self.num_substations = num_substations
        self.num_customers = num_customers
        self.grid_size = grid_size
        self.seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate basic network structure
        self.locations = self._generate_locations()
        self.distances = self._calculate_distances()
    
    def _generate_locations(self) -> Dict[str, Tuple[float, float]]:
        """Generate completely random locations for all nodes."""
        locations = {}
        
        # Generate facility locations randomly anywhere
        for i in range(self.num_facilities):
            x = random.uniform(0, self.grid_size)
            y = random.uniform(0, self.grid_size)
            locations[f'Plant{i+1}'] = (x, y)
        
        # Generate substation locations
        for j in range(self.num_substations):
            x = random.uniform(0, self.grid_size)
            y = random.uniform(0, self.grid_size)
            locations[f'Sub{j+1}'] = (x, y)
        
        # Generate customer locations
        # Optionally use random clusters
        if random.random() < 0.5:  # 50% chance of clustering
            num_clusters = random.randint(1, min(5, self.num_customers))
            clusters = [(random.uniform(0, self.grid_size), 
                        random.uniform(0, self.grid_size)) 
                       for _ in range(num_clusters)]
            
            for k in range(self.num_customers):
                center = random.choice(clusters)
                spread = random.uniform(0, self.grid_size/4)
                x = np.random.normal(center[0], spread)
                y = np.random.normal(center[1], spread)
                x = max(0, min(self.grid_size, x))
                y = max(0, min(self.grid_size, y))
                locations[f'City{k+1}'] = (x, y)
        else:
            # Completely random customer locations
            for k in range(self.num_customers):
                x = random.uniform(0, self.grid_size)
                y = random.uniform(0, self.grid_size)
                locations[f'City{k+1}'] = (x, y)
        
        return locations
    
    def _calculate_distances(self) -> Dict[str, Dict[str, float]]:
        """Calculate distances using random mixture of metrics."""
        distances = {}
        for name1, loc1 in self.locations.items():
            distances[name1] = {}
            for name2, loc2 in self.locations.items():
                # Randomly choose between Euclidean and Manhattan distance
                if random.random() < 0.5:
                    # Euclidean distance
                    distances[name1][name2] = math.dist(loc1, loc2)
                else:
                    # Manhattan distance
                    distances[name1][name2] = (abs(loc1[0] - loc2[0]) + 
                                            abs(loc1[1] - loc2[1]))
        return distances
    
    def generate_production_costs(self) -> Dict[Tuple[str, str], float]:
        """Generate randomized production costs."""
        production_cost = {}
        for i in range(self.num_facilities):
            facility = f'Plant{i+1}'
            base_cost = random.uniform(20, 100)  # Wider range
            for j in range(self.num_substations):
                substation = f'Sub{j+1}'
                # Random cost variation
                cost_factor = random.uniform(0.5, 2.0)
                distance_factor = self.distances[facility][substation] / self.grid_size
                production_cost[(facility, substation)] = base_cost * (1 + cost_factor * distance_factor)
        return production_cost
    
    def generate_fixed_costs(self) -> Dict[str, float]:
        """Generate randomized fixed costs for substations."""
        base_range = (500, 2000)  # Wider range for base costs
        return {
            f'Sub{j+1}': random.uniform(*base_range) * random.uniform(0.8, 1.5)
            for j in range(self.num_substations)
        }
    
    def generate_capacities(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Generate randomized capacities."""
        # Random base capacity scaling
        base_scale = random.uniform(200, 600)
        total_demand_estimate = self.num_customers * base_scale
        
        # Facility capacities
        facility_capacity = {}
        for i in range(self.num_facilities):
            facility_capacity[f'Plant{i+1}'] = (
                total_demand_estimate / self.num_facilities * 
                random.uniform(0.5, 2.0)  # More variation
            )
        
        # Substation capacities
        substation_capacity = {}
        for j in range(self.num_substations):
            substation_capacity[f'Sub{j+1}'] = (
                total_demand_estimate / self.num_substations * 
                random.uniform(0.3, 1.8)  # More variation
            )
        
        return facility_capacity, substation_capacity
    
    def generate_demand(self) -> Dict[Tuple[str, str], float]:
        """Generate randomized seasonal demand."""
        demand = {}
        # Random season factors
        season_factors = {
            season: random.uniform(0.2, 1.0) 
            for season in self.seasons
        }
        # Ensure winter has highest demand
        season_factors['Winter'] = 1.0
        
        # Generate demands
        for k in range(self.num_customers):
            base_demand = random.uniform(100, 500)  # Wider range
            for season in self.seasons:
                demand[(f'City{k+1}', season)] = base_demand * season_factors[season]
        
        return demand
    
    def generate_heat_loss_coefficients(self) -> Tuple[Dict[Tuple[str, str], float], 
                                                      Dict[Tuple[str, str], float]]:
        """Generate randomized heat loss coefficients."""
        alpha = {}  # Facility to Substation
        beta = {}   # Substation to Customer
        
        # Random base coefficients
        lambda_base = random.uniform(0.001, 0.01)
        
        # Generate coefficients
        for i in range(self.num_facilities):
            for j in range(self.num_substations):
                facility = f'Plant{i+1}'
                substation = f'Sub{j+1}'
                distance = self.distances[facility][substation]
                # Random variation in heat loss
                lambda_var = lambda_base * random.uniform(0.5, 1.5)
                alpha[(facility, substation)] = math.exp(-lambda_var * distance)
        
        for j in range(self.num_substations):
            for k in range(self.num_customers):
                substation = f'Sub{j+1}'
                customer = f'City{k+1}'
                distance = self.distances[substation][customer]
                lambda_var = lambda_base * random.uniform(0.5, 1.5)
                beta[(substation, customer)] = math.exp(-lambda_var * distance)
        
        return alpha, beta
    
    def generate_all_data(self) -> Dict:
        """Generate all network data with random variations."""
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
        """Visualize the network."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot facilities
        facilities_x = [loc[0] for name, loc in self.locations.items() if 'Plant' in name]
        facilities_y = [loc[1] for name, loc in self.locations.items() if 'Plant' in name]
        plt.scatter(facilities_x, facilities_y, c='red', s=100, label='Facilities', marker='s')
        
        # Plot substations
        substations_x = [loc[0] for name, loc in self.locations.items() if 'Sub' in name]
        substations_y = [loc[1] for name, loc in self.locations.items() if 'Sub' in name]
        plt.scatter(substations_x, substations_y, c='blue', s=100, label='Substations', marker='^')
        
        # Plot customers
        customers_x = [loc[0] for name, loc in self.locations.items() if 'City' in name]
        customers_y = [loc[1] for name, loc in self.locations.items() if 'City' in name]
        plt.scatter(customers_x, customers_y, c='green', s=100, label='Customers', marker='o')
        
        # Add labels
        for name, (x, y) in self.locations.items():
            plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.grid(True)
        plt.legend()
        plt.title('Thermal Network Layout')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        return plt