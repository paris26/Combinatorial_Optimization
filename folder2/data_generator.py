import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math
import warnings

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
        # Input validation
        self._validate_inputs(num_facilities, num_substations, num_customers, grid_size)
        
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
        
        # Check feasibility indicators
        self._check_feasibility_indicators()

    def _validate_inputs(self, num_facilities: int, num_substations: int, 
                        num_customers: int, grid_size: int):
        """Validate input parameters"""
        if num_facilities <= 0:
            raise ValueError("Number of facilities must be positive")
        if num_substations <= 0:
            raise ValueError("Number of substations must be positive")
        if num_customers <= 0:
            raise ValueError("Number of customers must be positive")
        if grid_size <= 0:
            raise ValueError("Grid size must be positive")
            
        # Warning for potentially problematic configurations
        if num_substations < math.ceil(num_customers / 5):
            warnings.warn(
                f"Number of substations ({num_substations}) might be too low "
                f"for the number of customers ({num_customers})"
            )
        if num_facilities < math.ceil(num_customers / 10):
            warnings.warn(
                f"Number of facilities ({num_facilities}) might be too low "
                f"for the number of customers ({num_customers})"
            )

    def _check_feasibility_indicators(self):
        """Check and report potential feasibility issues"""
        # Calculate average distances
        avg_facility_to_sub = np.mean([
            self.distances[f'Plant{i+1}'][f'Sub{j+1}']
            for i in range(self.num_facilities)
            for j in range(self.num_substations)
        ])
        
        avg_sub_to_customer = np.mean([
            self.distances[f'Sub{j+1}'][f'City{k+1}']
            for j in range(self.num_substations)
            for k in range(self.num_customers)
        ])
        
        # Check for potential issues
        if avg_facility_to_sub > self.grid_size / 2:
            warnings.warn(
                "Average facility-to-substation distance is high, "
                "might lead to significant heat losses"
            )
        
        if avg_sub_to_customer > self.grid_size / 3:
            warnings.warn(
                "Average substation-to-customer distance is high, "
                "might lead to significant heat losses"
            )

    def _generate_locations(self) -> Dict[str, Tuple[float, float]]:
        """Generate random locations for facilities, substations, and customers."""
        locations = {}
        max_attempts = 1000  # Prevent infinite loops
        
        # Generate facility locations (preferably on the edges of the grid)
        for i in range(self.num_facilities):
            if random.random() < 0.5:
                x = random.choice([0, self.grid_size])
                y = random.uniform(0, self.grid_size)
            else:
                x = random.uniform(0, self.grid_size)
                y = random.choice([0, self.grid_size])
            locations[f'Plant{i+1}'] = (x, y)
        
        # Generate substation locations with minimum separation
        for j in range(self.num_substations):
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(0, self.grid_size)
                y = random.uniform(0, self.grid_size)
                new_loc = (x, y)
                
                # Ensure minimum distance from other substations
                min_distance = self.grid_size/math.sqrt(self.num_substations)
                if all(math.dist(new_loc, loc) > min_distance 
                       for name, loc in locations.items() if 'Sub' in name):
                    locations[f'Sub{j+1}'] = new_loc
                    break
                attempts += 1
            
            if attempts == max_attempts:
                warnings.warn(f"Could not find suitable location for Sub{j+1} "
                            f"with minimum separation {min_distance}")
                # Place it anyway
                locations[f'Sub{j+1}'] = (x, y)
        
        # Generate customer locations (clustered)
        num_clusters = min(3, self.num_customers)
        cluster_centers = []
        
        # Generate well-separated cluster centers
        for _ in range(num_clusters):
            attempts = 0
            while attempts < max_attempts:
                center = (random.uniform(20, self.grid_size-20),
                         random.uniform(20, self.grid_size-20))
                if all(math.dist(center, c) > self.grid_size/4 for c in cluster_centers):
                    cluster_centers.append(center)
                    break
                attempts += 1
                
            if attempts == max_attempts:
                # If we can't find a well-separated center, just use what we have
                cluster_centers.append((
                    random.uniform(20, self.grid_size-20),
                    random.uniform(20, self.grid_size-20)
                ))
        
        # Generate customer locations around clusters
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
        # different distances ( euclidean -> factory-sub , manhattan -> sub-city)
    # def _calculate_distances(self) -> Dict[str, Dict[str, float]]:
    # """Calculate distances with different metrics based on node type."""
    # distances = {}
    # for name1, loc1 in self.locations.items():
    #     distances[name1] = {}
    #     for name2, loc2 in self.locations.items():
    #         if 'Plant' in name1 and 'Sub' in name2:
    #             # Euclidean for plant-to-substation
    #             distances[name1][name2] = math.dist(loc1, loc2)
    #         elif 'Sub' in name1 and 'City' in name2:
    #             # Manhattan for substation-to-customer
    #             distances[name1][name2] = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    #         else:
    #             # Default to Euclidean for other cases
    #             distances[name1][name2] = math.dist(loc1, loc2)
    # return distances
    
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
        # Base cost varies by number of customers to serve
        base_cost = 800 + (self.num_customers * 50)  # Scale with problem size
        return {
            f'Sub{j+1}': random.uniform(base_cost, base_cost * 1.5) 
            for j in range(self.num_substations)
        }

    def generate_capacities(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Generate capacities for facilities and substations."""
        # Calculate total winter demand
        total_winter_demand = self.num_customers * 300  # Base demand
        
        # Facility capacities - ensure total capacity exceeds maximum demand
        facility_capacity = {}
        min_facility_capacity = total_winter_demand / self.num_facilities * 1.2
        
        for i in range(self.num_facilities):
            facility_capacity[f'Plant{i+1}'] = min_facility_capacity * random.uniform(1.0, 1.3)

        # Substation capacities
        substation_capacity = {}
        min_substation_capacity = total_winter_demand / self.num_substations
        
        for j in range(self.num_substations):
            substation_capacity[f'Sub{j+1}'] = min_substation_capacity * random.uniform(0.8, 1.2)
            
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
        
        # Scale base demand with problem size
        base_demand_min = 200 + (self.num_customers * 2)  # Larger problems have higher per-customer demand
        base_demand_max = 300 + (self.num_customers * 3)
        
        for k in range(self.num_customers):
            base_demand = random.uniform(base_demand_min, base_demand_max)
            for season in self.seasons:
                demand[(f'City{k+1}', season)] = base_demand * season_factors[season]
        
        return demand

    def generate_heat_loss_coefficients(self) -> Tuple[Dict[Tuple[str, str], float], 
                                                      Dict[Tuple[str, str], float]]:
        """Generate heat loss coefficients based on distances."""
        alpha = {}  # Facility to Substation
        beta = {}   # Substation to Customer
        
        # Base heat loss coefficient - scales with grid size
        lambda_coef = 0.001 * (100 / self.grid_size)  # Normalize for grid size
        
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
