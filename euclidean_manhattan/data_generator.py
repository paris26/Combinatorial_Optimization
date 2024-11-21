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
        Initialize the data generator with support for both Euclidean and Manhattan distances.
        """
        self._validate_inputs(num_facilities, num_substations, num_customers, grid_size)
        
        self.num_facilities = num_facilities
        self.num_substations = num_substations
        self.num_customers = num_customers
        self.grid_size = grid_size
        self.seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.locations = self._generate_locations()
        self.euclidean_distances = self._calculate_distances('euclidean')
        self.manhattan_distances = self._calculate_distances('manhattan')
        
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

    def _manhattan_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Manhattan distance between two points"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.dist(p1, p2)

    def _generate_locations(self) -> Dict[str, Tuple[float, float]]:
        """Generate random locations for facilities, substations, and customers."""
        locations = {}
        max_attempts = 1000
        
        # Generate facility locations (preferably on edges)
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
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(0, self.grid_size)
                y = random.uniform(0, self.grid_size)
                new_loc = (x, y)
                
                min_distance = self.grid_size/math.sqrt(self.num_substations)
                if all(self._euclidean_distance(new_loc, loc) > min_distance 
                       for name, loc in locations.items() if 'Sub' in name):
                    locations[f'Sub{j+1}'] = new_loc
                    break
                attempts += 1
            
            if attempts == max_attempts:
                locations[f'Sub{j+1}'] = (x, y)
        
        # Generate customer locations (clustered)
        num_clusters = min(3, self.num_customers)
        cluster_centers = []
        
        for _ in range(num_clusters):
            attempts = 0
            while attempts < max_attempts:
                center = (random.uniform(20, self.grid_size-20),
                         random.uniform(20, self.grid_size-20))
                if all(self._euclidean_distance(center, c) > self.grid_size/4 for c in cluster_centers):
                    cluster_centers.append(center)
                    break
                attempts += 1
                
            if attempts == max_attempts:
                cluster_centers.append((
                    random.uniform(20, self.grid_size-20),
                    random.uniform(20, self.grid_size-20)
                ))
        
        for k in range(self.num_customers):
            center = random.choice(cluster_centers)
            x = np.random.normal(center[0], self.grid_size/10)
            y = np.random.normal(center[1], self.grid_size/10)
            x = max(0, min(self.grid_size, x))
            y = max(0, min(self.grid_size, y))
            locations[f'City{k+1}'] = (x, y)
            
        return locations

    def _calculate_distances(self, metric: str = 'euclidean') -> Dict[str, Dict[str, float]]:
        """Calculate distances between all points using specified metric."""
        distances = {}
        for name1, loc1 in self.locations.items():
            distances[name1] = {}
            for name2, loc2 in self.locations.items():
                if metric == 'manhattan':
                    distances[name1][name2] = self._manhattan_distance(loc1, loc2)
                else:
                    distances[name1][name2] = self._euclidean_distance(loc1, loc2)
        return distances

    def _check_feasibility_indicators(self):
        """Check feasibility using both distance metrics"""
        metrics = [
            ('euclidean', self.euclidean_distances),
            ('manhattan', self.manhattan_distances)
        ]
        
        for metric_name, distances in metrics:
            avg_facility_to_sub = np.mean([
                distances[f'Plant{i+1}'][f'Sub{j+1}']
                for i in range(self.num_facilities)
                for j in range(self.num_substations)
            ])
            
            avg_sub_to_customer = np.mean([
                distances[f'Sub{j+1}'][f'City{k+1}']
                for j in range(self.num_substations)
                for k in range(self.num_customers)
            ])
            
            if avg_facility_to_sub > self.grid_size / 2:
                warnings.warn(
                    f"Average facility-to-substation {metric_name} distance is high"
                )
            
            if avg_sub_to_customer > self.grid_size / 3:
                warnings.warn(
                    f"Average substation-to-customer {metric_name} distance is high"
                )

    def generate_production_costs(self) -> Dict[str, Dict[Tuple[str, str], float]]:
        """Generate production costs based on both distance metrics."""
        production_cost = {'euclidean': {}, 'manhattan': {}}
        base_cost = random.uniform(40, 60)
        
        for i in range(self.num_facilities):
            facility = f'Plant{i+1}'
            for j in range(self.num_substations):
                substation = f'Sub{j+1}'
                
                euclidean_factor = self.euclidean_distances[facility][substation] / self.grid_size
                manhattan_factor = self.manhattan_distances[facility][substation] / (2 * self.grid_size)
                
                production_cost['euclidean'][(facility, substation)] = base_cost * (1 + 0.5 * euclidean_factor)
                production_cost['manhattan'][(facility, substation)] = base_cost * (1 + 0.5 * manhattan_factor)
        
        return production_cost

    def generate_heat_loss_coefficients(self) -> Dict[str, Dict[str, Dict[Tuple[str, str], float]]]:
        """Generate heat loss coefficients for both distance metrics."""
        coefficients = {
            'euclidean': {'alpha': {}, 'beta': {}},
            'manhattan': {'alpha': {}, 'beta': {}}
        }
        
        lambda_coef = 0.001 * (100 / self.grid_size)
        
        for metric in ['euclidean', 'manhattan']:
            distances = (self.euclidean_distances if metric == 'euclidean' 
                       else self.manhattan_distances)
            
            # Facility to Substation coefficients
            for i in range(self.num_facilities):
                for j in range(self.num_substations):
                    facility = f'Plant{i+1}'
                    substation = f'Sub{j+1}'
                    distance = distances[facility][substation]
                    coefficients[metric]['alpha'][(facility, substation)] = math.exp(-lambda_coef * distance)
            
            # Substation to Customer coefficients
            for j in range(self.num_substations):
                for k in range(self.num_customers):
                    substation = f'Sub{j+1}'
                    customer = f'City{k+1}'
                    distance = distances[substation][customer]
                    coefficients[metric]['beta'][(substation, customer)] = math.exp(-lambda_coef * distance)
        
        return coefficients

    def generate_fixed_costs(self) -> Dict[str, float]:
        """Generate fixed costs for substations."""
        base_cost = 800 + (self.num_customers * 50)
        return {
            f'Sub{j+1}': random.uniform(base_cost, base_cost * 1.5) 
            for j in range(self.num_substations)
        }

    def generate_capacities(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Generate capacities for facilities and substations."""
        total_winter_demand = self.num_customers * 300
        
        facility_capacity = {}
        min_facility_capacity = total_winter_demand / self.num_facilities * 1.2
        
        for i in range(self.num_facilities):
            facility_capacity[f'Plant{i+1}'] = min_facility_capacity * random.uniform(1.0, 1.3)

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
        
        base_demand_min = 200 + (self.num_customers * 2)
        base_demand_max = 300 + (self.num_customers * 3)
        
        for k in range(self.num_customers):
            base_demand = random.uniform(base_demand_min, base_demand_max)
            for season in self.seasons:
                demand[(f'City{k+1}', season)] = base_demand * season_factors[season]
        
        return demand

    def generate_all_data(self) -> Dict:
        """Generate all necessary data for both distance metrics."""
        facility_capacity, substation_capacity = self.generate_capacities()
        heat_loss_coefficients = self.generate_heat_loss_coefficients()
        
        data = {
            'production_cost': self.generate_production_costs(),
            'fixed_cost': self.generate_fixed_costs(),
            'facility_capacity': facility_capacity,
            'substation_capacity': substation_capacity,
            'demand': self.generate_demand(),
            'heat_loss_coefficients': heat_loss_coefficients,
            'locations': self.locations,
            'euclidean_distances': self.euclidean_distances,
            'manhattan_distances': self.manhattan_distances
        }
        
        return data

    def visualize_network(self):
        """Visualize the network using matplotlib."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        facilities_x = [loc[0] for name, loc in self.locations.items() if 'Plant' in name]
        facilities_y = [loc[1] for name, loc in self.locations.items() if 'Plant' in name]
        plt.scatter(facilities_x, facilities_y, c='red', s=100, label='Facilities', marker='s')
        
        substations_x = [loc[0] for name, loc in self.locations.items() if 'Sub' in name]
        substations_y = [loc[1] for name, loc in self.locations.items() if 'Sub' in name]
        plt.scatter(substations_x, substations_y, c='blue', s=100, label='Substations', marker='^')
        
        customers_x = [loc[0] for name, loc in self.locations.items() if 'City' in name]
        customers_y = [loc[1] for name, loc in self.locations.items() if 'City' in name]
        plt.scatter(customers_x, customers_y, c='green', s=100, label='Customers', marker='o')
        
        for name, (x, y) in self.locations.items():
            plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.grid(True)
        plt.legend()
        plt.title('Thermal Network Layout')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        return plt