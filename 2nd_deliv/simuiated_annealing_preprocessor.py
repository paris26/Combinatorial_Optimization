import random
import math
import copy
from typing import Dict, List, Tuple, Optional


class SimulatedAnnealingPreprocessor:
    def __init__(self, optimizer):
        """Initialize the simulated annealing preprocessor"""
        self.optimizer = optimizer
        self.best_solution = None
        self.best_cost = float('inf')

    def generate_initial_solution(self) -> Dict:
        """Generate an initial feasible solution"""
        num_substations = self.optimizer.generator.num_substations
        num_customers = self.optimizer.generator.num_customers

        # Start by opening about half of the substations
        opened_substations = random.sample(
            [f'Sub{j + 1}' for j in range(num_substations)],
            max(1, num_substations // 2)
        )

        # Create initial y variables (substation opening decisions)
        y = {f'Sub{j + 1}': 1.0 if f'Sub{j + 1}' in opened_substations else 0.0
             for j in range(num_substations)}

        # Create initial x variables (customer assignments)
        x = {}
        for k in range(num_customers):
            customer = f'City{k + 1}'
            # Assign each customer to the nearest open substation
            assigned_sub = min(
                opened_substations,
                key=lambda sub: self.optimizer.generator.distances[sub][customer]
            )
            for j in range(num_substations):
                substation = f'Sub{j + 1}'
                x[(substation, customer)] = 1.0 if substation == assigned_sub else 0.0

        return {'y': y, 'x': x}

    def evaluate_solution(self, solution: Dict) -> float:
        """Evaluate solution cost and feasibility"""
        try:
            # Create model with fixed variables
            model = self.optimizer.create_model(
                relaxed=True,
                fixed_vars={
                               ('y', j): val for j, val in solution['y'].items()
                           } | {
                               ('x', (j, k)): val for (j, k), val in solution['x'].items()
                           }
            )

            # Solve to get flows and objective
            solver_results = self.optimizer.solve_node(model)
            if solver_results[0]:  # If feasible
                return solver_results[1]  # Return objective value
            return float('inf')
        except:
            return float('inf')

    def get_neighbor(self, current_solution: Dict) -> Dict:
        """Generate a neighboring solution with small changes"""
        neighbor = copy.deepcopy(current_solution)

        # Randomly choose between different neighborhood moves
        move_type = random.random()

        if move_type < 0.3:  # Open/close a substation
            sub_to_flip = random.choice(list(neighbor['y'].keys()))
            neighbor['y'][sub_to_flip] = 1.0 - neighbor['y'][sub_to_flip]

            # If we closed a substation, reassign its customers
            if neighbor['y'][sub_to_flip] < 0.5:
                open_subs = [j for j, val in neighbor['y'].items() if val > 0.5]
                if open_subs:  # Ensure we have at least one open substation
                    for k in [f'City{k + 1}' for k in range(self.optimizer.generator.num_customers)]:
                        if neighbor['x'][(sub_to_flip, k)] > 0.5:
                            # Assign to random open substation
                            new_sub = random.choice(open_subs)
                            for j in neighbor['y'].keys():
                                neighbor['x'][(j, k)] = 1.0 if j == new_sub else 0.0

        else:  # Reassign a customer
            customer = f'City{random.randrange(self.optimizer.generator.num_customers) + 1}'
            current_sub = next(j for j, k in neighbor['x'].keys()
                               if k == customer and neighbor['x'][(j, k)] > 0.5)

            # Find a new open substation
            open_subs = [j for j, val in neighbor['y'].items()
                         if val > 0.5 and j != current_sub]
            if open_subs:
                new_sub = random.choice(open_subs)
                for j in neighbor['y'].keys():
                    neighbor['x'][(j, customer)] = 1.0 if j == new_sub else 0.0

        return neighbor

    def run(self, initial_temp=100.0, final_temp=1.0, cooling_rate=0.95,
            iterations_per_temp=100) -> Tuple[Dict, float]:
        """Run simulated annealing to find a good initial solution"""
        print("\nStarting Simulated Annealing Preprocessing...")

        # Generate initial solution
        current_solution = self.generate_initial_solution()
        current_cost = self.evaluate_solution(current_solution)

        self.best_solution = current_solution
        self.best_cost = current_cost

        temp = initial_temp
        while temp > final_temp:
            for _ in range(iterations_per_temp):
                # Generate neighbor
                neighbor = self.get_neighbor(current_solution)
                neighbor_cost = self.evaluate_solution(neighbor)

                # Calculate acceptance probability
                delta = neighbor_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_solution = neighbor
                    current_cost = neighbor_cost

                    # Update best solution if improved
                    if current_cost < self.best_cost:
                        self.best_solution = current_solution
                        self.best_cost = current_cost
                        print(f"New best solution found: {self.best_cost:.2f}")

            # Cool down
            temp *= cooling_rate
            print(f"Temperature: {temp:.2f}, Current best: {self.best_cost:.2f}")

        print(f"Simulated annealing complete. Best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost