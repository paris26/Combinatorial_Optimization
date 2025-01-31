from typing import Dict, Tuple, Optional
import random
import copy
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pyomo.environ as pyo


class GurobiPreprocessor:
    def __init__(self, optimizer, num_instances=10, max_iterations=100):
        """
        Initialize preprocessor with optimizer and parameters.

        Args:
            optimizer: The thermal network optimizer
            num_instances: Number of random instances to generate and solve
            max_iterations: Number of iterations to improve the best solution
        """
        self.optimizer = optimizer
        self.num_instances = num_instances
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_cost = float('inf')

    def solve_instance(self) -> Tuple[Optional[Dict], float]:
        """Solve a single instance using Gurobi"""
        try:
            # Create model for the instance
            model = self.optimizer.create_model(relaxed=False)

            # Configure and run Gurobi solver
            solver = SolverFactory('gurobi')
            solver.options['TimeLimit'] = 60  # 60 second time limit per instance
            results = solver.solve(model)

            if (results.solver.status == SolverStatus.ok and
                    results.solver.termination_condition == TerminationCondition.optimal):
                solution = self.optimizer.extract_solution(model)
                return solution, pyo.value(model.objective)

            return None, float('inf')
        except Exception as e:
            print(f"Error solving instance: {str(e)}")
            return None, float('inf')

    def evaluate_solution(self, solution: Dict) -> float:
        """Evaluate a solution by solving with fixed variables"""
        try:
            # Create model with fixed variables from solution
            # Create dictionary of all fixed variables
            fixed_vars = {}
            # Add facility decisions
            fixed_vars.update({('z', i): val for i, val in solution['z'].items()})
            # Add substation decisions
            fixed_vars.update({('y', j): val for j, val in solution['y'].items()})
            # Add assignment decisions
            fixed_vars.update({('x', (j, k)): val for (j, k), val in solution['x'].items()})

            model = self.optimizer.create_model(relaxed=False, fixed_vars=fixed_vars)

            solver = SolverFactory('gurobi')
            results = solver.solve(model)

            if (results.solver.status == SolverStatus.ok and
                    results.solver.termination_condition == TerminationCondition.optimal):
                return pyo.value(model.objective)

            return float('inf')
        except Exception as e:
            print(f"Error evaluating solution: {str(e)}")
            return float('inf')

    def get_neighbor(self, solution: Dict) -> Dict:
        """Generate a neighboring solution with small modifications"""
        neighbor = copy.deepcopy(solution)

        # Randomly choose modification type
        if random.random() < 0.3:  # Modify facility openings
            facility = random.choice(list(neighbor['z'].keys()))
            neighbor['z'][facility] = 1 - neighbor['z'][facility]

        elif random.random() < 0.6:  # Modify substation openings
            substation = random.choice(list(neighbor['y'].keys()))
            neighbor['y'][substation] = 1 - neighbor['y'][substation]

            # If closing, reassign its customers
            if neighbor['y'][substation] == 0:
                affected_customers = [k for (j, k), val in neighbor['x'].items()
                                      if j == substation and val == 1]
                open_subs = [j for j, val in neighbor['y'].items()
                             if val == 1 and j != substation]
                if open_subs:
                    for customer in affected_customers:
                        new_sub = random.choice(open_subs)
                        for j in neighbor['y'].keys():
                            neighbor['x'][(j, customer)] = 1 if j == new_sub else 0

        else:  # Modify customer assignments
            x_keys = list(neighbor['x'].keys())
            if x_keys:
                j, k = random.choice(x_keys)
                if neighbor['x'][(j, k)] == 1:
                    open_subs = [s for s, val in neighbor['y'].items()
                                 if val == 1 and s != j]
                    if open_subs:
                        new_sub = random.choice(open_subs)
                        neighbor['x'][(j, k)] = 0
                        neighbor['x'][(new_sub, k)] = 1

        return neighbor

    def improve_solution(self, solution: Dict) -> Tuple[Dict, float]:
        """Try to improve a solution through local search"""
        current_solution = copy.deepcopy(solution)
        current_cost = self.evaluate_solution(current_solution)

        best_solution = current_solution
        best_cost = current_cost

        for i in range(self.max_iterations):
            if i % 100 == 0:  # Progress update every 100 iterations
                print(f"Improvement iteration {i}/{self.max_iterations}, best cost: {best_cost:.2f}")

            neighbor = self.get_neighbor(current_solution)
            neighbor_cost = self.evaluate_solution(neighbor)

            if neighbor_cost < current_cost:
                current_solution = copy.deepcopy(neighbor)
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
                    print(f"Found improved solution with cost: {best_cost:.2f}")

        return best_solution, best_cost

    def preprocess(self) -> Tuple[Optional[Dict], float]:
        """Run the full preprocessing procedure"""
        print("Starting Gurobi preprocessing...")

        # Generate and solve multiple instances
        for i in range(self.num_instances):
            print(f"\nSolving instance {i + 1}/{self.num_instances}")
            solution, cost = self.solve_instance()

            if solution and cost < self.best_cost:
                self.best_solution = solution
                self.best_cost = cost
                print(f"Found new best solution with cost: {self.best_cost:.2f}")

        if not self.best_solution:
            print("Failed to find initial solution")
            return None, float('inf')

        print(f"\nBest initial solution cost: {self.best_cost:.2f}")
        print("Starting solution improvement phase...")

        # Try to improve the best solution found
        improved_solution, improved_cost = self.improve_solution(self.best_solution)

        if improved_cost < self.best_cost:
            print(f"Improved solution found with cost: {improved_cost:.2f}")
            return improved_solution, improved_cost

        return self.best_solution, self.best_cost