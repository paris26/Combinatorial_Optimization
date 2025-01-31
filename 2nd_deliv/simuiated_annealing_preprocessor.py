import random
import math
import copy
from typing import Dict, List, Tuple, Optional

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from data_generator import ThermalNetworkDataGenerator


class SimulatedAnnealingPreprocessor:
    def __init__(self, optimizer):
        """Initialize the simulated annealing preprocessor"""
        self.optimizer = optimizer
        self.best_solution = None
        self.best_cost = float('inf')

    def generate_initial_solution(self) -> Dict:
        solution = {
               'z' : { f'Plant{i+1}': 1.0 for i in range(self.optimizer.generator.num_facilities)},
               'y' : { f'Sub{j+1}' : 1.0 for j in range(self.optimizer.generator.num_substations)},
               'x' : {}
        }

        total_demand = sum(self.optimizer.data['demand'].values())
        facility_capacities = self.optimizer.data['facility_capacity']
        substation_capacities = self.optimizer.data['substation_capacity']

        #but we need to assign the customers to the closest substation as initial assignment
        # we take the customers
        customers = {f'City{k+1}' for k in range(self.optimizer.generator.num_customers)}
        # for each customer we find the nearest substation
        for customer in customers:
            nearest_sub = min(solution['y'].keys(), key=lambda s: self.optimizer.generator.distances[s][customer])
            for sub in solution['y'].keys():
                solution['x'][(sub,customer)] = 1.0 if sub == nearest_sub else 0.0

        return solution

    def evaluate_solution(self, solution, debug=False):
        try:
            #recreate the model with fixed parameters
            model = self.optimizer.create_model(relaxed=True, fixed_vars={
                ('z', i): val for i,val in solution['z'].items()
            } | {
                ('y', j) : val for j, val in solution['y'].items()
            } | {
                ('x', (j,k)): val for (j,k) ,val in solution['x'].items()
            })

            if model is None:
                if debug: print("Failed to create model")
                return float('inf')

            solver = SolverFactory('gurobi')
            solver.options['FeasibilityTol'] = 1e-6
            solver.options['OptimalityTol'] = 1e-6
            results = solver.solve(model, tee=False)

            if(results.solver.status == SolverStatus.ok and
                results.solver.termination_condition == TerminationCondition.optimal):
                return pyo.value(model.objective)

            if debug:
                print(f"Solver Status: {results.solver.status}")
                print(f"Termination Condition : {results.solver.termination_condition}")
            return float('inf')
        except Exception as e:
            if debug: print(f"Error evaluating solution : {str(e)}")
            return float('inf')

    def get_neighbor(self, current_solution):
        neighbor = copy.deepcopy(current_solution)
        move_type = random.random()

        #we get a random number to choose our next options
        if move_type<0.2:
            opened_facilities = [i for i, val in neighbor['z'].items() if val > 0.5]
            closed_facilities = [i for i, val in neighbor['z'].items() if val < 0.5]

            if random.random() < 0.5 and len(opened_facilities) > 1 :
                facility_to_close = random.choice(opened_facilities)
                neighbor['z'][facility_to_close] = 0.0
            elif closed_facilities:
                facility_to_open = random.choice(closed_facilities)
                neighbor['z'][facility_to_open] = 1.0
        elif move_type < 0.4 :
            opened_subs = [j for j, val in neighbor['y'].items() if val > 0.5 ]
            closed_subs = [j for j, val in neighbor['y'].items() if val < 0.5 ]

            if random.random() < 0.5 and len(opened_subs) > 1:
                sub_to_close = random.choice(opened_subs)
                neighbor['y'][sub_to_close] = 0.0

                affected_customers = [k for k in self.optimizer.data['demand'].keys() if neighbor['x'][(sub_to_close, k[0])] > 0.5]
                remaining_subs = [ s for s in opened_subs if s != sub_to_close]

                for k in affected_customers:
                    best_sub = min(remaining_subs, key= lambda s: self.optimizer.generator.distances[s][k[0]])
                    for j in neighbor['y'].keys():
                        neighbor['x'][(j, k[0])] = 1.0 if j == best_sub else 0.0
            elif closed_subs:
                sub_to_open = random.choice(closed_subs)
                neighbor['y'][sub_to_open] = 1.0
        else:
            customer = f'City{random.randrange(self.optimizer.generator.num_customers) + 1}'
            opened_subs = [j for j, val in neighbor['y'].items() if val > 0.5 ]

            if opened_subs:
                new_sub = random.choice(opened_subs)
                for j in neighbor['y'].keys():
                    neighbor['x'][(j, customer)] = 1.0 if j == new_sub else 0.0

        return neighbor



    def run(self, initial_temp, final_temp=1.0, cooling_rate = 0.95, iterations_per_temp=100, max_non_improving = 1000):
        current_solution = self.generate_initial_solution()
        current_cost = self.evaluate_solution(current_solution, debug=True)

        if current_cost == float('inf'):
            print("Failed to generate feasible initial solution")
            return None, float('inf')

        self.best_solution = current_solution
        self.best_cost = current_cost

        temp = initial_temp
        non_improving = 0

        while temp > 0 and non_improving < max_non_improving :
            for _ in range(iterations_per_temp):
                neighbor = self.get_neighbor(current_solution)
                neighbor_cost = self.evaluate_solution(neighbor)

                if neighbor_cost < float('inf'):
                    delta = neighbor_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta/temp):
                        current_solution = neighbor
                        current_cost = neighbor_cost

                        if current_cost < self.best_cost:
                            self.best_solution = current_solution
                            self.best_cost = current_cost
                            non_improving = 0
                            print("New solution found")
                        else:
                            non_improving += 1

            temp *= cooling_rate
            print(f"Temperature: {temp:.2f}, Current best: {self.best_cost:.2f}")

        print(f"Simulated annealing complete. Best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost