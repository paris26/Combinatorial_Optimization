import random
import math
import copy
from typing import Dict, List, Tuple, Optional
import time
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


class GreedyHeuristicPreprocessor:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.generator = optimizer.generator
        self.data = optimizer.data
        self.max_attempts = 5  # Maximum number of attempts to find feasible solution

    def find_initial_solution(self):
        """Find an initial solution using greedy heuristic with multiple attempts"""
        best_solution = None
        best_cost = float('inf')

        for attempt in range(self.max_attempts):
            print(f"\nAttempt {attempt + 1} to find feasible solution...")
            solution = self._create_solution(randomization_factor=attempt * 0.2)

            # Add flow decisions
            solution = self._add_flow_decisions(solution)

            # Evaluate solution
            cost = self.evaluate_solution(solution, debug=True)

            if cost < best_cost:
                best_solution = solution
                best_cost = cost

            if cost < float('inf'):
                print(f"Found feasible solution on attempt {attempt + 1}")
                break

        return best_solution

    def _create_solution(self, randomization_factor=0.0):
        """Create initial solution with optional randomization"""
        solution = {
            'z': {},  # Facility decisions
            'y': {},  # Substation decisions
            'x': {},  # Assignment decisions
        }

        # Calculate total winter demand
        total_demand = {}
        for (city, season), demand in self.data['demand'].items():
            if season == 'Winter':
                total_demand[city] = demand

        # Score and open facilities with randomization
        facility_scores = self._score_facilities(total_demand)
        for facility, score in facility_scores.items():
            if randomization_factor > 0:
                facility_scores[facility] = score * random.uniform(1 - randomization_factor, 1 + randomization_factor)

        opened_facilities = self._select_facilities(facility_scores)
        for facility in self.generator.locations:
            if 'Plant' in facility:
                solution['z'][facility] = 1.0 if facility in opened_facilities else 0.0

        # Score and open substations with randomization
        substation_scores = self._score_substations(total_demand, opened_facilities)
        for substation, score in substation_scores.items():
            if randomization_factor > 0:
                substation_scores[substation] = score * random.uniform(1 - randomization_factor,
                                                                       1 + randomization_factor)

        opened_substations = self._select_substations(substation_scores)
        for substation in self.generator.locations:
            if 'Sub' in substation:
                solution['y'][substation] = 1.0 if substation in opened_substations else 0.0

        # Assign customers
        self._assign_customers(solution, total_demand, randomization_factor)

        return solution

    def _add_flow_decisions(self, solution):
        """Add flow decisions to make solution complete"""
        try:
            flows = {
                'facility_to_substation': {},
                'substation_to_customer': {}
            }

            # Get opened facilities and substations
            opened_facilities = [f for f, v in solution['z'].items() if v > 0.5]
            opened_substations = [s for s, v in solution['y'].items() if v > 0.5]

            # Initialize a simple flow distribution
            for season in self.generator.seasons:
                # Calculate demand per substation in this season
                substation_demand = {s: 0.0 for s in opened_substations}

                for city in [f'City{k + 1}' for k in range(self.generator.num_customers)]:
                    demand = self.data['demand'][(city, season)]
                    # Find assigned substation for this city
                    assigned_sub = None
                    for sub in opened_substations:
                        if solution['x'].get((sub, city), 0.0) > 0.5:
                            assigned_sub = sub
                            break

                    if assigned_sub:
                        # Account for heat loss in demand
                        substation_demand[assigned_sub] += demand / self.data['beta'][(assigned_sub, city)]

                # Distribute facility production to substations
                remaining_demand = substation_demand.copy()
                for facility in opened_facilities:
                    # Sort substations by distance to this facility
                    sorted_subs = sorted(opened_substations,
                                         key=lambda s: self.generator.distances[facility][s])

                    capacity = self.data['facility_capacity'][facility]
                    remaining_capacity = capacity

                    for sub in sorted_subs:
                        if remaining_capacity <= 0 or remaining_demand[sub] <= 0:
                            continue

                        # Account for heat loss in flow
                        flow = min(remaining_capacity, remaining_demand[sub])
                        actual_flow = flow / self.data['alpha'][(facility, sub)]
                        flows['facility_to_substation'][(facility, sub, season)] = actual_flow
                        remaining_capacity -= actual_flow
                        remaining_demand[sub] -= flow

                # Set customer flows
                for city in [f'City{k + 1}' for k in range(self.generator.num_customers)]:
                    demand = self.data['demand'][(city, season)]
                    for sub in opened_substations:
                        if solution['x'].get((sub, city), 0.0) > 0.5:
                            flows['substation_to_customer'][(sub, city, season)] = demand

            solution['flows'] = flows
            return solution

        except Exception as e:
            print(f"Error in _add_flow_decisions: {str(e)}")
            return solution  # Return original solution without flows on error

    def _score_facilities(self, total_demand):
        """Score facilities based on their position and capacity"""
        scores = {}
        for facility in [f for f in self.generator.locations if 'Plant' in f]:
            capacity = self.data['facility_capacity'][facility]
            weighted_distance = 0
            for city, demand in total_demand.items():
                distance = self.generator.distances[facility][city]
                weighted_distance += demand / (distance + 1)
            scores[facility] = capacity * weighted_distance
        return scores

    def _select_facilities(self, facility_scores):
        """Select facilities based on scores"""
        sorted_facilities = sorted(facility_scores.items(), key=lambda x: x[1], reverse=True)
        total_capacity_needed = sum(self.data['demand'].values()) * 1.5  # Increased margin

        opened_facilities = []
        current_capacity = 0

        for facility, _ in sorted_facilities:
            if current_capacity < total_capacity_needed:
                opened_facilities.append(facility)
                current_capacity += self.data['facility_capacity'][facility]
            else:
                break

        # Always keep at least half of facilities open
        min_facilities = max(1, len(facility_scores) // 2)
        while len(opened_facilities) < min_facilities:
            for facility, _ in sorted_facilities:
                if facility not in opened_facilities:
                    opened_facilities.append(facility)
                    break

        return opened_facilities

    def _score_substations(self, total_demand, opened_facilities):
        """Score substations based on position between facilities and customers"""
        scores = {}
        for substation in [s for s in self.generator.locations if 'Sub' in s]:
            facility_proximity = sum(1 / (self.generator.distances[substation][f] + 1)
                                     for f in opened_facilities)
            customer_proximity = sum(demand / (self.generator.distances[substation][city] + 1)
                                     for city, demand in total_demand.items())
            scores[substation] = facility_proximity * customer_proximity * self.data['substation_capacity'][substation]
        return scores

    def _select_substations(self, substation_scores):
        """Select substations based on scores"""
        sorted_substations = sorted(substation_scores.items(), key=lambda x: x[1], reverse=True)
        total_capacity_needed = sum(self.data['demand'].values()) * 1.5  # Increased margin

        opened_substations = []
        current_capacity = 0

        for substation, _ in sorted_substations:
            if current_capacity < total_capacity_needed:
                opened_substations.append(substation)
                current_capacity += self.data['substation_capacity'][substation]
            else:
                break

        # Always keep at least half of substations open
        min_substations = max(1, len(substation_scores) // 2)
        while len(opened_substations) < min_substations:
            for substation, _ in sorted_substations:
                if substation not in opened_substations:
                    opened_substations.append(substation)
                    break

        return opened_substations

    def _assign_customers(self, solution, total_demand, randomization_factor=0.0):
        """Assign customers to nearest feasible substations"""
        opened_substations = [s for s, v in solution['y'].items() if v > 0.5]
        remaining_capacity = {s: self.data['substation_capacity'][s] for s in opened_substations}

        # Sort customers by demand with optional randomization
        customers = list(total_demand.items())
        if randomization_factor > 0:
            random.shuffle(customers)
        else:
            customers.sort(key=lambda x: x[1], reverse=True)

        for city, demand in customers:
            feasible_subs = [s for s in opened_substations if remaining_capacity[s] >= demand]

            if not feasible_subs:
                # If no feasible substation, try to open another one
                closed_subs = [s for s, v in solution['y'].items() if v < 0.5]
                if closed_subs:
                    new_sub = random.choice(closed_subs)
                    solution['y'][new_sub] = 1.0
                    remaining_capacity[new_sub] = self.data['substation_capacity'][new_sub]
                    feasible_subs = [new_sub]
                else:
                    # If can't open new substation, assign to one with most capacity
                    assigned_sub = max(opened_substations, key=lambda s: remaining_capacity[s])
                    feasible_subs = [assigned_sub]

            # Calculate scores for feasible substations
            sub_scores = {}
            for sub in feasible_subs:
                distance_score = 1 / (self.generator.distances[sub][city] + 1)
                capacity_score = remaining_capacity[sub] / self.data['substation_capacity'][sub]
                sub_scores[sub] = distance_score * capacity_score

                if randomization_factor > 0:
                    sub_scores[sub] *= random.uniform(1 - randomization_factor, 1 + randomization_factor)

            # Assign to best scoring substation
            assigned_sub = max(sub_scores.items(), key=lambda x: x[1])[0]

            # Update assignment and capacity
            for substation in [s for s in self.generator.locations if 'Sub' in s]:
                solution['x'][(substation, city)] = 1.0 if substation == assigned_sub else 0.0
            remaining_capacity[assigned_sub] -= demand

    def evaluate_solution(self, solution, debug=False):
        """Evaluate the cost of a given solution"""
        try:
            # Create fixed_vars dictionary by combining all decisions
            fixed_vars = {}
            # Add facility decisions
            for i, val in solution['z'].items():
                fixed_vars[('z', i)] = val
            # Add substation decisions
            for j, val in solution['y'].items():
                fixed_vars[('y', j)] = val
            # Add assignment decisions
            for (j, k), val in solution['x'].items():
                fixed_vars[('x', (j, k))] = val

            model = self.optimizer.create_model(relaxed=True, fixed_vars=fixed_vars)

            if model is None:
                if debug: print("Failed to create model")
                return float('inf')

            solver = SolverFactory('gurobi')
            solver.options['FeasibilityTol'] = 1e-6
            solver.options['OptimalityTol'] = 1e-6
            results = solver.solve(model, tee=debug)  # Enable solver output in debug mode

            if (results.solver.status == SolverStatus.ok and
                    results.solver.termination_condition == TerminationCondition.optimal):
                return pyo.value(model.objective)

            if debug:
                print(f"Solver Status: {results.solver.status}")
                print(f"Termination Condition: {results.solver.termination_condition}")
            return float('inf')
        except Exception as e:
            if debug: print(f"Error evaluating solution: {str(e)}")
            return float('inf')

    def run(self):
        """Run the heuristic and return solution"""
        initial_solution = self.find_initial_solution()

        if initial_solution is None:
            print("Failed to find feasible solution")
            return None, float('inf')

        # Evaluate solution cost using our own method
        cost = self.evaluate_solution(initial_solution)
        print(f"Greedy heuristic found solution with cost: {cost:.2f}")

        return initial_solution, cost