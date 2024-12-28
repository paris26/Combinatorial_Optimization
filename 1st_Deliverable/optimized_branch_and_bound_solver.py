from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import heapq
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition



class ThermalNetworkPreprocessor:
    def __init__(self, optimizer):
        """Initialize preprocessor with reference to thermal network optimizer"""
        self.optimizer = optimizer
        self.data = optimizer.data
        self.generator = optimizer.generator

    def calculate_substation_scores(self):
        """Calculate scores for each potential substation location"""
        scores = {}

        for j in range(self.generator.num_substations):
            substation = f'Sub{j + 1}'
            score = 0.0

            # 1. Location-based score
            customer_distances = [
                self.generator.distances[substation][f'City{k + 1}']
                for k in range(self.generator.num_customers)
            ]
            avg_distance = sum(customer_distances) / len(customer_distances)
            location_score = 1.0 / (1.0 + avg_distance)

            # 2. Capacity utilization score
            capacity = self.data['substation_capacity'][substation]
            max_winter_demand = sum(
                self.data['demand'][(f'City{k + 1}', 'Winter')]
                for k in range(self.generator.num_customers)
            )
            capacity_score = min(1.0, capacity / max_winter_demand)

            # 3. Cost efficiency score
            fixed_cost = self.data['fixed_cost'][substation]
            avg_fixed_cost = sum(self.data['fixed_cost'].values()) / len(self.data['fixed_cost'])
            cost_score = avg_fixed_cost / fixed_cost

            # 4. Heat loss efficiency score
            avg_heat_loss = sum(
                self.data['beta'][(substation, f'City{k + 1}')]
                for k in range(self.generator.num_customers)
            ) / self.generator.num_customers
            heat_loss_score = avg_heat_loss

            score = (
                    0.3 * location_score +  # Distance importance
                    0.3 * capacity_score +  # Capacity importance
                    0.2 * cost_score +      # Cost importance
                    0.2 * heat_loss_score   # Heat loss importance
            )

            scores[substation] = score

        return scores

    def calculate_customer_preferences(self):
        """Calculate preferred substations for each customer"""
        preferences = {}

        for k in range(self.generator.num_customers):
            customer = f'City{k + 1}'
            customer_scores = {}

            for j in range(self.generator.num_substations):
                substation = f'Sub{j + 1}'

                distance = self.generator.distances[substation][customer]
                distance_score = 1.0 / (1.0 + distance)

                heat_loss = self.data['beta'][(substation, customer)]

                winter_demand = self.data['demand'][(customer, 'Winter')]
                substation_capacity = self.data['substation_capacity'][substation]
                capacity_score = min(1.0, substation_capacity / winter_demand)

                score = (
                        0.4 * distance_score +  # Distance is most important
                        0.4 * heat_loss +       # Heat loss equally important
                        0.2 * capacity_score    # Capacity as secondary consideration
                )

                customer_scores[substation] = score

            preferences[customer] = sorted(
                customer_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

        return preferences

    def generate_initial_solution(self):
        """Generate initial solution based on heuristic scores"""
        substation_scores = self.calculate_substation_scores()
        customer_preferences = self.calculate_customer_preferences()

        num_min_substations = max(
            2,  # At least 2 substations
            self.generator.num_customers // 5  # Or 1 substation per 5 customers
        )

        open_substations = sorted(
            substation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_min_substations]

        initial_solution = {
            'opened_substations': [sub for sub, _ in open_substations],
            'assignments': {},
            'substation_loads': {sub: 0 for sub, _ in open_substations}
        }

        for k in range(self.generator.num_customers):
            customer = f'City{k + 1}'
            assigned = False

            for substation, _ in customer_preferences[customer]:
                if substation in initial_solution['opened_substations']:
                    current_load = initial_solution['substation_loads'][substation]
                    winter_demand = self.data['demand'][(customer, 'Winter')]
                    capacity = self.data['substation_capacity'][substation]

                    if current_load + winter_demand <= capacity:
                        initial_solution['assignments'][customer] = substation
                        initial_solution['substation_loads'][substation] += winter_demand
                        assigned = True
                        break

            if not assigned:
                for substation, score in sorted(
                        substation_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                ):
                    if substation not in initial_solution['opened_substations']:
                        initial_solution['opened_substations'].append(substation)
                        initial_solution['substation_loads'][substation] = winter_demand
                        initial_solution['assignments'][customer] = substation
                        break

        return initial_solution

    def get_heuristic_values(self):
        """Get heuristic values for branch and bound solver"""
        substation_scores = self.calculate_substation_scores()
        customer_preferences = self.calculate_customer_preferences()
        initial_solution = self.generate_initial_solution()

        heuristic_values = {
            'substation_scores': substation_scores,
            'customer_preferences': customer_preferences,
            'initial_solution': initial_solution,
            'pseudo_costs': {
                'opening_cost': {},
                'assignment_cost': {}
            }
        }

        for j in range(self.generator.num_substations):
            substation = f'Sub{j + 1}'
            heuristic_values['pseudo_costs']['opening_cost'][substation] = (
                    self.data['fixed_cost'][substation] / substation_scores[substation]
            )

        for k in range(self.generator.num_customers):
            customer = f'City{k + 1}'
            for j in range(self.generator.num_substations):
                substation = f'Sub{j + 1}'
                preference_score = next(
                    score for sub, score in customer_preferences[customer]
                    if sub == substation
                )
                heuristic_values['pseudo_costs']['assignment_cost'][(customer, substation)] = (
                        1.0 / preference_score
                )

        return heuristic_values


@dataclass
class OptimizedNode:
    """Represents a node in the optimized branch and bound tree"""
    level: int
    fixed_vars: Dict[Tuple[str, str], int]
    parent: Optional['OptimizedNode']
    obj_value: float = float('inf')
    is_feasible: bool = False
    solution: Optional[Dict] = None
    num_integer_vars: int = 0
    infeasibility_measure: float = float('inf')

    def calculate_infeasibility(self):
        """Calculate how far variables are from integer values"""
        if not self.solution:
            return float('inf')

        total_diff = 0.0
        for val in self.solution['y'].values():
            total_diff += abs(val - round(val))
        for val in self.solution['x'].values():
            total_diff += abs(val - round(val))

        self.infeasibility_measure = total_diff
        return total_diff

    def calculate_integer_vars(self):
        """Calculate number of variables that are effectively integer"""
        if not self.solution:
            return 0

        EPSILON = 1e-6
        count = 0

        for val in self.solution['y'].values():
            if abs(val - round(val)) <= EPSILON:
                count += 1
        for val in self.solution['x'].values():
            if abs(val - round(val)) <= EPSILON:
                count += 1

        self.num_integer_vars = count
        return count

    def get_priority(self, nodes_explored: int):
        """Calculate node priority using multiple factors"""
        priority = self.obj_value

        if nodes_explored < 1000:
            priority += self.infeasibility_measure * 0.2
            fixed_vars_bonus = -len(self.fixed_vars) * 0.05
            priority += fixed_vars_bonus
        else:
            priority += self.infeasibility_measure * 0.5
            fixed_vars_bonus = -len(self.fixed_vars) * 0.1
            priority += fixed_vars_bonus
            level_penalty = self.level * 0.1
            priority += level_penalty

        return priority

    def __lt__(self, other):
        return self.obj_value < other.obj_value


class OptimizedBranchAndBoundSolver:
    def __init__(self, optimizer):
        """Initialize with reference to thermal network optimizer"""
        self.optimizer = optimizer
        self.nodes_explored = 0
        self.best_objective = float('inf')
        self.best_solution = None

        # Initialize preprocessor and get heuristic values
        self.preprocessor = ThermalNetworkPreprocessor(optimizer)
        self.heuristic_values = self.preprocessor.get_heuristic_values()

        # Initialize with potentially better solution from heuristic
        initial_solution = self.heuristic_values['initial_solution']
        if initial_solution:
            self.best_solution = initial_solution
            self.best_objective = self.calculate_solution_objective(initial_solution)
            print(f"Initial solution found with objective: {self.best_objective:.2f}")

        self.search_stats = {
            'total_nodes': 0,
            'pruned_nodes': 0,
            'integer_solutions': 0,
            'best_bound': float('inf')
        }

    def is_integer_feasible(self, solution: Dict) -> bool:
        """Check if all binary variables have integer values"""
        EPSILON = 1e-6
        return all(abs(val - round(val)) <= EPSILON for val in solution['y'].values()) and \
            all(abs(val - round(val)) <= EPSILON for val in solution['x'].values())

    def get_branching_variable(self, solution: Dict) -> Optional[Tuple[str, str]]:
        """Get next variable to branch on using heuristic guidance"""
        EPSILON = 1e-6
        candidates = []

        # Check y variables (substation opening decisions)
        for j in range(self.optimizer.generator.num_substations):
            substation = f'Sub{j + 1}'
            val = solution['y'][substation]
            if abs(val - round(val)) > EPSILON:
                score = self.heuristic_values['substation_scores'][substation]
                pseudo_cost = self.heuristic_values['pseudo_costs']['opening_cost'][substation]
                candidates.append(('y', substation, score * pseudo_cost))

        # Check x variables (assignments)
        for j in range(self.optimizer.generator.num_substations):
            for k in range(self.optimizer.generator.num_customers):
                substation = f'Sub{j + 1}'
                customer = f'City{k + 1}'
                val = solution['x'][(substation, customer)]
                if abs(val - round(val)) > EPSILON:
                    preferences = self.heuristic_values['customer_preferences'][customer]
                    preference_score = next(
                        score for sub, score in preferences
                        if sub == substation
                    )
                    pseudo_cost = self.heuristic_values['pseudo_costs']['assignment_cost'][(customer, substation)]
                    candidates.append(('x', (substation, customer), preference_score * pseudo_cost))

        if not candidates:
            return None

        return max(candidates, key=lambda x: x[2])[:2]

    def solve_node(self, node: OptimizedNode) -> Tuple[bool, float, Dict]:
        """Solve LP relaxation for a node"""
        model = self.optimizer.create_model(relaxed=True, fixed_vars=node.fixed_vars)
        solver = SolverFactory('gurobi')

        solver_options = {
            'FeasibilityTol': 1e-6,
            'OptimalityTol': 1e-6,
            'NumericFocus': 3,
            'Method': 2
        }

        results = solver.solve(model, options=solver_options)

        if (results.solver.status == SolverStatus.ok and
                results.solver.termination_condition == TerminationCondition.optimal):
            solution = self.optimizer.extract_solution(model)
            return True, pyo.value(model.objective), solution
        else:
            return False, float('inf'), None

    def calculate_solution_objective(self, solution) -> float:
        """Calculate objective value for a given solution"""
        total_cost = 0

        # Fixed costs for open substations
        for sub in solution['opened_substations']:
            total_cost += self.optimizer.data['fixed_cost'][sub]

        # Add production costs for flows
        if 'flows' in solution:
            # Production costs for facility-to-substation flows
            for (i, j, s), flow in solution['flows']['facility_to_substation'].items():
                if flow > 1e-6:  # Only count significant flows
                    total_cost += self.optimizer.data['production_cost'][(i, j)] * flow

        return total_cost

    def update_search_stats(self, node: OptimizedNode, pruned: bool = False):
        """Update search statistics"""
        self.search_stats['total_nodes'] += 1
        if pruned:
            self.search_stats['pruned_nodes'] += 1
        if node.is_feasible and self.is_integer_feasible(node.solution):
            self.search_stats['integer_solutions'] += 1
        self.search_stats['best_bound'] = min(
            self.search_stats['best_bound'],
            node.obj_value
        )

    def solve(self) -> Dict:
        """Main branch and bound solving method using heuristic guidance"""
        # Reset statistics
        self.nodes_explored = 0
        self.search_stats = {
            'total_nodes': 0,
            'pruned_nodes': 0,
            'integer_solutions': 0,
            'best_bound': float('inf')
        }

        # Initialize priority queue
        priority_queue = []

        # Create and solve root node
        root = OptimizedNode(level=0, fixed_vars={}, parent=None)
        is_feasible, obj_value, solution = self.solve_node(root)

        if not is_feasible:
            print("Root node infeasible - problem has no solution")
            return None

        root.obj_value = obj_value
        root.is_feasible = True
        root.solution = solution
        root.calculate_infeasibility()
        root.calculate_integer_vars()

        # Add root node to priority queue
        heapq.heappush(priority_queue, (root.get_priority(0), id(root), root))

        print(f"Starting branch and bound search...")

        while priority_queue:  # Main branch and bound loop
            _, _, current = heapq.heappop(priority_queue)
            self.nodes_explored += 1

            # Print progress every 100 nodes
            if self.nodes_explored % 100 == 0:
                print(f"Nodes explored: {self.nodes_explored}, Best objective: {self.best_objective:.2f}")
                print(f"Queue size: {len(priority_queue)}")

            if current.obj_value >= self.best_objective:
                self.update_search_stats(current, pruned=True)
                continue

            if self.is_integer_feasible(current.solution):
                if current.obj_value < self.best_objective:
                    self.best_objective = current.obj_value
                    self.best_solution = current.solution
                    print(f"New best solution found: {self.best_objective:.2f}")
                self.update_search_stats(current)
                continue

            branch_var = self.get_branching_variable(current.solution)
            if not branch_var:
                continue

            # Use heuristic to determine branching order
            if branch_var[0] == 'y':
                # For substation variables, check score
                substation = branch_var[1]
                score = self.heuristic_values['substation_scores'][substation]
                branch_values = [1, 0] if score > 0.5 else [0, 1]
            else:
                # For assignment variables, check customer preference
                j, k = branch_var[1]
                preferences = self.heuristic_values['customer_preferences'][k]
                preferred_sub = preferences[0][0]  # Most preferred substation
                branch_values = [1, 0] if j == preferred_sub else [0, 1]

            for value in branch_values:
                child_fixed_vars = current.fixed_vars.copy()
                child_fixed_vars[branch_var] = value

                child = OptimizedNode(
                    level=current.level + 1,
                    fixed_vars=child_fixed_vars,
                    parent=current
                )

                is_feasible, obj_value, child_solution = self.solve_node(child)
                if is_feasible and obj_value < self.best_objective:
                    child.obj_value = obj_value
                    child.is_feasible = True
                    child.solution = child_solution
                    child.calculate_infeasibility()
                    child.calculate_integer_vars()

                    priority = child.get_priority(self.nodes_explored)
                    heapq.heappush(priority_queue, (priority, id(child), child))
                    self.update_search_stats(child)

                    # Update best bound
                    self.search_stats['best_bound'] = min(
                        self.search_stats['best_bound'],
                        child.obj_value
                    )

        # Print final statistics
        print(f"\nBranch and Bound search completed:")
        print(f"Total nodes explored: {self.nodes_explored}")
        print(f"Pruned nodes: {self.search_stats['pruned_nodes']}")
        print(f"Integer solutions found: {self.search_stats['integer_solutions']}")
        print(f"Final best objective: {self.best_objective:.2f}")

        # Format solution to match original optimizer output
        if self.best_solution:
            return {
                'objective_value': self.best_objective,
                'opened_substations': [j for j, val in self.best_solution['y'].items()
                                       if abs(val - 1) < 1e-6],
                'assignments': {k: next(j for j, k2 in self.best_solution['x'].keys()
                                        if k2 == k and abs(self.best_solution['x'][(j, k)] - 1) < 1e-6)
                                for k in [f'City{k + 1}' for k in range(self.optimizer.generator.num_customers)]},
                'flows': self.best_solution['flows'],
                'branch_and_bound_stats': {
                    'nodes_explored': self.nodes_explored,
                    'search_statistics': self.search_stats
                }
            }
        return None