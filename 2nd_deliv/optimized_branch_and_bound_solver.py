from typing import Dict, List, Optional, Tuple
import heapq
from dataclasses import dataclass
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from simuiated_annealing_preprocessor import SimulatedAnnealingPreprocessor


@dataclass
class OptimizedNode:
    """Represents a node in the branch and bound tree with optimization features"""
    level: int
    fixed_vars: Dict
    parent: Optional['OptimizedNode']
    obj_value: float = float('inf')
    is_feasible: bool = False
    solution: Optional[Dict] = None

    def calculate_infeasibility(self):
        if not self.solution:
            self.infeasibility = float('inf')
            return

        total_infeasibility = 0
        for val in self.solution['y'].values():
            if abs(val - round(val)) > 1e-6:
                total_infeasibility += abs(val - round(val))

        for val in self.solution['x'].values():
            if abs(val - round(val)) > 1e-6:
                total_infeasibility += abs(val - round(val))

        self.infeasibility = total_infeasibility

    def calculate_integer_vars(self):
        if not self.solution:
            self.integer_vars = 0
            return

        count = 0
        for val in self.solution['y'].values():
            if abs(val - round(val)) <= 1e-6:
                count += 1

        for val in self.solution['x'].values():
            if abs(val - round(val)) <= 1e-6:
                count += 1

        self.integer_vars = count

    def get_priority(self, iteration):
        if not hasattr(self, 'infeasibility'):
            self.calculate_infeasibility()
        if not hasattr(self, 'integer_vars'):
            self.calculate_integer_vars()

        obj_weight = 1.0
        infeas_weight = 10.0
        level_weight = 0.1

        priority = (
                obj_weight * self.obj_value +
                infeas_weight * self.infeasibility -
                level_weight * self.integer_vars
        )

        return priority


class OptimizedBranchAndBoundSolver:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.nodes_explored = 0
        self.best_objective = float('inf')
        self.best_solution = None

        self.search_stats = {
            'total_nodes': 0,
            'pruned_nodes': 0,
            'integer_solutions': 0,
            'best_bound': float('inf'),
            'preprocessing_time': 0,
            'preprocessing_objective': float('inf')
        }

    def is_integer_feasible(self, solution: Dict) -> bool:
        """Check if all binary variables have integer values"""
        EPSILON = 1e-5
        return all(abs(val - round(val)) <= EPSILON for val in solution['y'].values()) and \
            all(abs(val - round(val)) <= EPSILON for val in solution['x'].values())

    def get_branching_variable(self, solution: Dict) -> Optional[Tuple[str, str]]:
        """Get next variable to branch on"""
        EPSILON = 1e-5
        candidates = []

        # Check y variables first (substation opening decisions)
        for j in [f'Sub{j + 1}' for j in range(self.optimizer.generator.num_substations)]:
            if abs(solution['y'][j] - round(solution['y'][j])) > EPSILON:
                candidates.append(('y', j))

        # Then check x variables (assignment decisions)
        if not candidates:
            for j in [f'Sub{j + 1}' for j in range(self.optimizer.generator.num_substations)]:
                for k in [f'City{k + 1}' for k in range(self.optimizer.generator.num_customers)]:
                    if abs(solution['x'][(j, k)] - round(solution['x'][(j, k)])) > EPSILON:
                        candidates.append(('x', (j, k)))

        if not candidates:
            return None

        return max(candidates,
                   key=lambda c: abs(solution[c[0]][c[1]] - 0.5) if c[0] == 'y'
                   else abs(solution[c[0]][c[1][0], c[1][1]] - 0.5))

    def solve_node(self, node: OptimizedNode) -> Tuple[bool, float, Dict]:
        """Solve LP relaxation for a node"""
        try:
            model = self.optimizer.create_model(relaxed=True, fixed_vars=node.fixed_vars)
            if model is None:
                return False, float('inf'), None

            solver = SolverFactory('gurobi')
            solver.options['FeasibilityTol'] = 1e-6
            solver.options['OptimalityTol'] = 1e-6
            solver.options['NumericFocus'] = 3

            results = solver.solve(model, tee=False)

            if (results.solver.status == SolverStatus.ok and
                    results.solver.termination_condition == TerminationCondition.optimal):
                solution = self.optimizer.extract_solution(model)
                obj_value = pyo.value(model.objective)
                return True, obj_value, solution
            else:
                return False, float('inf'), None

        except Exception as e:
            print(f"Error solving node: {str(e)}")
            return False, float('inf'), None

    def preprocess(self):
        """Run simulated annealing preprocessing"""
        import time
        start_time = time.time()

        # Initialize and run simulated annealing
        sa_preprocessor = SimulatedAnnealingPreprocessor(self.optimizer)
        initial_solution, initial_cost = sa_preprocessor.run()

        # Update statistics
        self.search_stats['preprocessing_time'] = time.time() - start_time
        self.search_stats['preprocessing_objective'] = initial_cost

        # If preprocessing found a better solution, update best solution
        if initial_cost < self.best_objective:
            self.best_objective = initial_cost
            self.best_solution = initial_solution
            print(f"Preprocessing found better solution: {initial_cost:.2f}")

        return initial_solution

    def update_search_stats(self, node: OptimizedNode, pruned: bool = False):
        """Update search statistics"""
        self.search_stats['total_nodes'] += 1
        if pruned:
            self.search_stats['pruned_nodes'] += 1
        if self.is_integer_feasible(node.solution):
            self.search_stats['integer_solutions'] += 1
        self.search_stats['best_bound'] = min(self.search_stats['best_bound'], node.obj_value)

    def solve(self) -> Dict:
        """Main branch and bound solving method with preprocessing"""
        print("\nStarting Optimized Branch and Bound Solver with Preprocessing")
        print("=====================================================")

        # Reset statistics
        self.nodes_explored = 0
        self.best_objective = float('inf')
        self.best_solution = None

        # Run preprocessing
        print("\nRunning preprocessing phase...")
        initial_solution = self.preprocess()

        # Create and solve root node
        print("\nSolving root node...")
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

        # Initialize priority queue with root node
        priority_queue = [(root.get_priority(0), id(root), root)]
        heapq.heapify(priority_queue)

        print(f"Root node solved - Starting branch and bound search")
        print(f"Initial objective: {obj_value:.2f}")
        print(f"Best known solution from preprocessing: {self.best_objective:.2f}")

        while priority_queue:
            _, _, current = heapq.heappop(priority_queue)
            self.nodes_explored += 1

            if self.nodes_explored % 10 == 0:
                print(f"Nodes explored: {self.nodes_explored}, Best objective: {self.best_objective:.2f}")

            # Prune by bound
            if current.obj_value >= self.best_objective:
                self.update_search_stats(current, pruned=True)
                continue

            # Check if solution is integer feasible
            if self.is_integer_feasible(current.solution):
                if current.obj_value < self.best_objective:
                    self.best_objective = current.obj_value
                    self.best_solution = current.solution
                    print(f"New best solution found: {self.best_objective:.2f}")
                self.update_search_stats(current)
                continue

            # Get branching variable
            branch_var = self.get_branching_variable(current.solution)
            if not branch_var:
                continue

            # Create child nodes
            for value in [0, 1]:
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

        print(f"\nBranch and Bound search completed:")
        print(f"Preprocessing time: {self.search_stats['preprocessing_time']:.2f} seconds")
        print(f"Preprocessing objective: {self.search_stats['preprocessing_objective']:.2f}")
        print(f"Total nodes explored: {self.nodes_explored}")
        print(f"Pruned nodes: {self.search_stats['pruned_nodes']}")
        print(f"Integer solutions found: {self.search_stats['integer_solutions']}")
        print(f"Final best objective: {self.best_objective:.2f}")

        if self.best_solution:
            return {
                'objective_value': self.best_objective,
                'opened_substations': [j for j, val in self.best_solution['y'].items()
                                       if abs(val - 1) < 1e-6],
                'assignments': {k: next(j for j, k2 in self.best_solution['x'].keys()
                                        if k2 == k and abs(self.best_solution['x'][(j, k)] - 1) < 1e-6)
                                for k in [f'City{k + 1}' for k in range(self.optimizer.generator.num_customers)]},
                'flows': self.best_solution.get('flows', {
                    'facility_to_substation': {},
                    'substation_to_customer': {}
                }),
                'branch_and_bound_stats': {
                    'nodes_explored': self.nodes_explored,
                    'search_statistics': self.search_stats
                }
            }
        return None