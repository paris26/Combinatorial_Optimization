from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import heapq
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from CPLEXPreprocessor import GurobiPreprocessor
import time


@dataclass
class BestBoundNode:
    """Node for best bound branch and bound"""
    fixed_vars: Dict  # Dictionary of fixed variables
    obj_value: float = float('inf')
    solution: Optional[Dict] = None

    def __lt__(self, other):
        # For priority queue ordering - lower objective value has higher priority
        return self.obj_value < other.obj_value


class BestBoundBranchAndBoundSolver:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.nodes_explored = 0
        self.best_objective = float('inf')
        self.best_solution = None

        # Statistics tracking
        self.stats = {
            'preprocessing_time': 0,
            'preprocessing_objective': float('inf'),
            'total_nodes': 0,
            'pruned_nodes': 0,
            'best_bound_progress': []  # Track improvement in lower bound
        }

    def is_integer_feasible(self, solution: Dict) -> bool:
        """Check if all binary variables have integer values"""
        EPSILON = 1e-5
        return all(abs(val - round(val)) <= EPSILON for val in solution['y'].values()) and \
            all(abs(val - round(val)) <= EPSILON for val in solution['x'].values())

    def get_branching_variable(self, solution: Dict) -> Optional[Tuple[str, str]]:
        """Get variable furthest from integrality to branch on"""
        EPSILON = 1e-5
        max_fractional = 0
        selected_var = None

        # Check y variables (substation opening decisions)
        for j in [f'Sub{j + 1}' for j in range(self.optimizer.generator.num_substations)]:
            fractional = abs(solution['y'][j] - round(solution['y'][j]))
            if fractional > EPSILON and fractional > max_fractional:
                max_fractional = fractional
                selected_var = ('y', j)

        # Check x variables (assignment decisions)
        for j in [f'Sub{j + 1}' for j in range(self.optimizer.generator.num_substations)]:
            for k in [f'City{k + 1}' for k in range(self.optimizer.generator.num_customers)]:
                fractional = abs(solution['x'][(j, k)] - round(solution['x'][(j, k)]))
                if fractional > EPSILON and fractional > max_fractional:
                    max_fractional = fractional
                    selected_var = ('x', (j, k))

        return selected_var

    def solve_node(self, node: BestBoundNode) -> Tuple[bool, float, Dict]:
        """Solve LP relaxation for a node"""
        try:
            model = self.optimizer.create_model(relaxed=True, fixed_vars=node.fixed_vars)
            if model is None:
                return False, float('inf'), None

            solver = SolverFactory('gurobi')
            solver.options['FeasibilityTol'] = 1e-6
            solver.options['OptimalityTol'] = 1e-6
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
        """Run preprocessing phase"""
        start_time = time.time()
        try:
            preprocessor = GurobiPreprocessor(optimizer=self.optimizer)
            initial_solution, initial_cost = preprocessor.preprocess()

            self.stats['preprocessing_time'] = time.time() - start_time
            self.stats['preprocessing_objective'] = initial_cost

            if initial_solution and initial_cost < float('inf'):
                print(f"Preprocessing found solution with cost: {initial_cost:.2f}")
                return initial_solution
            else:
                print("Preprocessing did not find a valid solution")
                return None

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            self.stats['preprocessing_time'] = time.time() - start_time
            return None

    def solve(self) -> Dict:
        """Main branch and bound solving method with preprocessing"""
        print("\nStarting Best Bound Branch and Bound with Preprocessing")
        print("=================================================")

        # Reset statistics
        self.nodes_explored = 0
        self.best_objective = float('inf')
        self.best_solution = None

        # Run preprocessing
        print("\nRunning preprocessing phase...")
        initial_solution = self.preprocess()

        if initial_solution:
            self.best_solution = initial_solution
            self.best_objective = self.stats['preprocessing_objective']
            print(f"Initial solution from preprocessing: {self.best_objective:.2f}")

        # Create and solve root node
        print("\nSolving root node...")
        root = BestBoundNode(fixed_vars={})
        is_feasible, obj_value, solution = self.solve_node(root)

        if not is_feasible:
            print("Root node infeasible - problem has no solution")
            return None

        root.obj_value = obj_value
        root.solution = solution

        # Initialize priority queue with root node
        # Queue elements are tuples of (objective_value, node_count, node)
        # node_count ensures FIFO behavior for nodes with equal objective values
        priority_queue = [(root.obj_value, 0, root)]
        heapq.heapify(priority_queue)
        self.stats['total_nodes'] = 1
        node_count = 1

        # Track best lower bound
        best_lower_bound = obj_value
        self.stats['best_bound_progress'].append((0, best_lower_bound))

        while priority_queue:
            current_bound, _, current = heapq.heappop(priority_queue)
            self.nodes_explored += 1

            # Update best lower bound
            if current_bound > best_lower_bound:
                best_lower_bound = current_bound
                self.stats['best_bound_progress'].append((self.nodes_explored, best_lower_bound))

            if self.nodes_explored % 10 == 0:
                print(f"Nodes explored: {self.nodes_explored}, "
                      f"Best objective: {self.best_objective:.2f}, "
                      f"Best bound: {best_lower_bound:.2f}")

            # Prune by bound
            if current.obj_value >= self.best_objective:
                self.stats['pruned_nodes'] += 1
                continue

            # Check if solution is integer feasible
            if self.is_integer_feasible(current.solution):
                if current.obj_value < self.best_objective:
                    self.best_objective = current.obj_value
                    self.best_solution = current.solution
                    print(f"New best solution found: {self.best_objective:.2f}")
                continue

            # Get branching variable
            branch_var = self.get_branching_variable(current.solution)
            if not branch_var:
                continue

            # Create child nodes (branch on 0 and 1)
            for value in [0, 1]:
                child_fixed_vars = current.fixed_vars.copy()
                child_fixed_vars[branch_var] = value

                child = BestBoundNode(fixed_vars=child_fixed_vars)
                is_feasible, obj_value, child_solution = self.solve_node(child)

                if is_feasible and obj_value < self.best_objective:
                    child.obj_value = obj_value
                    child.solution = child_solution
                    heapq.heappush(priority_queue, (obj_value, node_count, child))
                    self.stats['total_nodes'] += 1
                    node_count += 1
                    print(f"Added node {node_count} with objective: {obj_value:.2f}")
                else:
                    print(f"Node pruned: feasible={is_feasible}, obj={obj_value:.2f}")
                    self.stats['pruned_nodes'] += 1

        # Print final statistics
        print(f"\nBranch and Bound search completed:")
        print(f"Preprocessing time: {self.stats['preprocessing_time']:.2f} seconds")
        print(f"Total nodes explored: {self.nodes_explored}")
        print(f"Pruned nodes: {self.stats['pruned_nodes']}")
        print(f"Final best objective: {self.best_objective:.2f}")
        print(f"Final best bound: {best_lower_bound:.2f}")

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
                    'search_statistics': self.stats
                }
            }
        return None