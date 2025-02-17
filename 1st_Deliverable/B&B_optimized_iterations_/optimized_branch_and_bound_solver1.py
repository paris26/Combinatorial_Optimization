    from dataclasses import dataclass
    from typing import Dict, List, Optional, Tuple, Any
    import heapq
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


    @dataclass
    class OptimizedNode:
        """Represents a node in the optimized branch and bound tree"""
        level: int
        fixed_vars: Dict[Tuple[str, str], int]  # Dictionary of fixed variables (y_j, x_jk) and their values
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
            # Check y variables (substation opening decisions)
            for val in self.solution['y'].values():
                total_diff += abs(val - round(val))

            # Check x variables (assignment decisions)
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

            # Count integer y variables
            for val in self.solution['y'].values():
                if abs(val - round(val)) <= EPSILON:
                    count += 1

            # Count integer x variables
            for val in self.solution['x'].values():
                if abs(val - round(val)) <= EPSILON:
                    count += 1

            self.num_integer_vars = count
            return count

        def get_priority(self, nodes_explored: int):
            """Calculate node priority using multiple factors"""
            # Base priority from bound
            priority = self.obj_value

            if nodes_explored < 1000:  # Early in search
                # Focus on bound quality and integer feasibility
                priority += self.infeasibility_measure * 0.2
                # Small bonus for having more fixed variables
                fixed_vars_bonus = -len(self.fixed_vars) * 0.05
                priority += fixed_vars_bonus
            else:  # Later in search
                # Focus more on finding integer solutions
                priority += self.infeasibility_measure * 0.5
                # Stronger bonus for fixed variables
                fixed_vars_bonus = -len(self.fixed_vars) * 0.1
                priority += fixed_vars_bonus
                # Add level penalty to prevent deep diving
                level_penalty = self.level * 0.1
                priority += level_penalty

            return priority

        def __lt__(self, other):
            """Comparison method for priority queue ordering"""
            # Compare based on objective value as a default
            # The solver will use get_priority() for more sophisticated comparison
            return self.obj_value < other.obj_value


    class OptimizedBranchAndBoundSolver:
        """
        An optimized implementation of Branch and Bound using best-first search strategy
        with sophisticated node prioritization.
        """

        def __init__(self, optimizer):
            """Initialize with reference to thermal network optimizer"""
            self.optimizer = optimizer
            self.nodes_explored = 0
            self.best_objective = float('inf')
            self.best_solution = None
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
            """
            Get next variable to branch on using most fractional variable rule
            """
            EPSILON = 1e-6
            max_frac_diff = 0
            selected_var = None

            # Check y variables first (they have higher impact)
            for j in [f'Sub{j + 1}' for j in range(self.optimizer.generator.num_substations)]:
                val = solution['y'][j]
                if abs(val - round(val)) > EPSILON:
                    frac_diff = abs(val - 0.5)  # Distance from 0.5
                    if frac_diff > max_frac_diff:
                        max_frac_diff = frac_diff
                        selected_var = ('y', j)

            # Then check x variables
            for j in [f'Sub{j + 1}' for j in range(self.optimizer.generator.num_substations)]:
                for k in [f'City{k + 1}' for k in range(self.optimizer.generator.num_customers)]:
                    val = solution['x'][(j, k)]
                    if abs(val - round(val)) > EPSILON:
                        frac_diff = abs(val - 0.5)
                        if frac_diff > max_frac_diff:
                            max_frac_diff = frac_diff
                            selected_var = ('x', (j, k))

            return selected_var

        def solve_node(self, node: OptimizedNode) -> Tuple[bool, float, Dict]:
            """Solve LP relaxation for a node"""
            # Create and solve relaxed model with fixed variables
            model = self.optimizer.create_model(relaxed=True, fixed_vars=node.fixed_vars)
            solver = SolverFactory('gurobi')

            # Add solver parameters for better numerical stability
            solver_options = {
                'FeasibilityTol': 1e-6,
                'OptimalityTol': 1e-6,
                'NumericFocus': 3,
                'Method': 2,  # Barrier method
            }

            results = solver.solve(model, options=solver_options)

            if (results.solver.status == SolverStatus.ok and
                    results.solver.termination_condition == TerminationCondition.optimal):
                solution = self.optimizer.extract_solution(model)
                return True, pyo.value(model.objective), solution
            else:
                return False, float('inf'), None

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
            """Main branch and bound solving method using best-first search"""
            # Reset statistics
            self.nodes_explored = 0
            self.best_objective = float('inf')
            self.best_solution = None
            self.search_stats = {
                'total_nodes': 0,
                'pruned_nodes': 0,
                'integer_solutions': 0,
                'best_bound': float('inf')
            }

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

            # Initialize priority queue with root node
            priority_queue = [(root.get_priority(0), id(root), root)]
            heapq.heapify(priority_queue)

            while priority_queue:
                # Get the most promising node
                _, _, current = heapq.heappop(priority_queue)
                self.nodes_explored += 1

                print(f"Exploring node {self.nodes_explored} at level {current.level}:")
                print(f"  Bound: {current.obj_value:.2f}")
                print(f"  Infeasibility: {current.infeasibility_measure:.3f}")
                print(f"  Integer vars: {current.num_integer_vars}")
                print(f"  Fixed vars: {len(current.fixed_vars)}")

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

                # Get branching variable using most fractional rule
                branch_var = self.get_branching_variable(current.solution)
                if not branch_var:
                    continue

                # Create child nodes with different branching strategies for y and x variables
                if branch_var[0] == 'y':
                    # For y variables, try 1 first (opening a substation)
                    branch_values = [1, 0]
                else:
                    # For x variables, let value in solution guide the order
                    current_val = current.solution['x'][branch_var[1]]
                    branch_values = [1, 0] if current_val > 0.5 else [0, 1]

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

                        # Add to priority queue with complex priority
                        priority = child.get_priority(self.nodes_explored)
                        heapq.heappush(priority_queue, (priority, id(child), child))
                        self.update_search_stats(child)

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