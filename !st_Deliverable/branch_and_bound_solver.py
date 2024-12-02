# branch_and_bound.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

@dataclass
class Node:
    """Represents a node in the branch and bound tree"""
    level: int
    fixed_vars: Dict[Tuple[str, str], int]  # Dictionary of fixed variables (y_j, x_jk) and their values
    parent: Optional['Node']
    obj_value: float = float('inf')
    is_feasible: bool = False
    solution: Optional[Dict] = None

class BranchAndBoundSolver:
    def __init__(self, optimizer):
        """Initialize with reference to thermal network optimizer"""
        self.optimizer = optimizer
        self.nodes_explored = 0
        self.best_objective = float('inf')
        self.best_solution = None
        
    def is_integer_feasible(self, solution: Dict) -> bool:
        """Check if all binary variables have integer values"""
        EPSILON = 1e-6
        return all(abs(val - round(val)) <= EPSILON for val in solution['y'].values()) and \
               all(abs(val - round(val)) <= EPSILON for val in solution['x'].values())
               
    def get_branching_variable(self, solution: Dict) -> Optional[Tuple[str, str]]:
        """Get next variable to branch on (lexicographically)"""
        EPSILON = 1e-6
        
        # First check y variables
        for j in [f'Sub{j+1}' for j in range(self.optimizer.generator.num_substations)]:
            if abs(solution['y'][j] - round(solution['y'][j])) > EPSILON:
                return ('y', j)
        
        # Then check x variables
        for j in [f'Sub{j+1}' for j in range(self.optimizer.generator.num_substations)]:
            for k in [f'City{k+1}' for k in range(self.optimizer.generator.num_customers)]:
                if abs(solution['x'][(j,k)] - round(solution['x'][(j,k)])) > EPSILON:
                    return ('x', (j,k))
        
        return None
        
    def solve_node(self, node: Node) -> Tuple[bool, float, Dict]:
        """Solve LP relaxation for a node"""
        # Create and solve relaxed model with fixed variables
        model = self.optimizer.create_model(relaxed=True, fixed_vars=node.fixed_vars)
        solver = SolverFactory('gurobi')
        results = solver.solve(model)
        
        if (results.solver.status == SolverStatus.ok and 
            results.solver.termination_condition == TerminationCondition.optimal):
            solution = self.optimizer.extract_solution(model)
            return True, pyo.value(model.objective), solution
        else:
            return False, float('inf'), None
            
    def solve(self) -> Dict:
        """Main branch and bound solving method"""
        # Reset statistics
        self.nodes_explored = 0
        self.best_objective = float('inf')
        self.best_solution = None
        
        # Create and solve root node
        root = Node(level=0, fixed_vars={}, parent=None)
        is_feasible, obj_value, solution = self.solve_node(root)
        
        if not is_feasible:
            print("Root node infeasible - problem has no solution")
            return None
            
        root.obj_value = obj_value
        root.is_feasible = True
        root.solution = solution
        
        # Initialize stack with root node
        stack = [root]
        
        while stack:
            current = stack.pop()
            self.nodes_explored += 1
            
            print(f"Exploring node {self.nodes_explored} at level {current.level}, bound: {current.obj_value:.2f}")
            
            # Prune by bound
            if current.obj_value >= self.best_objective:
                continue
            
            # Check if solution is integer feasible
            if self.is_integer_feasible(current.solution):
                if current.obj_value < self.best_objective:
                    self.best_objective = current.obj_value
                    self.best_solution = current.solution
                continue
            
            # Get branching variable
            branch_var = self.get_branching_variable(current.solution)
            if not branch_var:
                continue
            
            # Create child nodes
            for value in [0, 1]:
                child_fixed_vars = current.fixed_vars.copy()
                child_fixed_vars[branch_var] = value
                
                child = Node(
                    level=current.level + 1,
                    fixed_vars=child_fixed_vars,
                    parent=current
                )
                
                is_feasible, obj_value, child_solution = self.solve_node(child)
                if is_feasible and obj_value < self.best_objective:
                    child.obj_value = obj_value
                    child.is_feasible = True
                    child.solution = child_solution
                    stack.append(child)
        
        # Format solution to match original optimizer output
        if self.best_solution:
            return {
                'objective_value': self.best_objective,
                'opened_substations': [j for j, val in self.best_solution['y'].items() 
                                     if abs(val - 1) < 1e-6],
                'assignments': {k: next(j for j, k2 in self.best_solution['x'].keys() 
                                      if k2 == k and abs(self.best_solution['x'][(j,k)] - 1) < 1e-6)
                              for k in [f'City{k+1}' for k in range(self.optimizer.generator.num_customers)]},
                'flows': self.best_solution['flows'],
                'branch_and_bound_stats': {
                    'nodes_explored': self.nodes_explored
                }
            }
        return None