import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

class ThermalNetworkOptimizer:
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.generate_all_data()

    def create_model(self, relaxed=False, fixed_vars=None):
        """Create the optimization model with option for relaxation"""
        try:
            model = pyo.ConcreteModel()
            
            # Sets
            model.I = pyo.Set(initialize=[f'Plant{i+1}' for i in range(self.generator.num_facilities)])
            model.J = pyo.Set(initialize=[f'Sub{j+1}' for j in range(self.generator.num_substations)])
            model.K = pyo.Set(initialize=[f'City{k+1}' for k in range(self.generator.num_customers)])
            model.S = pyo.Set(initialize=self.generator.seasons)
            
            # Parameters
            model.PC = pyo.Param(model.I, model.J, initialize=self.data['production_cost'])
            model.FC = pyo.Param(model.J, initialize=self.data['fixed_cost'])
            model.Cap = pyo.Param(model.I, initialize=self.data['facility_capacity'])
            model.SC = pyo.Param(model.J, initialize=self.data['substation_capacity'])
            model.D = pyo.Param(model.K, model.S, initialize=self.data['demand'])
            model.alpha = pyo.Param(model.I, model.J, initialize=self.data['alpha'])
            model.beta = pyo.Param(model.J, model.K, initialize=self.data['beta'])
            
            # Variables - with relaxation option
            if relaxed:
                model.y = pyo.Var(model.J, bounds=(0,1))
                model.x = pyo.Var(model.J, model.K, bounds=(0,1))
            else:
                model.y = pyo.Var(model.J, domain=pyo.Binary)
                model.x = pyo.Var(model.J, model.K, domain=pyo.Binary)
                
            model.f_ij = pyo.Var(model.I, model.J, model.S, domain=pyo.NonNegativeReals)
            model.f_jk = pyo.Var(model.J, model.K, model.S, domain=pyo.NonNegativeReals)
            
            # Fix variables if provided
            if fixed_vars:
                for (var_type, var_id), value in fixed_vars.items():
                    if var_type == 'y':
                        model.y[var_id].fix(value)
                    elif var_type == 'x':
                        j, k = var_id
                        model.x[j,k].fix(value)
                
            # Objective function
            def obj_rule(model):
                return (sum(model.FC[j] * model.y[j] for j in model.J) +
                       sum(model.PC[i,j] * model.f_ij[i,j,s] 
                           for i in model.I for j in model.J for s in model.S))
            model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
            
            # Constraints
            def flow_conservation_rule(model, j, s):
                return (sum(model.f_ij[i,j,s] * model.alpha[i,j] for i in model.I) == 
                       sum(model.f_jk[j,k,s] for k in model.K))
            model.flow_conservation = pyo.Constraint(model.J, model.S, rule=flow_conservation_rule)
            
            def demand_satisfaction_rule(model, k, s):
                return (sum(model.f_jk[j,k,s] * model.beta[j,k] for j in model.J) >= 
                       model.D[k,s])
            model.demand_satisfaction = pyo.Constraint(model.K, model.S, rule=demand_satisfaction_rule)
            
            def production_capacity_rule(model, i, s):
                return sum(model.f_ij[i,j,s] for j in model.J) <= model.Cap[i]
            model.production_capacity = pyo.Constraint(model.I, model.S, rule=production_capacity_rule)
            
            def substation_capacity_rule(model, j, s):
                return sum(model.f_jk[j,k,s] for k in model.K) <= model.SC[j] * model.y[j]
            model.substation_capacity = pyo.Constraint(model.J, model.S, rule=substation_capacity_rule)
            
            def single_assignment_rule(model, k):
                return sum(model.x[j,k] for j in model.J) == 1
            model.single_assignment = pyo.Constraint(model.K, rule=single_assignment_rule)
            
            def assignment_to_open_rule(model, j, k):
                return model.x[j,k] <= model.y[j]
            model.assignment_to_open = pyo.Constraint(model.J, model.K, rule=assignment_to_open_rule)
            
            def flow_assignment_rule(model, j, k, s):
                M = max(self.data['demand'].values()) * 2
                return model.f_jk[j,k,s] <= M * model.x[j,k]
            model.flow_assignment = pyo.Constraint(model.J, model.K, model.S, rule=flow_assignment_rule)
            
            return model
            
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            return None
    
    def solve_single_stage(self):
        """Solve the single-stage optimization problem"""
        model = self.create_model(relaxed=False)
        if model is None:
            return None
            
        try:
            solver = SolverFactory('gurobi')
            results = solver.solve(model, tee=True)

            #  # Add solver parameters for better numerical stability
            # solver_options = {
            # 'FeasibilityTol': 1e-5,    # Relaxed from default 1e-6
            # 'IntFeasTol': 1e-4,        # Relaxed from default 1e-5
            # 'OptimalityTol': 1e-5,     # Relaxed from default 1e-6
            # 'NumericFocus': 3,         # Maximum numerical precision
            # 'Method': 2,               # Barrier method
            # 'BarConvTol': 1e-5,        # Barrier convergence tolerance
            # 'MIPGap': 1e-4             # Slightly relaxed MIP gap
            # }
            
            # Check solution status
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                # Collect solution details
                solution = self.extract_solution(model)
                solution['objective_value'] = pyo.value(model.objective)
                solution['opened_substations'] = [j for j in model.J if pyo.value(model.y[j]) > 0.5]
                solution['assignments'] = {k: next(j for j in model.J if pyo.value(model.x[j,k]) > 0.5) 
                                        for k in model.K}
                return solution
            else:
                print(f"Solver status: {results.solver.status}")
                print(f"Termination condition: {results.solver.termination_condition}")
                return None
                
        except Exception as e:
            print(f"Error solving model: {str(e)}")
            return None
        
    def extract_solution(self, model):
        """Helper to extract solution from model in standard format"""
        return {
            'y': {j: pyo.value(model.y[j]) for j in model.J},
            'x': {(j,k): pyo.value(model.x[j,k]) for j in model.J for k in model.K},
            'flows': {
                'facility_to_substation': {
                    (i,j,s): pyo.value(model.f_ij[i,j,s])
                    for i in model.I for j in model.J for s in model.S
                },
                'substation_to_customer': {
                    (j,k,s): pyo.value(model.f_jk[j,k,s])
                    for j in model.J for k in model.K for s in model.S
                }
            }
        }