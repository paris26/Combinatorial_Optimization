import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

class ThermalNetworkOptimizer:
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.generate_all_data()
        
    def create_model(self, min_facility_per_sub=1, max_facility_per_sub=None):
        """Create the optimization model with distance metric selection."""
        model = pyo.ConcreteModel()
        
        # Sets
        model.I = pyo.Set(initialize=[f'Plant{i+1}' for i in range(self.generator.num_facilities)])
        model.J = pyo.Set(initialize=[f'Sub{j+1}' for j in range(self.generator.num_substations)])
        model.K = pyo.Set(initialize=[f'City{k+1}' for k in range(self.generator.num_customers)])
        model.S = pyo.Set(initialize=self.generator.seasons)
        
        # Binary variables for distance metric selection
        model.d_ij = pyo.Var(model.I, model.J, domain=pyo.Binary)  # 1 if manhattan, 0 if euclidean
        model.d_jk = pyo.Var(model.J, model.K, domain=pyo.Binary)  # 1 if manhattan, 0 if euclidean
        
        # Parameters for both metrics
        def init_pc(model, i, j, metric):
            return self.data['production_cost'][metric][(i,j)]
        model.PC = pyo.Param(model.I, model.J, ['euclidean', 'manhattan'], initialize=init_pc)
        
        model.FC = pyo.Param(model.J, initialize=self.data['fixed_cost'])
        model.Cap = pyo.Param(model.I, initialize=self.data['facility_capacity'])
        model.SC = pyo.Param(model.J, initialize=self.data['substation_capacity'])
        model.D = pyo.Param(model.K, model.S, initialize=self.data['demand'])
        
        # Heat loss coefficients for both metrics
        def init_alpha(model, i, j, metric):
            return self.data['heat_loss_coefficients'][metric]['alpha'][(i,j)]
        def init_beta(model, j, k, metric):
            return self.data['heat_loss_coefficients'][metric]['beta'][(j,k)]
            
        model.alpha_e = pyo.Param(model.I, model.J, initialize=lambda m,i,j: init_alpha(m,i,j,'euclidean'))
        model.alpha_m = pyo.Param(model.I, model.J, initialize=lambda m,i,j: init_alpha(m,i,j,'manhattan'))
        model.beta_e = pyo.Param(model.J, model.K, initialize=lambda m,j,k: init_beta(m,j,k,'euclidean'))
        model.beta_m = pyo.Param(model.J, model.K, initialize=lambda m,j,k: init_beta(m,j,k,'manhattan'))
        
        # Variables
        model.y = pyo.Var(model.J, domain=pyo.Binary)
        model.x = pyo.Var(model.J, model.K, domain=pyo.Binary)
        model.f_ij = pyo.Var(model.I, model.J, model.S, domain=pyo.NonNegativeReals)
        model.f_jk = pyo.Var(model.J, model.K, model.S, domain=pyo.NonNegativeReals)
        
        # Binary variables for facility-substation connections
        model.z_ij = pyo.Var(model.I, model.J, domain=pyo.Binary)
        
        # Objective function with metric selection
        def obj_rule(model):
            return (
                sum(model.FC[j] * model.y[j] for j in model.J) +
                sum(
                    (model.PC[i,j,'manhattan'] * model.d_ij[i,j] + 
                     model.PC[i,j,'euclidean'] * (1-model.d_ij[i,j])) * model.f_ij[i,j,s]
                    for i in model.I for j in model.J for s in model.S
                )
            )
        model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        # Flow conservation with dynamic heat loss coefficients
    def flow_conservation_rule(model, j, s):
        epsilon = 1e-6
        expr = (sum(model.f_ij[i,j,s] * (
            model.alpha_m[i,j] * model.d_ij[i,j] + 
            model.alpha_e[i,j] * (1-model.d_ij[i,j])
            ) for i in model.I) - 
            sum(model.f_jk[j,k,s] for k in model.K))
        return -epsilon <= expr <= epsilon
        model.flow_conservation = pyo.Constraint(model.J, model.S, rule=flow_conservation_rule)
        
        # Demand satisfaction with dynamic heat loss coefficients
        def demand_satisfaction_rule(model, k, s):
            return (
                sum(
                    model.f_jk[j,k,s] * (
                        model.beta_m[j,k] * model.d_jk[j,k] + 
                        model.beta_e[j,k] * (1-model.d_jk[j,k])
                    ) for j in model.J
                ) >= model.D[k,s]
            )
        model.demand_satisfaction = pyo.Constraint(model.K, model.S, rule=demand_satisfaction_rule)
        
        # Production capacity
        def production_capacity_rule(model, i, s):
            return sum(model.f_ij[i,j,s] for j in model.J) <= model.Cap[i]
        model.production_capacity = pyo.Constraint(model.I, model.S, rule=production_capacity_rule)
        
        # Substation capacity
        def substation_capacity_rule(model, j, s):
            return sum(model.f_jk[j,k,s] for k in model.K) <= model.SC[j] * model.y[j]
        model.substation_capacity = pyo.Constraint(model.J, model.S, rule=substation_capacity_rule)
        
        # Single assignment
        def single_assignment_rule(model, k):
            return sum(model.x[j,k] for j in model.J) == 1
        model.single_assignment = pyo.Constraint(model.K, rule=single_assignment_rule)
        
        # Flow assignment
        def flow_assignment_rule(model, j, k, s):
            M = max(self.data['demand'].values()) * 2
            return model.f_jk[j,k,s] <= M * model.x[j,k]
        model.flow_assignment = pyo.Constraint(model.J, model.K, model.S, rule=flow_assignment_rule)
        
        # Connection indicator constraints
        def connection_indicator_rule1(model, i, j):
            M = max(self.data['demand'].values()) * 2
            return sum(model.f_ij[i,j,s] for s in model.S) <= M * model.z_ij[i,j]
        model.connection_indicator1 = pyo.Constraint(model.I, model.J, rule=connection_indicator_rule1)
        
        def connection_indicator_rule2(model, i, j):
            epsilon = 1e-6
            return sum(model.f_ij[i,j,s] for s in model.S) >= epsilon * model.z_ij[i,j]
        model.connection_indicator2 = pyo.Constraint(model.I, model.J, rule=connection_indicator_rule2)
        
        # Minimum and maximum facilities per substation
        def min_facility_per_sub_rule(model, j):
            return sum(model.z_ij[i,j] for i in model.I) >= min_facility_per_sub * model.y[j]
        model.min_facility_per_sub = pyo.Constraint(model.J, rule=min_facility_per_sub_rule)
        
        if max_facility_per_sub:
            def max_facility_per_sub_rule(model, j):
                return sum(model.z_ij[i,j] for i in model.I) <= max_facility_per_sub * model.y[j]
            model.max_facility_per_sub = pyo.Constraint(model.J, rule=max_facility_per_sub_rule)

        return model
    
    def solve_single_stage(self, min_facility_per_sub=1, max_facility_per_sub=None):
        model = self.create_model(min_facility_per_sub, max_facility_per_sub)
        
        try:
            solver = SolverFactory('gurobi')
            results = solver.solve(model, tee=True)
            
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                solution = {
                    'objective_value': pyo.value(model.objective),
                    'opened_substations': [j for j in model.J if pyo.value(model.y[j]) > 0.5],
                    'assignments': {k: next(j for j in model.J if pyo.value(model.x[j,k]) > 0.5) 
                                  for k in model.K},
                    'distance_metrics': {
                        'facility_substation': {(i,j): 'manhattan' if pyo.value(model.d_ij[i,j]) > 0.5 else 'euclidean'
                                              for i in model.I for j in model.J},
                        'substation_customer': {(j,k): 'manhattan' if pyo.value(model.d_jk[j,k]) > 0.5 else 'euclidean'
                                              for j in model.J for k in model.K}
                    },
                    'flows': {
                        'facility_to_substation': {(i,j,s): pyo.value(model.f_ij[i,j,s])
                                                 for i in model.I for j in model.J for s in model.S},
                        'substation_to_customer': {(j,k,s): pyo.value(model.f_jk[j,k,s])
                                                 for j in model.J for k in model.K for s in model.S}
                    },
                    'facility_connections': {
                        'connections': {(i,j): pyo.value(model.z_ij[i,j]) > 0.5
                                      for i in model.I for j in model.J},
                        'facilities_per_sub': {j: sum(1 for i in model.I
                                                    if pyo.value(model.z_ij[i,j]) > 0.5)
                                             for j in model.J}
                    }
                }
                return solution
            else:
                print(f"Solver status: {results.solver.status}")
                print(f"Termination condition: {results.solver.termination_condition}")
                return None
                
        except Exception as e:
            print(f"Error solving model: {str(e)}")
            return None