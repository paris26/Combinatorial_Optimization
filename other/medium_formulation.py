import pyomo.environ as pyo
from pyomo.opt import SolverFactory

model = pyo.ConcreteModel()

# SETS
model.I = pyo.Set(initialize=['Plant1', 'Plant2'])
model.J = pyo.Set(initialize=['Sub1', 'Sub2', 'Sub3'])
model.K = pyo.Set(initialize=['City1', 'City2', 'City3'])
model.S = pyo.Set(initialize=['Winter', 'Summer', 'Fall', 'Spring'])

# PARAMETERS
# Production Cost
production_cost = {
    ('Plant1', 'Sub1'): 50,
    ('Plant1', 'Sub2'): 60,
    ('Plant1', 'Sub3'): 70,
    ('Plant2', 'Sub1'): 55,
    ('Plant2', 'Sub2'): 65,
    ('Plant2', 'Sub3'): 75
}

# Fixed Costs for Substations
fixed_cost = {
    'Sub1': 1000,
    'Sub2': 1200,
    'Sub3': 1500
}

# Capacity of Production Facilities
facility_capacity = {
    'Plant1': 1000,
    'Plant2': 1200
}

# Substation Capacity
substation_capacity = {
    'Sub1': 500,
    'Sub2': 600,
    'Sub3': 700
}

# Customer Demand per Season
demand = {
    ('City1', 'Winter'): 200,
    ('City1', 'Summer'): 40,
    ('City1', 'Fall'): 120,
    ('City1', 'Spring'): 120,
    ('City2', 'Winter'): 300,
    ('City2', 'Summer'): 60,
    ('City2', 'Fall'): 180,
    ('City2', 'Spring'): 180,
    ('City3', 'Winter'): 250,
    ('City3', 'Summer'): 50,
    ('City3', 'Fall'): 150,
    ('City3', 'Spring'): 150
}

# Heat Loss Coefficients (pre-calculated)
alpha = {  # Facility to Substation
    ('Plant1', 'Sub1'): 0.95,
    ('Plant1', 'Sub2'): 0.92,
    ('Plant1', 'Sub3'): 0.90,
    ('Plant2', 'Sub1'): 0.93,
    ('Plant2', 'Sub2'): 0.94,
    ('Plant2', 'Sub3'): 0.91
}

beta = {  # Substation to Customer
    ('Sub1', 'City1'): 0.94,
    ('Sub1', 'City2'): 0.92,
    ('Sub1', 'City3'): 0.91,
    ('Sub2', 'City1'): 0.93,
    ('Sub2', 'City2'): 0.95,
    ('Sub2', 'City3'): 0.92,
    ('Sub3', 'City1'): 0.90,
    ('Sub3', 'City2'): 0.93,
    ('Sub3', 'City3'): 0.94
}

# Parameter Initialization
model.PC = pyo.Param(model.I, model.J, initialize=production_cost)
model.FC = pyo.Param(model.J, initialize=fixed_cost)
model.Cap = pyo.Param(model.I, initialize=facility_capacity)
model.SC = pyo.Param(model.J, initialize=substation_capacity)
model.D = pyo.Param(model.K, model.S, initialize=demand)
model.alpha = pyo.Param(model.I, model.J, initialize=alpha)
model.beta = pyo.Param(model.J, model.K, initialize=beta)

# Variables
model.y = pyo.Var(model.J, domain=pyo.Binary)  # Substation opening decision
model.x = pyo.Var(model.J, model.K, domain=pyo.Binary)  # Customer assignment
model.f_ij = pyo.Var(model.I, model.J, model.S, domain=pyo.NonNegativeReals)  # Flow facility to substation
model.f_jk = pyo.Var(model.J, model.K, model.S, domain=pyo.NonNegativeReals)  # Flow substation to customer

# Objective Function
def obj_rule(model):
    return (sum(model.FC[j] * model.y[j] for j in model.J) +
            sum(model.PC[i,j] * model.f_ij[i,j,s] 
                for i in model.I for j in model.J for s in model.S))
model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Constraints
# 1. Flow Conservation
def flow_conservation_rule(model, j, s):
    return (sum(model.f_ij[i,j,s] * model.alpha[i,j] for i in model.I) == 
            sum(model.f_jk[j,k,s] for k in model.K))
model.flow_conservation = pyo.Constraint(model.J, model.S, rule=flow_conservation_rule)

# 2. Demand Satisfaction
def demand_satisfaction_rule(model, k, s):
    return (sum(model.f_jk[j,k,s] * model.beta[j,k] for j in model.J) >= 
            model.D[k,s])
model.demand_satisfaction = pyo.Constraint(model.K, model.S, rule=demand_satisfaction_rule)

# 3. Production Capacity
def production_capacity_rule(model, i, s):
    return sum(model.f_ij[i,j,s] for j in model.J) <= model.Cap[i]
model.production_capacity = pyo.Constraint(model.I, model.S, rule=production_capacity_rule)

# 4. Substation Capacity
def substation_capacity_rule(model, j, s):
    return sum(model.f_jk[j,k,s] for k in model.K) <= model.SC[j] * model.y[j]
model.substation_capacity = pyo.Constraint(model.J, model.S, rule=substation_capacity_rule)

# 5. Single Assignment
def single_assignment_rule(model, k):
    return sum(model.x[j,k] for j in model.J) == 1
model.single_assignment = pyo.Constraint(model.K, rule=single_assignment_rule)

# 6. Flow Only Through Assignment
def flow_assignment_rule(model, j, k, s):
    M = 10000  # Big M value
    return model.f_jk[j,k,s] <= M * model.x[j,k]
model.flow_assignment = pyo.Constraint(model.J, model.K, model.S, rule=flow_assignment_rule)

# Solve
solver = SolverFactory('gurobi')  
results = solver.solve(model, tee=True)

# Print Results
if results.solver.status == 'ok':
    print("\nOptimal Solution Found:")
    print("\nOpened Substations:")
    for j in model.J:
        if pyo.value(model.y[j]) > 0.5:
            print(f"Substation {j}")
            
    print("\nCustomer Assignments:")
    for j in model.J:
        for k in model.K:
            if pyo.value(model.x[j,k]) > 0.5:
                print(f"Customer {k} assigned to Substation {j}")
                
    print("\nObjective Value:", pyo.value(model.objective))
else:
    print("Problem could not be solved to optimality")