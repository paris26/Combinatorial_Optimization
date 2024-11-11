import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Create the model
model = pyo.ConcreteModel()

# SETS

# I: Production Facilities
model.I = pyo.Set(initialize=['Plant1', 'Plant2'])

# J: Substations
model.J = pyo.Set(initialize=['Sub1', 'Sub2', 'Sub3'])

# K: Customers
model.K = pyo.Set(initialize=['City1', 'City2', 'City3'])

# S: Seasons
model.S = pyo.Set(initialize=['Winter', 'Summer', 'Fall', 'Spring'])

# PARAMETERS

# Production Cost (PC_{i,s})
production_cost_data = {
    ('Plant1', 'Winter'): 50,
    ('Plant1', 'Summer'): 45,
    ('Plant1', 'Fall'): 48,
    ('Plant1', 'Spring'): 47,
    ('Plant2', 'Winter'): 55,
    ('Plant2', 'Summer'): 50,
    ('Plant2', 'Fall'): 53,
    ('Plant2', 'Spring'): 52,
}
model.PC = pyo.Param(model.I, model.S, initialize=production_cost_data)

# Production Capacity (Cap_{i,s})
production_capacity_data = {
    ('Plant1', 'Winter'): 1000,
    ('Plant1', 'Summer'): 800,
    ('Plant1', 'Fall'): 900,
    ('Plant1', 'Spring'): 850,
    ('Plant2', 'Winter'): 1100,
    ('Plant2', 'Summer'): 900,
    ('Plant2', 'Fall'): 950,
    ('Plant2', 'Spring'): 920,
}
model.Cap = pyo.Param(model.I, model.S, initialize=production_capacity_data)

# Fixed Cost of opening substation (FC_j)
fixed_cost_data = {
    'Sub1': 10000,
    'Sub2': 12000,
    'Sub3': 11000,
}
model.FC = pyo.Param(model.J, initialize=fixed_cost_data)

# Substation Capacity (SC_{j,s})
substation_capacity_data = {}
for j in model.J:
    for s in model.S:
        substation_capacity_data[(j, s)] = 1000  # Assuming constant capacity
model.SC = pyo.Param(model.J, model.S, initialize=substation_capacity_data)

# Demand of customer k in season s (D_{k,s})
demand_data = {
    ('City1', 'Winter'): 300,
    ('City1', 'Summer'): 200,
    ('City1', 'Fall'): 250,
    ('City1', 'Spring'): 240,
    ('City2', 'Winter'): 400,
    ('City2', 'Summer'): 250,
    ('City2', 'Fall'): 300,
    ('City2', 'Spring'): 280,
    ('City3', 'Winter'): 350,
    ('City3', 'Summer'): 220,
    ('City3', 'Fall'): 270,
    ('City3', 'Spring'): 260,
}
model.D = pyo.Param(model.K, model.S, initialize=demand_data)

# Heat loss coefficients (alpha_{ij} and beta_{jk})
# For simplicity, let's define distances and calculate coefficients
# Distance between facilities and substations (d_{ij})
distance_ij_data = {
    ('Plant1', 'Sub1'): 10,
    ('Plant1', 'Sub2'): 20,
    ('Plant1', 'Sub3'): 15,
    ('Plant2', 'Sub1'): 25,
    ('Plant2', 'Sub2'): 10,
    ('Plant2', 'Sub3'): 20,
}

# Distance between substations and customers (d_{jk})
distance_jk_data = {
    ('Sub1', 'City1'): 5,
    ('Sub1', 'City2'): 15,
    ('Sub1', 'City3'): 10,
    ('Sub2', 'City1'): 10,
    ('Sub2', 'City2'): 5,
    ('Sub2', 'City3'): 15,
    ('Sub3', 'City1'): 15,
    ('Sub3', 'City2'): 10,
    ('Sub3', 'City3'): 5,
}

# Heat loss rate per unit distance (lambda)
lambda_value = 0.02

# Calculate alpha_{ij}
def alpha_rule(model, i, j):
    d = distance_ij_data[(i, j)]
    return pyo.exp(-lambda_value * d)
model.alpha = pyo.Param(model.I, model.J, initialize=alpha_rule, within=pyo.Reals)

# Calculate beta_{jk}
def beta_rule(model, j, k):
    d = distance_jk_data[(j, k)]
    return pyo.exp(-lambda_value * d)
model.beta = pyo.Param(model.J, model.K, initialize=beta_rule, within=pyo.Reals)

# Big M values for constraints
# Let's calculate M_{jk} = max_s (D_{k,s} / beta_{jk})
M_value = {}
for j in model.J:
    for k in model.K:
        max_D_over_beta = max(model.D[k, s] / model.beta[j, k] for s in model.S)
        M_value[(j, k)] = max_D_over_beta
model.M = pyo.Param(model.J, model.K, initialize=M_value)

# VARIABLES

# Binary variables y_j
model.y = pyo.Var(model.J, within=pyo.Binary)

# Binary variables x_{jk}
model.x = pyo.Var(model.J, model.K, within=pyo.Binary)

# Flow variables f_{ij,s}
model.f_ij = pyo.Var(model.I, model.J, model.S, within=pyo.NonNegativeReals)

# Flow variables f_{jk,s}
model.f_jk = pyo.Var(model.J, model.K, model.S, within=pyo.NonNegativeReals)

# OBJECTIVE FUNCTION

def objective_rule(model):
    return sum(model.FC[j] * model.y[j] for j in model.J) + \
           sum(model.PC[i, s] * model.f_ij[i, j, s] for i in model.I for j in model.J for s in model.S)
model.TotalCost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# CONSTRAINTS

# 1. Flow Conservation at Substations
def flow_conservation_rule(model, j, s):
    return sum(model.f_ij[i, j, s] * model.alpha[i, j] for i in model.I) == \
           sum(model.f_jk[j, k, s] for k in model.K)
model.FlowConservation = pyo.Constraint(model.J, model.S, rule=flow_conservation_rule)

# 2. Demand Satisfaction
def demand_satisfaction_rule(model, k, s):
    return sum(model.f_jk[j, k, s] * model.beta[j, k] for j in model.J) >= model.D[k, s]
model.DemandSatisfaction = pyo.Constraint(model.K, model.S, rule=demand_satisfaction_rule)

# 3. Production Capacity Limits
def production_capacity_rule(model, i, s):
    return sum(model.f_ij[i, j, s] for j in model.J) <= model.Cap[i, s]
model.ProductionCapacity = pyo.Constraint(model.I, model.S, rule=production_capacity_rule)

# 4. Substation Capacity Limits
def substation_capacity_rule(model, j, s):
    return sum(model.f_jk[j, k, s] for k in model.K) <= model.SC[j, s] * model.y[j]
model.SubstationCapacity = pyo.Constraint(model.J, model.S, rule=substation_capacity_rule)

# 5. Customer Assignment
def customer_assignment_rule(model, k):
    return sum(model.x[j, k] for j in model.J) == 1
model.CustomerAssignment = pyo.Constraint(model.K, rule=customer_assignment_rule)

# 6. Assignment Only to Open Substations
def assignment_open_substation_rule(model, j, k):
    return model.x[j, k] <= model.y[j]
model.AssignmentOpenSubstation = pyo.Constraint(model.J, model.K, rule=assignment_open_substation_rule)

# 7. Flow Only Through Assigned Substations
def flow_assignment_rule(model, j, k, s):
    return model.f_jk[j, k, s] <= model.M[j, k] * model.x[j, k]
model.FlowAssignment = pyo.Constraint(model.J, model.K, model.S, rule=flow_assignment_rule)

# SOLVE THE MODEL

# Create a solver
solver = SolverFactory('gurobi')  

# Solve the model
results = solver.solve(model, tee=True)

# DISPLAY RESULTS

# Print the objective function value
print(f"Total Cost: {pyo.value(model.TotalCost)}")

# Print which substations are opened
print("Opened Substations:")
for j in model.J:
    if pyo.value(model.y[j]) > 0.5:
        print(f"Substation {j} is opened.")

# Print customer assignments
print("Customer Assignments:")
for k in model.K:
    for j in model.J:
        if pyo.value(model.x[j, k]) > 0.5:
            print(f"Customer {k} is assigned to Substation {j}.")

# Print flows from facilities to substations
print("Flows from Facilities to Substations:")
for i in model.I:
    for j in model.J:
        for s in model.S:
            flow_value = pyo.value(model.f_ij[i, j, s])
            if flow_value > 1e-6:
                print(f"Flow from {i} to {j} in {s}: {flow_value}")

# Print flows from substations to customers
print("Flows from Substations to Customers:")
for j in model.J:
    for k in model.K:
        for s in model.S:
            flow_value = pyo.value(model.f_jk[j, k, s])
            if flow_value > 1e-6:
                print(f"Flow from {j} to {k} in {s}: {flow_value}")
