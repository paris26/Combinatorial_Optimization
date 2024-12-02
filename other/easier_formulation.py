import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Create a concrete model
model = pyo.ConcreteModel()

# Sets
# Time periods (e.g., months)
model.T = pyo.Set(initialize=[1, 2, 3])
# Production plants
model.P = pyo.Set(initialize=['Plant1', 'Plant2'])
# Demand centers
model.D = pyo.Set(initialize=['City1', 'City2', 'City3'])

# Parameters
# Production capacity for each plant (MW)
production_capacity = {
    'Plant1': 1000,
    'Plant2': 800
}
model.production_capacity = pyo.Param(model.P, initialize=production_capacity)

# Production cost per MW at each plant
production_cost = {
    'Plant1': 50,  # $/MW
    'Plant2': 45   # $/MW
}
model.production_cost = pyo.Param(model.P, initialize=production_cost)

# Demand at each center for each time period (MW)
demand_data = {
    ('City1', 1): 400, ('City1', 2): 450, ('City1', 3): 500,
    ('City2', 1): 300, ('City2', 2): 350, ('City2', 3): 400,
    ('City3', 1): 250, ('City3', 2): 300, ('City3', 3): 350
}
model.demand = pyo.Param(model.D, model.T, initialize=demand_data)

# Transportation cost per MW per km
# Simplified distance-based cost between plants and demand centers
transport_cost = {
    ('Plant1', 'City1'): 2,  # $/MW
    ('Plant1', 'City2'): 3,
    ('Plant1', 'City3'): 4,
    ('Plant2', 'City1'): 3,
    ('Plant2', 'City2'): 2,
    ('Plant2', 'City3'): 3
}
model.transport_cost = pyo.Param(model.P, model.D, initialize=transport_cost)

# Variables
# Production amount at each plant in each time period
model.production = pyo.Var(model.P, model.T, domain=pyo.NonNegativeReals)
# Amount transported from each plant to each demand center in each time period
model.transport = pyo.Var(model.P, model.D, model.T, domain=pyo.NonNegativeReals)

# Objective function: Minimize total cost
def objective_rule(model):
    production_costs = sum(model.production_cost[p] * model.production[p,t] 
                         for p in model.P for t in model.T)
    transport_costs = sum(model.transport_cost[p,d] * model.transport[p,d,t]
                        for p in model.P for d in model.D for t in model.T)
    return production_costs + transport_costs

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints
# Production capacity constraint
def production_capacity_rule(model, p, t):
    return model.production[p,t] <= model.production_capacity[p]

model.production_capacity_constraint = pyo.Constraint(model.P, model.T, 
                                                    rule=production_capacity_rule)

# Demand satisfaction constraint
def demand_satisfaction_rule(model, d, t):
    supply = sum(model.transport[p,d,t] for p in model.P)
    return supply == model.demand[d,t]

model.demand_satisfaction = pyo.Constraint(model.D, model.T, 
                                         rule=demand_satisfaction_rule)

# Production balance constraint
def production_balance_rule(model, p, t):
    production = model.production[p,t]
    shipments = sum(model.transport[p,d,t] for d in model.D)
    return production == shipments

model.production_balance = pyo.Constraint(model.P, model.T, 
                                        rule=production_balance_rule)

# Solve the model
def solve_model():
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=True)
    
    # Print results
    if results.solver.status == 'ok':
        print("\nOptimal Solution Found!")
        print("\nProduction Schedule:")
        for t in model.T:
            print(f"\nTime Period {t}:")
            for p in model.P:
                print(f"{p}: {pyo.value(model.production[p,t]):.2f} MW")
        
        print("\nTransportation Schedule:")
        for t in model.T:
            print(f"\nTime Period {t}:")
            for p in model.P:
                for d in model.D:
                    if pyo.value(model.transport[p,d,t]) > 0.1:
                        print(f"{p} to {d}: {pyo.value(model.transport[p,d,t]):.2f} MW")
        
        print(f"\nTotal Cost: ${pyo.value(model.objective):,.2f}")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    solve_model()