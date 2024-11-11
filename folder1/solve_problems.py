# solve_problems.py

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
import json

# Directory where data files are stored
data_directory = 'problem_instances'

# List of data files to process
data_files = [f for f in os.listdir(data_directory) if f.endswith('.json')]
data_files.sort()  # Sort files to process them in order

# Solver to use
solver_name = 'gurobi'  # Replace with your preferred solver

# Process each data file
for data_file in data_files:
    problem_name = data_file.replace('.json', '')
    print(f"\nSolving {problem_name}...")
    
    # Load data
    filepath = os.path.join(data_directory, data_file)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct the tuple keys from strings
    def convert_keys_to_tuples(data_dict):
        new_dict = {}
        for key_str, value in data_dict.items():
            # Remove parentheses if present
            key_str = key_str.strip("()")
            # Split the string into components
            key_parts = key_str.split(", ")
            # Remove quotes around elements if any
            key_parts = [part.strip("'") for part in key_parts]
            # Reconstruct the tuple key
            key = tuple(key_parts)
            new_dict[key] = value
        return new_dict
    
    # Reconstruct data with tuple keys
    PC_data = convert_keys_to_tuples(data['PC'])
    Cap_data = convert_keys_to_tuples(data['Cap'])
    SC_data = convert_keys_to_tuples(data['SC'])
    D_data = convert_keys_to_tuples(data['D'])
    alpha_data = convert_keys_to_tuples(data['alpha'])
    beta_data = convert_keys_to_tuples(data['beta'])
    M_data = convert_keys_to_tuples(data['M'])
    
    # Create the model
    model = pyo.ConcreteModel()
    
    # SETS
    model.I = pyo.Set(initialize=data['I'])
    model.J = pyo.Set(initialize=data['J'])
    model.K = pyo.Set(initialize=data['K'])
    model.S = pyo.Set(initialize=data['S'])
    
    # PARAMETERS
    model.PC = pyo.Param(model.I, model.S, initialize=PC_data)
    model.Cap = pyo.Param(model.I, model.S, initialize=Cap_data)
    model.FC = pyo.Param(model.J, initialize=data['FC'])
    model.SC = pyo.Param(model.J, model.S, initialize=SC_data)
    model.D = pyo.Param(model.K, model.S, initialize=D_data)
    model.alpha = pyo.Param(model.I, model.J, initialize=alpha_data)
    model.beta = pyo.Param(model.J, model.K, initialize=beta_data)
    model.M = pyo.Param(model.J, model.K, initialize=M_data)
    
    # VARIABLES
    model.y = pyo.Var(model.J, within=pyo.Binary)
    model.x = pyo.Var(model.J, model.K, within=pyo.Binary)
    model.f_ij = pyo.Var(model.I, model.J, model.S, within=pyo.NonNegativeReals)
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
    solver = SolverFactory(solver_name)
    results = solver.solve(model, tee=False)
    
    # Check solver status
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print(f"{problem_name} solved to optimality.")
    else:
        print(f"{problem_name} did not solve to optimality. Solver Status: {results.solver.status}")
        continue  # Skip to next instance if not optimal
    
    # DISPLAY RESULTS
    total_cost = pyo.value(model.TotalCost)
    print(f"Total Cost for {problem_name}: {total_cost:,.2f}")
    
    print("\nOpened Substations:")
    for j in model.J:
        if pyo.value(model.y[j]) > 0.5:
            print(f"Substation {j} is opened.")
    
    print("\nCustomer Assignments:")
    for k in model.K:
        for j in model.J:
            if pyo.value(model.x[j, k]) > 0.5:
                print(f"Customer {k} is assigned to Substation {j}.")
