# solve_problems.py

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
import json
import matplotlib.pyplot as plt
import networkx as nx

# Find all JSON files named 'problem_instance_*.json' in subdirectories
data_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.startswith('problem_instance_') and file.endswith('.json'):
            data_files.append(os.path.join(root, file))

data_files.sort()

# Solver to use
solver_name = 'gurobi'  # Replace with your preferred solver

# Function to flatten nested dictionaries
def flatten_nested_dict(nested_dict):
    flat_dict = {}
    for outer_key in nested_dict:
        for inner_key in nested_dict[outer_key]:
            flat_dict[(outer_key, inner_key)] = nested_dict[outer_key][inner_key]
    return flat_dict

# Function to plot and save the solved model as a PNG
def plot_and_save_model(model, problem_name):
    # Set up positions and graph structure
    positions = {}
    for idx, i in enumerate(model.I):
        positions[i] = (0, idx * 10)
    for idx, j in enumerate(model.J):
        positions[j] = (50, idx * 10)
    for idx, k in enumerate(model.K):
        positions[k] = (100, idx * 10)

    # Create the graph
    G = nx.DiGraph()
    G.add_nodes_from(model.I, node_type='Facility')
    G.add_nodes_from(model.J, node_type='Substation')
    G.add_nodes_from(model.K, node_type='Customer')

    # Add edges based on flows
    for i in model.I:
        for j in model.J:
            flow_ij = sum(pyo.value(model.f_ij[i, j, s]) for s in model.S)
            if flow_ij > 0.01:  # Threshold for negligible flows
                G.add_edge(i, j, weight=flow_ij)

    for j in model.J:
        for k in model.K:
            flow_jk = sum(pyo.value(model.f_jk[j, k, s]) for s in model.S)
            if flow_jk > 0.01:
                G.add_edge(j, k, weight=flow_jk)

    # Plot and save as PNG
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, with_labels=True, node_size=800, arrows=True)
    plt.title(f"Solution Network for {problem_name}")
    plt.savefig(f"{problem_name}_solution.png")  # Save as PNG
    plt.close()

# Process each data file
for filepath in data_files:
    problem_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\nSolving {problem_name}...")
    
    # Load data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Flatten data
    PC_data = flatten_nested_dict(data['PC'])
    Cap_data = flatten_nested_dict(data['Cap'])
    SC_data = flatten_nested_dict(data['SC'])
    D_data = flatten_nested_dict(data['D'])
    alpha_data = flatten_nested_dict(data['alpha'])
    beta_data = flatten_nested_dict(data['beta'])
    M_data = flatten_nested_dict(data['M'])
    
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
        # Plot and save the model as a PNG file
        plot_and_save_model(model, problem_name)
    else:
        print(f"{problem_name} did not solve to optimality. Solver Status: {results.solver.status}")
        continue  # Skip to next instance if not optimal
    
    # DISPLAY RESULTS
    total_cost = pyo.value(model.TotalCost)
    print(f"Total Cost for {problem_name}: {total_cost:,.2f}")
    
    print("\\nOpened Substations:")
    for j in model.J:
        if pyo.value(model.y[j]) > 0.5:
            print(f"Substation {j} is opened.")
    
    print("\\nCustomer Assignments:")
    for k in model.K:
        for j in model.J:
            if pyo.value(model.x[j, k]) > 0.5:
                print(f"Customer {k} is assigned to Substation {j}.")
