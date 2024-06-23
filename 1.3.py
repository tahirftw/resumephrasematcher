import pandas as pd
import pulp

# Load the dataset (simulated for this example)
data = {
    "Time Windows": ["9:00-10:00", "10:00 – 11:00", "11:00 – 12:00", "12:00 – 13:00",
                     "13:00 – 14:00", "14:00 – 15:00", "15:00-16:00", "16:00 – 17:00"],
    "Shift 1": ["X", "X", "X", "X", "", "", "", ""],
    "Shift 2": ["", "", "", "", "X", "X", "X", "X"],
    "Avg_Customer_Number": [28, 35, 21, 46, 32, 14, 24, 32]
}
df = pd.DataFrame(data)

# Define the problem
prob = pulp.LpProblem("FAU_Bank_Tellers_Optimization", pulp.LpMinimize)

# Define decision variables
x1 = pulp.LpVariable('x1', lowBound=0, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, cat='Integer')

# Define the objective function
prob += 16 * x1 + 14 * x2, "Total Wage Cost"

# Define the constraints based on the dataset
customer_demands_shift_1 = df['Avg_Customer_Number'][df['Shift 1'] == 'X']
customer_demands_shift_2 = df['Avg_Customer_Number'][df['Shift 2'] == 'X']

for demand in customer_demands_shift_1:
    prob += 8 * x1 >= demand

for demand in customer_demands_shift_2:
    prob += 8 * x2 >= demand

# Solve the problem
prob.solve()

# Get the results
tellers_shift_1 = pulp.value(x1)
tellers_shift_2 = pulp.value(x2)
total_wage_cost = pulp.value(prob.objective)

print(f"Number of Tellers for Shift 1 (9:00-13:00): {tellers_shift_1} tellers")
print(f"Number of Tellers for Shift 2 (13:00-17:00): {tellers_shift_2} tellers")
print(f"Total Wage Cost: {total_wage_cost} EUR")
