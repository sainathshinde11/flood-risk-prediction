import pandas as pd

df = pd.read_csv("train.csv")

# Transpose first 5 rows — columns become rows
first5 = df.head(5).T
first5.columns = ['Record 1', 'Record 2', 'Record 3', 'Record 4', 'Record 5']
first5.index.name = 'Feature'

# Transpose last 5 rows
last5 = df.tail(5).T
last5.columns = ['Record 1', 'Record 2', 'Record 3', 'Record 4', 'Record 5']
last5.index.name = 'Feature'

# Save to Excel
with pd.ExcelWriter("first_last_transposed.xlsx") as writer:
    first5.to_excel(writer, sheet_name="First 5 Rows")
    last5.to_excel(writer, sheet_name="Last 5 Rows")

print("Done! Open first_last_transposed.xlsx")