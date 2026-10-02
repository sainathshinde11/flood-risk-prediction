import pandas as pd

df = pd.read_csv("train.csv")

print("Shape:", df.shape)

# Save first 5 and last 5 to one Excel file
with pd.ExcelWriter("first_last_rows.xlsx") as writer:
    df.head(5).to_excel(writer, sheet_name="First 5 Rows", index=False)
    df.tail(5).to_excel(writer, sheet_name="Last 5 Rows", index=False)

print("Done! Open first_last_rows.xlsx")