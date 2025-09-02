# create_db_from_excel.py
import sqlite3
import pandas as pd
file_path = "C://Users//CSC\Desktop//cms_data//Quarter4.xlsx"
conn = sqlite3.connect("claims.db")

# Load and store each sheet
provider_df = pd.read_excel(file_path, sheet_name="Provider_Table")
client_df = pd.read_excel(file_path, sheet_name="Client_Table")
claim_df = pd.read_excel(file_path, sheet_name="Claim_Table")

provider_df.to_sql("Provider", conn, if_exists="replace", index=False)
client_df.to_sql("Client", conn, if_exists="replace", index=False)
claim_df.to_sql("Claim", conn, if_exists="replace", index=False)

conn.close()
print("Database created with tables: Provider, Client, Claim")