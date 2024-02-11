import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

cwd = os.getcwd()
data_base_path = os.path.join(cwd, "orlen_stock_v2.db")

def connect_db(path):
    db = sqlite3.connect(path)
    return db

con = connect_db(data_base_path)
df = pd.read_sql_query("SELECT * from pkn_stock_v2 ORDER by date ASC", con)

# Verify that result of SQL query is stored in the dataframe
print(df.head())
con.close()

plt.plot(df["date"], df["close"])
plt.show()