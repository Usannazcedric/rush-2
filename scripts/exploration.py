import pandas as pd
import numpy as np

df = pd.read_excel('output/Pharma_Ventes_Hourly_CLEAN.xlsx')

print(df.head())
print(df.info())
print(df.describe())