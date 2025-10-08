import pandas as pd
import numpy as np

df = pd.read_excel('output/Pharma_Ventes_Master_2018plusplus.xlsx')

print(df.head())
print(df.info())
print(df.describe())