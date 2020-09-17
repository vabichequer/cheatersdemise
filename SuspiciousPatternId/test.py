import pandas as pd

all = pd.read_pickle('original_data/final_df-2.pkl')
x = pd.read_pickle('original_data/final_df_with_innocents-2.pkl')

i1 = all.index
i2 = x.index

print(i1.intersection(i2))