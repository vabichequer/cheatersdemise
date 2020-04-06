import pandas as pd
import numpy as np
from ast import literal_eval

ta = np.load(str(tol) + '/' + str(tol) + '.npy')

#df_type_array = pd.read_csv('5/5.csv', index_col=0)
#df_type_array.fillna('', inplace=True)

#users = literal_eval(df_type_array['0'].to_numpy())
#td = literal_eval(df_type_array['1'].to_numpy())
#ub = literal_eval(df_type_array['2'].to_numpy())
#total_ex = literal_eval(df_type_array['3'].to_numpy())

print(ta)

