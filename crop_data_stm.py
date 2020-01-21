import pickle
from typing import Dict

dct: Dict = pickle.load(open("./finance_data/data_stm_2004_to_2017.pkl", "rb"))

new_dct: Dict = dict()

print("len keys", len(dct.keys()))


# This dict has two Keys, "Volume" and "Price"
# "Volume" contains a pd.DataFrame with a DatetimeIndex and the Cols: VIX Price   Gold Price   SPX Close   Label
# "Price"  contains a pd.DataFrame with a DatetimeIndex and the Cols: VIX Volume  Gold Volume  SPX Volume  Label

for key, datafr in dct.items():
    print(datafr)
    new_dct[key] = datafr.loc['2007-1-1 00:00:00':'2008-7-1 00:00:00']  # type: ignore

output = open('finance_data/data_stm.pkl', 'wb')
pickle.dump(new_dct, output)
output.close()

