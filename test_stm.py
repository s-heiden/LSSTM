from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LSSTM.LS_STM import LSSTM
from hottbox.core import Tensor
from pandas._libs.tslibs import timestamps

from LSSTM.aiding_functions import load_obj
from LSSTM.data_makers import create_stm_slice

# Load Data

# This dict has two Keys, "Volume" and "Price"
# "Volume" contains a pd.DataFrame with a DatetimeIndex and the Cols: VIX Price   Gold Price   SPX Close   Label
# "Price"  contains a pd.DataFrame with a DatetimeIndex and the Cols: VIX Volume  Gold Volume  SPX Volume  Label
data_stm = load_obj("data_stm")

num_rows = len(data_stm["Price"])
tensor_shape = [2, 3, 2]
slice_width = 250

num_slices = num_rows - slice_width - tensor_shape[0] + 1  # TODO: why?

y_pred = np.zeros(num_slices - 1)  # I guess this is because there is no prediction for the last row

stm = LSSTM(C=10, max_iter=100)
success = 0
pred_dates_idcs = []

# iteratively calculate slices
for i in range(num_slices - 1): # called num_slices times with i as the i_th iteration

    # print progress
    print("\r{0}".format((float(i) / (num_slices - 1)) * 100))

    # prepare data ## TODO adapt to my data
    stm_slice: Tuple[List[Tensor], np.array, Tensor, np.float64, timestamps.Timestamp] = create_stm_slice(
        d=data_stm, start_index=i, slice_width=slice_width, tensor_shape=tensor_shape
    )
    xs_train, y_train, xs_test, y_test, associated_index = stm_slice

    # fit a model
    stm.fit(xs_train, y_train)
    y_tmp, _ = stm.predict(xs_test)
    y_pred[i] = y_tmp[0]

    #
    pred_dates_idcs.append(associated_index)
    if y_test == y_pred[i]:
        success += 1

#############
# EVALUATION
#############
# read from file
raw_data = pd.read_csv("./finance_data/raw_data.csv")  # Date, SPX Close, SPX Volume, VIX Price, VIX Volume, Gold Price, Gold Volume

# create DatetimeIndex
raw_data["Date"] = pd.to_datetime(raw_data["Date"], format="%Y-%m-%d")
raw_data = raw_data.set_index("Date")

# Calculate the Strategy Price
plot_data = raw_data.diff().dropna().loc[pred_dates_idcs]  # each row contains difference to the previous row
plot_data["Strategy"] = plot_data["SPX Close"] * y_pred
plot_data = plot_data[["SPX Close", "Strategy"]]

# Plot the result
plot_data.cumsum().plot(figsize=(10, 5))  # plot the cummulated sum
plt.show()
