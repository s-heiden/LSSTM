import copy
from typing import List, Tuple, Dict

import numpy as np
from hottbox.core import Tensor
from pandas import DataFrame
from pandas._libs.tslibs import timestamps


def make_data_svm(df, start_index, L):
    """

    Parameters
    ----------
    df: pd.DataFrame, the data organized for SVM
    start_index: int
    L: int (lookback window)

    Returns
    -------
    xs_train
    y_train
    xs_test
    y_test
    """

    df_svm = copy.deepcopy(df)

    df_svm = df_svm[start_index: start_index + L]

    df_data = df_svm.drop("Label", axis=1)
    df_label = df_svm["Label"]

    xs_train = np.array(df_data[0: L - 1])
    y_train = np.array(df_label[0: L - 1])

    xs_test = np.array(df_data.iloc[L - 1])
    y_test = df_label[L - 1]

    """
    This is the time-index of how the S&P actually performed on that day. Multiply the predicted labels with the #
    differences at these indices to get the generated profits"
    """
    associated_index = df_data.iloc[-2].name

    return xs_train, y_train, xs_test, y_test, associated_index


def create_stm_slice(d: Dict[str, DataFrame], start_index, slice_width, tensor_shape) -> Tuple[List[Tensor], np.array, Tensor, np.float64, timestamps.Timestamp]:
    """

    Notes: Assumes 3rd order tensors for now, and that the order of the idcs are known

    Parameters
    ----------
    data: dict, the data organized for STM. Each entry of the dict is a dataframe.
    The labels are in the first slice

    start_index: int
    slice_width: int
    tensor_shape:list,  the size of your tensor data.

    Returns
    -------
    xs_train
    y_train
    xs_test
    y_test

    """

    # This dict has two Keys, "Volume" and "Price"
    # "Volume" contains a pd.DataFrame with a DatetimeIndex and the Cols: VIX Price   Gold Price   SPX Close   Label
    # "Price"  contains a pd.DataFrame with a DatetimeIndex and the Cols: VIX Volume  Gold Volume  SPX Volume  Label
    dict_stm = copy.deepcopy(d)

    idcs = dict_stm.keys()
    for idx in idcs:  # Price and Volume
        # schneidet die Daten aus dict_stm["Price" & "Index"] weg, die nicht zu diesem Slice gehören
        dict_stm[idx] = dict_stm[idx][start_index: start_index + slice_width]

    # initialize a 3rd dim array. It has the shape [slice_width] + tensor_shape[1:] ==
    # [250, 3, 2]
    # slice_as_3d_np_arr contains 250 (days) x 3 (features) in 2 auspraegungen (Price/Volume)
    slice_as_3d_np_arr = np.zeros([slice_width] + tensor_shape[1:])

    for i, idx in enumerate(idcs):  # Price and Volume
        slice_as_3d_np_arr[:, :, i] = np.array(dict_stm[idx].drop("Label", axis=1))

    xs_train: List[Tensor] = []
    n_tensors: int = slice_width - tensor_shape[0] + 1  # 251 tensors
    y_train: np.array = np.zeros(n_tensors - 1)
    xs_test: Tensor
    y_test: np.float64

    for i in range(n_tensors):
        upper_idx = i + tensor_shape[0]

        # if training data
        if i < n_tensors - 1:
            xs_train.append(Tensor(slice_as_3d_np_arr[i: upper_idx, :, :]))
            y_train[i] = dict_stm["Price"]["Label"][upper_idx - 1]

        # otherwise it's testing data
        else:
            xs_test = Tensor(slice_as_3d_np_arr[i: upper_idx, :, :])
            y_test = dict_stm["Price"]["Label"][upper_idx - 1]

    associated_index: timestamps.Timestamp = dict_stm["Price"].iloc[-2].name

    return xs_train, y_train, xs_test, y_test, associated_index
