import copy
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVC

from LSSTM.LS_STM import LSSTM
from LSSTM.data_makers import make_data_svm, create_stm_slice


class Strategy:
    def __init__(self, raw_data):
        """

        Parameters
        ----------
        raw_data: pd.DataFrame, having the prices of the time series
        """
        self.raw_data = raw_data

    def svm(self, data_svm, slice_width=250, kernel="linear", C=10, gamma="auto"):
        raw_data = copy.deepcopy(self.raw_data)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"], format="%Y-%m-%d")
        raw_data = raw_data.set_index("Date")

        data_svm = copy.deepcopy(data_svm)
        data_svm["Date"] = pd.to_datetime(data_svm["Date"], format="%Y-%m-%d")
        data_svm = data_svm.set_index("Date")

        rows = len(data_svm)
        slice_width = slice_width
        y_pred_svm = np.zeros(rows - 1 - slice_width)
        success = 0

        clf = SVC(kernel=kernel, C=C, gamma=gamma)

        indices = []
        for i in range(rows - 1 - slice_width):
            (
                train_data,
                train_label,
                test_data,
                test_label,
                associated_index,
            ) = make_data_svm(data_svm, start_index=i, L=slice_width)
            clf.fit(train_data, train_label)
            y_pred_svm[i] = clf.predict(test_data.reshape(1, -1))

            indices.append(associated_index)
            if test_label == y_pred_svm[i]:
                success += 1

        raw_data_diff = raw_data.diff().dropna().loc[indices]
        raw_data_diff["Strategy"] = raw_data_diff["SPX Close"] * y_pred_svm
        raw_data_diff = raw_data_diff[["SPX Close", "Strategy"]]

        results = {
            "Accuracy": success / (rows - 1 - slice_width),
            "Performance": raw_data_diff.cumsum(),
        }

        return results

    def stm(self,
            data_stm: DataFrame,
            tensor_shape: List[int],
            slice_width: int,
            C: int = 10,
            kernel: str = "linear",
            sig2=1,
            max_iter=100,
            verbose=False):

        raw_data = copy.deepcopy(self.raw_data)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"], format="%Y-%m-%d")
        raw_data = raw_data.set_index("Date")

        data_stm = copy.deepcopy(data_stm)

        rows = len(data_stm["Price"])

        tensors = rows - slice_width - tensor_shape[0] + 1

        y_pred_stm: np.ndarray = np.zeros(tensors - 1)

        stm = LSSTM(C=C, kernel=kernel, sig2=sig2, max_iter=max_iter)
        success = 0
        indices = []
        for i in range(tensors - 1):
            if verbose:
                print("\r{0}".format(float(i) / (tensors - 1) * 100))
            (
                train_data,
                train_labels,
                test_data,
                test_label,
                associated_index,
            ) = create_stm_slice(
                d=data_stm, start_index=i, slice_width=slice_width, tensor_shape=tensor_shape
            )
            stm.fit(train_data, train_labels)
            y_tmp, _ = stm.predict(test_data)
            y_pred_stm[i] = y_tmp[0]

            indices.append(associated_index)
            if test_label == y_pred_stm[i]:
                success += 1

        raw_data_diff = raw_data.diff().dropna().loc[indices]
        raw_data_diff["Strategy"] = raw_data_diff["SPX Close"] * y_pred_stm
        raw_data_diff = raw_data_diff[["SPX Close", "Strategy"]]

        # plot_data.cumsum().plot(figsize=(10, 5))

        results = {
            "Accuracy": success / (tensors - 1),
            "Ypred": y_pred_stm,
            "Performance": raw_data_diff.cumsum(),
        }

        return results
