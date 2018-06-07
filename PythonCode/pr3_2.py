#!/usr/bin/env python

import scipy
import math
from sklearn import mixture
import numpy as np
import pandas as pd
import util.util as util
import copy
from util.VisualizeDataset import VisualizeDataset
from OutlierDetection import DistributionBasedOutlierDetection
from OutlierDetection import DistanceBasedOutlierDetection


def main():
    dataset_path = './intermediate_datafiles/'
    dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
    outlier_columns = ['acc_phone_x', 'light_phone_lux']
    DataViz = VisualizeDataset()

    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    dataset.index = dataset.index.to_datetime()
    start = input("choose method: [1],[2],[3],[4]")
    if start == 1:
        param = input("Chauvenet\ninput parameters: c")
        for col in outlier_columns:
            dataset = OutlierDistr.chauvenet(dataset, col, param)
            DataViz.plot_binary_outliers(dataset, col, col + '_outlier')

    elif start == 2:
        # param = input("Mixture model\n input parameters: components, iter")
        components, iter = raw_input("Mixture model\n input parameters: components, iter").split(',')
        components = int(components)
        iter = int(iter)
        for col in outlier_columns:
            dataset = OutlierDistr.mixture_model(dataset, col, components, iter)
            DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])

    elif start == 3:
        d_min, f_min = raw_input("Simple distance-based\n input parameters: d_min, f_min").split()
        d_min = float(d_min)
        f_min = float(f_min)
        for col in outlier_columns:
            dataset = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', d_min, f_min)
            DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')

    elif start == 4:
        param = input("Local outlier factor\n input parameters: k")
        for col in outlier_columns:
            dataset = OutlierDist.local_outlier_factor(dataset, col, 'euclidean', k)
            DataViz.plot_dataset(dataset, [col, 'lof'], ['exact','exact'], ['line', 'points'])

    else :
        print("no method selected")

if __name__ == '__main__':
    main()
