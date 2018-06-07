#!/usr/bin/env python

import scipy
import math
from sklearn import mixture
import numpy as np
import pandas as pd
import util.util as util
import copy
from util.VisualizeDataset import VisualizeDataset

# dataset_path = './intermediate_datafiles/'
# dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
# outlier_columns = ['acc_phone_x', 'light_phone_lux']
# DataViz = VisualizeDataset()


def main():
    start = input("choose method: [1],[2],[3],[4]")
    if start == 1:
        param = input("Chauvenet\ninput parameters: c")
        for col in outlier_columns:
            ch = chauvenet(dataset, col, param)
            DataViz.plot_binary_outliers(dataset, col, col + '_outlier')

    elif start == 2:
        # param = input("Mixture model\n input parameters: components, iter")
        components, iter = raw_input("Mixture model\n input parameters: components, iter").split(',')
        print(components)
        print(iter)
        for col in outlier_columns:
            mm = mixture_model(dataset, col, components, iter)
            DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])

    elif start == 3:
        d_min, f_min = raw_input("Simple distance-based\n input parameters: d_min, f_min").split()
        for col in outlier_columns:
            sdb = simple_distance_based(dataset, col, 'euclidean', d_min, f_min)
            DataViz.plot_binary_outliers(dataset, col, 'simple_dist_outlier')

    elif start == 4:
        param = input("Local outlier factor\n input parameters: k")
        for col in outlier_columns:
            lof = local_outlier_factor(dataset, col, 'euclidean', k)
            DataViz.plot_dataset(dataset, [col, 'lof'], ['exact','exact'], ['line', 'points'])
        
    else :
        print("no method selected")

def chauvenet(self, data_table, col, c):
    # Taken partly from: https://www.astro.rug.nl/software/kapteyn/

    # Computer the mean and standard deviation.
    mean = data_table[col].mean()
    std = data_table[col].std()
    N = len(data_table.index)
    constant = c
    criterion = 1.0/(constant*N)

    # Consider the deviation for the data points.
    deviation = abs(data_table[col] - mean)/std

    # Express the upper and lower bounds.
    low = -deviation/math.sqrt(2)
    high = deviation/math.sqrt(2)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(data_table.index)):
        # Determine the probability of observing the point
        prob.append(1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i])))
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    data_table[col + '_outlier'] = mask
    return data_table

def mixture_model(self, data_table, col, n_components, n_iter):
    # Fit a mixture model to our data.
    data = data_table[data_table[col].notnull()][col]
    g = mixture.GMM(n_components, n_iter)

    g.fit(data.reshape(-1,1))

    # Predict the probabilities
    probs = g.score(data.reshape(-1,1))

    # Create the right data frame and concatenate the two.
    data_probs = pd.DataFrame(np.power(10, probs), index=data.index, columns=[col+'_mixture'])
    data_table = pd.concat([data_table, data_probs], axis=1)

    return data_table

def simple_distance_based(self, data_table, cols, d_function, dmin, fmin):
    # Normalize the dataset first.
    new_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)
    # Create the distance table first between all instances:
    distances = self.distance_table(new_data_table, cols, d_function)

    mask = []
    # Pass the rows in our table.
    for i in range(0, len(new_data_table.index)):
        # Check what faction of neighbors are beyond dmin.
        frac = (float(sum([1 for col_val in distances.ix[i,:].tolist() if col_val > dmin]))/len(new_data_table.index))
        # Mark as an outlier if beyond the minimum frequency.
        mask.append(frac > fmin)
    data_mask = pd.DataFrame(mask, index=new_data_table.index, columns=['simple_dist_outlier'])
    data_table = pd.concat([data_table, data_mask], axis=1)
    return data_table

def local_outlier_factor(self, data_table, cols, d_function, k):
    # Inspired on https://github.com/damjankuznar/pylof/blob/master/lof.py
    # but tailored towards the distance metrics and data structures used here.

    # Normalize the dataset first.
    new_data_table = util.normalize_dataset(data_table.dropna(axis=0, subset=cols), cols)
    # Create the distance table first between all instances:
    self.distances = self.distance_table(new_data_table, cols, d_function)

    outlier_factor = []
    # Compute the outlier score per row.
    for i in range(0, len(new_data_table.index)):
        print i
        outlier_factor.append(self.local_outlier_factor_instance(i, k))
    data_outlier_probs = pd.DataFrame(outlier_factor, index=new_data_table.index, columns=['lof'])
    data_table = pd.concat([data_table, data_outlier_probs], axis=1)
    return data_table

if __name__ == '__main__':
    main()
