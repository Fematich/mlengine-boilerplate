#!/usr/bin/env python2.7

import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import argparse

np.set_printoptions(linewidth=100)


def gen_regression_dataset(n_samples, n_features):
    """Generate a reproducible regression dataset

    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
    Returns:
        (tuple): A two-element tuple containing:

            X (numpy.ndarray): generated samples
            y (numpy.ndarray): labels
    """
    # We'll say 50% of the features are informative
    n_informative = int(n_features * .5)

    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=1,
        noise=0.1,
        bias=np.pi,
        coef=True,
        random_state=31337  # For teh reproducibilities (and lulz)
    )
    # Don't need the coefficients yet
    return X, y


def write_regression_data_to_csv(X, y, filename):
    """Write a regression data set to a CSV file

    Takes X and y, and generates a single CSV file with:
        - sample index number as column 0
        - label as column 1
        - X, starting at column 2

    Args:
        X (numpy.ndarray): generated sample matrix
        y (numpy.ndarray): labels
        filename (str): filename to write out

    Returns:
        None
    """
    # flatten y for the insertion into X
    y = y.reshape(
        -1, )
    # insert y into X as a column at index 0
    X = np.insert(X, 0, y, axis=1)

    # This is pretty lazy, but ordinarily I'd just use np.tofile, but pandas conveniently
    # writes an index column, which I sort of want.
    pd.DataFrame(X).to_csv(filename, header=False)


def tf_recordtest(args):
    X, y = gen_regression_dataset(
        n_samples=args.n_samples, n_features=args.n_features)
    write_regression_data_to_csv(X, y, args.filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="write some random tfrecords")
    parser.add_argument(
        '-s',
        '--samples',
        metavar='N',
        type=int,
        default=500,
        dest='n_samples',
        help='Number of samples/observations')
    parser.add_argument(
        '-f',
        '--features',
        metavar='M',
        type=int,
        default=10,
        dest='n_features',
        help='Number of features/targets')
    parser.add_argument(
        '-F',
        '--filename',
        metavar="FILE",
        dest='filename',
        required=True,
        help='Filename for CSV output')
    args = parser.parse_args()

    tf_recordtest(args)
