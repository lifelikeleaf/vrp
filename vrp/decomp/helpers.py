import os
import argparse
import math
import numpy as np
import time
from typing import Callable
import json

import pandas as pd
import openpyxl as xl

from .decomposition import Node, VRPInstance
from .logger import logger


def get_args_parser(script_name):
    parser = argparse.ArgumentParser(
        description="Example usages: "
            f"python {script_name} -b=1 -n='C206' -k=3 -t | "
            f"python {script_name} -b=2 -n='C1_2_1' -k=3 -t"
    )

    parser.add_argument(
        '-n', '--instance_name', required=True,
        help='benchmark instance name without file extension, e.g. "C206"'
    )
    parser.add_argument(
        '-b', '--benchmark', default=1, choices=[1, 2], type=int,
        help='benchmark dataset to use: 1=Solomon (1987), '
            '2=Homberger and Gehring (1999); Default=1'
    )
    parser.add_argument('-k', '--num_clusters', required=True, type=int,
                        help='number of clusters')
    parser.add_argument('-t', '--include_time_windows', action='store_true',
                        help='use time windows for clustering or not')

    args = parser.parse_args()
    return args


def get_min_tours(demands, capacity):
    """Get the minimum number of tours (i.e. vehicles required)."""
    # total demand of all customers / vehicle capacity
    return math.ceil(sum(demands) / capacity)


def normalize_feature_vectors(fv):
    """Normalize feature vectors using z-score."""
    fv = np.array(fv)
    # axis=0 -> row axis, runs down the rows, i.e. calculate the mean for each column/feature
    mean = np.mean(fv, axis=0)
    # ddof=1 -> degrees of freedom = N-1, i.e. sample std
    # ddof = 'delta degrees of freedom'
    # set ddof=0 for population std
    std = np.std(fv, axis=0, ddof=1)
    norm = (fv - mean) / std
    return norm.tolist()


def normalize_matrix(m):
    """Normalize a matrix using z-score."""
    m = np.array(m)
    mean = np.mean(m)
    # ddof=1 -> degrees of freedom = N-1, i.e. sample std
    # ddof = 'delta degrees of freedom'
    # set ddof=0 for population std
    std = np.std(m, ddof=1)
    norm = (m - mean) / std
    return norm.tolist()


def convert_cvrplib_to_vrp_instance(instance) -> VRPInstance:
    """Converts a `cvrplib.Instance.VRPTW` object to a
    `decomposition.VRPInstance` object.

    Parameters
    ----------
    instance: `cvrplib.Instance.VRPTW`
        A VRPTW problem instance returned by `cvrplib.read()`.

    Returns
    -------
    inst: `decomposition.VRPInstance`
        A `VRPInstance` object representing the VRP problem instance.

    """
    node_list = []
    for customer_id in range(len(instance.coordinates)):
        params = dict(
            x_coord = instance.coordinates[customer_id][0],
            y_coord = instance.coordinates[customer_id][1],
            demand = instance.demands[customer_id],
            distances = instance.distances[customer_id],
            start_time = instance.earliest[customer_id],
            end_time = instance.latest[customer_id],
            service_time = instance.service_times[customer_id],
        )
        node = Node(**params)
        node_list.append(node)

    inst = VRPInstance(node_list, instance.capacity, extra={'name': instance.name})
    # Example: how to tag extra data fields to VRPInstance
    # inst.extra = {'num_vehicle': 20, 'distance_limit': 100}
    return inst


def get_time_window_overlap_or_gap(tw_1, tw_2):
    """Computes the amount of overlap or gap between 2 time windows.
        - Overlap: if overlap_or_gap > 0
        - Gap: if overlap_or_gap < 0

    For example:
        - let tw_1 = [0, 15], tw_2 = [5, 20], then there's an overlap of 10.
        - let tw_1 = [0, 10], tw_2 = [5, 9], then there's an overlap of 4.
        - let tw_1 = [0, 10], tw_2 = [15, 20], then there's a gap of 5.
    """
    tw_start_1, tw_end_1 = tw_1
    tw_start_2, tw_end_2 = tw_2
    overlap_or_gap = min(tw_end_1, tw_end_2) - max(tw_start_1, tw_start_2)
    return overlap_or_gap


def log_run_time(func: Callable):
    """Decorator for logging the run time of a function or method."""
    log = logger.getChild(func.__module__)

    def decorator(*args, **kwargs):
        start = time.time()
        return_val = func(*args, **kwargs)
        end = time.time()
        log.info(f'{func.__qualname__}() run time = {end-start} sec')
        return return_val

    return decorator


def df_to_excel(df: pd.DataFrame, file_name, sheet_name, overlay=True, index=False):
    file_name = file_name + '.xlsx'
    if_sheet_exists = 'overlay' if overlay else 'replace'
    start_row = 0
    header = True

    try:
        if overlay:
            # if sheet exists, append to the end of the same sheet
            wb = xl.load_workbook(file_name)
            if sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                # The maximum row index containing data (1-based)
                # `startrow` param of `pandas.DataFrame.to_excel()` is 0-based
                # So next row to append to is exactly max_row
                start_row = sheet.max_row
                # do not write out the column names again
                header = False
        # if sheet does not exist, create a new sheet in the existing excel file
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists=if_sheet_exists) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index, header=header, startrow=start_row)
    except FileNotFoundError:
        # if file does not yet exist, create it
        with pd.ExcelWriter(file_name, mode='x') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)


# alias for backward compatibility
write_to_excel = df_to_excel


def write_to_json(data, file_name):
    file_name = file_name + '.json'
    with open(file_name, 'a') as f:
        json.dump(data, f) #, indent=4)
        f.write('\n')


def read_json(file_name):
    file_name = file_name + '.json'
    data = []
    # reading a file with multiple json objects must read 1 line/1 json object
    # at a time, bc json.load() can only handle a single json object
    with open(file_name) as f:
        for line in f:
            py_obj = json.loads(line)
            data.append(py_obj)

    return data


def sleep(sec, logger_name=__name__):
    log = logger.getChild(logger_name)
    log.info(f'Sleeping for {sec} sec')
    time.sleep(sec)


def safe_divide(numerator, denominator):
    # prevent ZeroDivisionError by returning 0 if denominator is 0
    return numerator / denominator if denominator else 0


def make_dirs(dir_name):
    # make sure the parent directory exists
    dir_name = os.path.join(dir_name, 'placeholder_file_name')
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)

