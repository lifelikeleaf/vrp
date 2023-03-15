# Author: Xu Ye <kan.ye@tum.de>

import os
import traceback

from vrp.decomp.decomposers import (
    KMedoidsDecomposer,
    HierarchicalDecomposer,
)
from vrp.decomp.solvers import HgsSolverWrapper, GortoolsSolverWrapper
import vrp.decomp.helpers as helpers
from vrp.decomp.logger import logger
from vrp.decomp.constants import *
import vrp.decomp.distance_matrices as DM
from experiments import ExperimentRunner

logger = logger.getChild(__name__)


if __name__ == "__main__":

    def hierarchical():
        experiments = []

        '''k-medoids'''
        '''euclidean'''
        experiments.append(KMedoidsDecomposer(DM.euclidean_vectorized, name='euclidean'))

        '''hierarchical'''
        '''euclidean'''
        experiments.append(HierarchicalDecomposer(DM.euclidean_vectorized, name='hierarchical'))

        '''lambda = 100%'''
        # dist_matrix_func = DM.v2_2_vectorized
        # ext = dist_matrix_func.__name__.removesuffix('_vectorized')
        # experiments.append(HierarchicalDecomposer(dist_matrix_func, name=f'OL_{ext}', use_overlap=True, normalize=True))
        # experiments.append(HierarchicalDecomposer(dist_matrix_func, name=f'Gap_{ext}', use_gap=True, normalize=True))
        # experiments.append(HierarchicalDecomposer(dist_matrix_func, name=f'Both_{ext}', use_overlap=True, use_gap=True, normalize=True))

        return experiments


    '''parameters for experiments'''

    '''Basis'''
    experiments_only = False # True if only run experiments, don't run Basis

    allow_sleep = True
    num_clusters_range = (4, 4) # inclusive
    repeat_n_times = 1
    time_limit = 10
    output_dir_name = 'E27_hierarchical'
    experiments = hierarchical
    benchmark_dir_name = HG
    input = {
        # HG all
        'C1': C1_10,
        'C2': C2_10,
        'R1': R1_10,
        'R2': R2_10,
        'RC1': RC1_10,
        'RC2': RC2_10,
    }

    '''parameters for experiments'''


    '''only considers driving time in OF'''
    solver = HgsSolverWrapper(time_limit)
    # solver = GortoolsSolverWrapper(time_limit, min_total=False)

    '''considers total time (incl. wait time) in OF'''
    # solver = GortoolsSolverWrapper(time_limit)

    for name, benchmark_class in input.items():
        helpers.make_dirs(output_dir_name)
        file_name = f'{output_dir_name}_' + experiments.__name__ + f'_{name}'
        file_name = os.path.join(output_dir_name, file_name)

        # e.g. [(['C101', 'C102', 'C103'], SOLOMON)]
        benchmarks = [(benchmark_class, benchmark_dir_name)]

        runner = ExperimentRunner(solver, benchmarks, num_clusters_range, repeat_n_times, file_name, allow_sleep=allow_sleep)
        runner.add_experiements(experiments())

        try:
            runner.run(experiments_only)
        except Exception as err:
            tb_msg = traceback.format_exc()
            logger.error(tb_msg)
            raise

