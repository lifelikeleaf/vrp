# Author: Xu Ye <kan.ye@tum.de>

# dir names for benchmark data sets
CVRPLIB = 'CVRPLIB'
SOLOMON = 'Vrp-Set-Solomon' # n=100; 56 instances
HG = 'Vrp-Set-HG' # n=[200, 400, 600, 800, 1000]; 60 instances each

# dir name for test output
TEST_DIR = 'Test'

# metrics and extra data returned by the solver
METRIC_COST = 'cost'
EXTRA_ROUTE_STARTS = 'route_starts'

# ortools dimension names
DIMENSION_TIME = 'time'
DIMENSION_CAPACITY = 'capacity'

# keys for data output
KEY_COST = 'cost'
KEY_NUM_ROUTES = 'num_routes'
KEY_INSTANCE_NAME = 'instance_name'
KEY_ITERATION = 'iteration'
KEY_NUM_SUBPROBS = 'num_subprobs'
KEY_EXPERIMENT_NAME = 'experiment_name'
KEY_ROUTES = 'routes'

# ortools solver status
# https://developers.google.com/optimization/routing/routing_options#search_status
SOLVER_STATUS = {
    0: 'ROUTING_NOT_SOLVED: Problem not solved yet.',
    1: 'ROUTING_SUCCESS: Problem solved successfully.',
    2: 'ROUTING_FAIL: No solution found to the problem.',
    3: 'ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution.',
    4: 'ROUTING_INVALID: Model, model parameters, or flags are not valid.',
}

'''Solomon 100-node benchmarks'''
## geographically clustered
## narrow TWs
C1 = ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109']
## wide TWs
C2 = ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208']
## randomly distributed
R1 = ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108', 'R109', 'R110', 'R111', 'R112']
R2 = ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R211']
## random clustered
RC1 = ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108']
RC2 = ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']

'''HG 1k-node benchmarks'''
C1_10 = ['C1_10_1', 'C1_10_2', 'C1_10_3', 'C1_10_4', 'C1_10_5', 'C1_10_6', 'C1_10_7', 'C1_10_8', 'C1_10_9', 'C1_10_10']
C2_10 = ['C2_10_1', 'C2_10_2', 'C2_10_3', 'C2_10_4', 'C2_10_5', 'C2_10_6', 'C2_10_7', 'C2_10_8', 'C2_10_9', 'C2_10_10']
R1_10 = ['R1_10_1', 'R1_10_2', 'R1_10_3', 'R1_10_4', 'R1_10_5', 'R1_10_6', 'R1_10_7', 'R1_10_8', 'R1_10_9', 'R1_10_10']
R2_10 = ['R2_10_1', 'R2_10_2', 'R2_10_3', 'R2_10_4', 'R2_10_5', 'R2_10_6', 'R2_10_7', 'R2_10_8', 'R2_10_9', 'R2_10_10']
RC1_10 = ['RC1_10_1', 'RC1_10_2', 'RC1_10_3', 'RC1_10_4', 'RC1_10_5', 'RC1_10_6', 'RC1_10_7', 'RC1_10_8', 'RC1_10_9', 'RC1_10_10']
RC2_10 = ['RC2_10_1', 'RC2_10_2', 'RC2_10_3', 'RC2_10_4', 'RC2_10_5', 'RC2_10_6', 'RC2_10_7', 'RC2_10_8', 'RC2_10_9', 'RC2_10_10']
FOCUS_GROUP_C1 = ['C1_10_1', 'C1_10_4', 'C1_10_8']
FOCUS_GROUP_C2 = ['C2_10_1', 'C2_10_4', 'C2_10_8']
FOCUS_GROUP_R1 = ['R1_10_1', 'R1_10_4', 'R1_10_8']
FOCUS_GROUP_R2 = ['R2_10_1', 'R2_10_4', 'R2_10_8']
FOCUS_GROUP_RC1 = ['RC1_10_1', 'RC1_10_4', 'RC1_10_8']
FOCUS_GROUP_RC2 = ['RC2_10_1', 'RC2_10_4', 'RC2_10_8']

