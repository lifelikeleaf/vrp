# dir names for benchmark data sets
CVRPLIB = 'CVRPLIB'
SOLOMON = 'Vrp-Set-Solomon' # n=100; 56 instances
HG = 'Vrp-Set-HG' # n=[200, 400, 600, 800, 1000]; 60 instances each

# metrics returned by HGS solver
METRIC_COST = 'cost'
METRIC_DISTANCE = 'distance'
METRIC_WAIT_TIME = 'wait_time'

# keys for data output
KEY_COST = 'cost'
KEY_COST_WAIT = 'cost_wait' # cost + wait time (post-routing)
KEY_NUM_ROUTES = 'num_routes'
KEY_INSTANCE_NAME = 'instance_name'
KEY_ITERATION = 'iteration'
KEY_NUM_SUBPROBS = 'num_subprobs'
KEY_EXPERIMENT_NAME = 'experiment_name'
KEY_ROUTES = 'routes'
