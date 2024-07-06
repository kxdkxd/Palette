OPEN_WORLD = 0
MONITORED_SITE_NUM = 95
MONITORED_INST_NUM = 100
UNMONITORED_SITE_NUM = 40000
TAM_LENGTH = 1000
CUTOFF_TIME = 80
TIME_SLOT = CUTOFF_TIME / TAM_LENGTH

# extract_list.py
# traces path:
TRACES_PATH = ''
# output path: use the exact path to save the extracted .npy dataset or save to datasets/ by default
OUTPUT_PATH = ''

# dataset path: use the exact path or put the dataset in datasets/
DATASET_PATH = 'datasets/'
TRAIN_DATA_FILE = 'Undefence-train-packets_per_slot.npy'
TEST_DATA_FILE = 'Undefence-test-packets_per_slot.npy'

# parameter for cluster.py
'''
    You can set ROUND > 1 to generate diverse anonymity sets.
    Specifically, each website can be assigned to ROUND anonymity sets, to ensure different visits to the same
    website can exhibit diverse traffic patterns.  
    In this work, we simply set ROUND = 1, i.e., each website is assigned to a single anonymity set.
'''
ROUND = 1
# anonymity set size
SET_SIZE = 30


