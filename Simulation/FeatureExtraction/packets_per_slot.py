from Simulation.palette.const import *

def fun(times, sizes):
    feature = [[0 for _ in range(TAM_LENGTH)], [0 for _ in range(TAM_LENGTH)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= CUTOFF_TIME:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (TAM_LENGTH - 1) / CUTOFF_TIME)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= CUTOFF_TIME:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (TAM_LENGTH - 1) / CUTOFF_TIME)
                feature[1][idx] += 1

    return feature
