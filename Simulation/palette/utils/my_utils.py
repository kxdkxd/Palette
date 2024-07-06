import random

import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)


def load_data_tensor(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    train_X = torch.from_numpy(train_X).type(torch.FloatTensor)
    train_X = train_X.view(train_X.size(0), 1, 2, -1)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    print(train_X.shape, train_y.shape)

    return train_X, train_y


def get_from_set(train_X, train_Y, st, out_dim=1000, use_mapping=True):
    sorted_st = sorted(st)
    new_train_X = torch.empty(0, 1, 2, out_dim)
    new_train_Y = torch.empty(0, dtype=torch.long)
    label_mapping = []
    for idx, label in enumerate(sorted_st):
        label_mapping.append(label)
        now_X = train_X[train_Y == label]
        length = now_X.shape[0]
        new_train_X = torch.cat((new_train_X, train_X[train_Y == label]), 0)
        if use_mapping:
            new_train_Y = torch.cat((new_train_Y, torch.tensor([idx] * length)), 0)
        else:
            new_train_Y = torch.cat((new_train_Y, torch.tensor([label] * length)), 0)

    label_mapping = torch.LongTensor(label_mapping).to(device)

    return new_train_X, new_train_Y, label_mapping


def getArray(line, row):
    trace_cum = []

    for i in range(line):
        sub_arr = []
        for j in range(row):
            sub_arr.append(0)
        trace_cum.append(sub_arr)
    return trace_cum


def load_data_npy(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    return train_X, train_y
def get_PMF(dataset_dir, data_dict, set_num, tam_len):
    feature, label = load_data_npy(dataset_dir)
    arr = [[] for i in range(set_num)]
    print(len(label))
    for i in range(len(label)):
        index = int(label[i])
        for j in range(len(data_dict[index])):
            arr[data_dict[index][j]].append(i)

    trace_cum_upload = getArray(set_num, tam_len)
    trace_cum_download = getArray(set_num, tam_len)
    for i in range(len(arr)):
        trace_sum_res_upload = [0] * tam_len
        trace_sum_res_download = [0] * tam_len
        print(len(arr[i]))
        for j in range(len(arr[i])):
            trace = feature[arr[i][j]]

            trace_upload = trace[0]
            trace_download = trace[1]
            new_trace_upload = [1 if i != 0 else 0 for i in trace_upload]

            new_trace_download = [1 if i != 0 else 0 for i in trace_download]
            trace_sum_res_upload = [x + y for x, y in zip(new_trace_upload, trace_sum_res_upload)]
            trace_sum_res_download = [x + y for x, y in zip(new_trace_download, trace_sum_res_download)]

        trace_cum_upload[i] = trace_sum_res_upload
        trace_cum_download[i] = trace_sum_res_download

    trace_cum_res_upload = getArray(set_num, tam_len)
    trace_cum_res_download = getArray(set_num, tam_len)
    for i in range(len(trace_cum_upload)):
        normalized_arr_upload = trace_cum_upload[i] / np.sum(trace_cum_upload[i])
        trace_cum_res_upload[i] = normalized_arr_upload
        normalized_arr_download = trace_cum_download[i] / np.sum(trace_cum_download[i])
        trace_cum_res_download[i] = normalized_arr_download

    return trace_cum_res_upload, trace_cum_res_download
