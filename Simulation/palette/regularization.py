# encoding: utf8
import argparse
import multiprocessing as mp
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import tqdm
import const

def parse_args():
    parser = argparse.ArgumentParser(description='Regularization')
    parser.add_argument('--anonymity_dir', default='cluster_result/', type=str)
    parser.add_argument('--anonymity_suffix', default='_1000_30_1', type=str)
    parser.add_argument('--alpha_upload', default=0.16, type=float)
    parser.add_argument('--alpha_download', default=0.16, type=float)
    parser.add_argument('--U_upload', default=45, type=float)
    parser.add_argument('--U_download', default=45, type=float)
    parser.add_argument('--B', default=20, type=float)
    parser.add_argument('--n_jobs', default=10, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--is_dump', default=False, type=bool)

    args = parser.parse_args()

    return args


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
eps = 1e-6

args = parse_args()

out_dir = const.OUTPUT_PATH
file_dir = const.TRACES_PATH
anonymity_dir = args.anonymity_dir
anonymity_suffix = args.anonymity_suffix
alpha_upload = args.alpha_upload
alpha_download = args.alpha_download
U_upload = args.U_upload
U_download = args.U_download
B = args.B
seed = args.seed
n_jobs = args.n_jobs
is_dump = args.is_dump

random.seed(seed)


def sample(trace_prob, threshold, tam_len):
    slot_idx = list(range(0, tam_len))
    random.shuffle(slot_idx)
    sampled_slots = []
    cum_prob = 0
    for i in range(tam_len):
        if cum_prob >= threshold:
            return sampled_slots
        cum_prob = cum_prob + trace_prob[slot_idx[i]]
        sampled_slots.append(slot_idx[i])
    return sampled_slots


def dump(trace, output_path, file):
    with open(os.path.join(output_path, file), 'w') as fo:
        for i in range(len(trace)):
            fo.write("{}".format(trace[i][0]) + '\t' + "{}".format(int(trace[i][1])) + '\n')


def generate_defense(para):
    cur_shrunk_super_matrix, now_dir, now_file_path, sampled_slots_upload, sampled_slots_download, is_dump = para
    now_file_name = now_file_path.split('/')[-1]

    with open(now_file_path, 'r') as f:
        tcp_dump = f.readlines()

    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype(
        "float")
    times = np.array(seq.iloc[:, 0]) - np.array(seq.iloc[0, 0])
    length_seq = np.array(seq.iloc[:, 1]).astype("int")

    packets = np.empty((0, 2), dtype=np.float32)

    now_timestamp = const.TIME_SLOT
    # current slot index
    now_slot_idx = 0

    # the number of buffered packets
    cum_upload = 0.
    cum_download = 0.

    # total real packets
    total_pkt = 0.
    # total upload packets
    total_pkt_upload = 0.
    # total download packets
    total_pkt_download = 0.

    # the last time slot which has real packets
    final_slot = 0.

    # the current packet sequences idx
    now_sequence_idx = 0

    # threshold for delayed packets
    u_upload = random.randint(0, U_upload - 1)
    u_download = random.randint(0, U_download - 1)

    # a flag which indicates if there do not have any buffered packets
    flag_end_upload = False
    flag_end_download = False

    # a flag which indicates if the first upload/download packet has been sent
    flag_first_upload = False
    flag_first_download = False

    while now_timestamp <= const.CUTOFF_TIME:

        # the sending budget in next u slots
        budget_upload = np.sum(cur_shrunk_super_matrix[0, now_slot_idx: now_slot_idx + u_upload])
        budget_download = np.sum(cur_shrunk_super_matrix[1, now_slot_idx: now_slot_idx + u_download])

        sm_upload = cur_shrunk_super_matrix[0, now_slot_idx]
        sm_download = cur_shrunk_super_matrix[1, now_slot_idx]

        target_upload = sm_upload
        target_download = sm_download

        # super-matrix sampling
        if now_slot_idx not in sampled_slots_upload:
            target_upload = 0.
        if now_slot_idx not in sampled_slots_download:
            target_download = 0.

        # get the number of real packets sent at current slot
        while now_sequence_idx < len(times) and times[now_sequence_idx] < now_timestamp:
            total_pkt += 1
            if length_seq[now_sequence_idx] > 0:
                cum_upload += 1
                total_pkt_upload += 1
            elif length_seq[now_sequence_idx] < 0:
                cum_download += 1
                total_pkt_download += 1
            now_sequence_idx += 1

        if cum_upload != 0.:
            final_slot = now_slot_idx
            flag_end_upload = False
        if cum_download != 0.:
            final_slot = now_slot_idx
            flag_end_download = False

        # Regulate upload packets
        send_upload = target_upload
        # if no buffered real packets, we will check if the current time slot idx is multiple of B
        if cum_upload == 0:
            # send packets until real packets arrive
            if flag_end_upload:
                send_upload = 0
            elif now_slot_idx != 0 and now_slot_idx % B == 0:
                flag_end_upload = True

        if total_pkt_upload >= 1 and flag_first_upload:
            # early sending
            if cum_upload != 0 and cum_upload >= budget_upload:
                send_upload = max(min(10, cum_upload), sm_upload)
        else:
            if cum_upload != 0:
                send_upload = max(cum_upload, send_upload)
            else:
                send_upload = 0

        # Regulate download packets
        send_download = target_download
        if cum_download == 0:
            if flag_end_download:
                send_download = 0
            elif now_slot_idx != 0 and now_slot_idx % B == 0:
                flag_end_download = True

        if total_pkt_download >= 1 and flag_first_download:
            if cum_download != 0 and cum_download >= budget_download:
                send_download = max(min(30, cum_download), sm_download)
        else:
            if cum_download != 0:
                send_download = max(cum_download, send_download)
            else:
                send_download = 0

        if total_pkt_upload >= 1:
            flag_first_upload = True
        if total_pkt_download >= 1:
            flag_first_download = True

        cum_upload = max(0., cum_upload - send_upload)
        cum_download = max(0., cum_download - send_download)

        # sample the timestamps for packets sent in current slot
        upload_timestamps = np.clip(now_timestamp + np.random.rayleigh(0.03, int(send_upload)), a_min=now_timestamp,
                                    a_max=now_timestamp + const.TAM_LENGTH - eps)
        download_timestamps = np.clip(now_timestamp + np.random.rayleigh(0.03, int(send_download)),
                                      a_min=now_timestamp, a_max=now_timestamp + const.TAM_LENGTH - eps)

        for j in range(len(upload_timestamps)):
            packets = np.append(packets, np.array([[upload_timestamps[j], 1]]), axis=0)
        for j in range(len(download_timestamps)):
            packets = np.append(packets, np.array([[download_timestamps[j], -1]]), axis=0)

        now_slot_idx += 1
        now_timestamp = const.TIME_SLOT * (now_slot_idx + 1)

    if cum_upload > 0.:
        final_slot = now_slot_idx
        upload_timestamps = np.clip(now_timestamp + np.random.rayleigh(0.1, int(cum_upload)), a_min=now_timestamp,
                                    a_max=now_timestamp + 5.)
        for p in range(len(upload_timestamps)):
            packets = np.append(packets, np.array([[upload_timestamps[p], 1]]), axis=0)

    if cum_download > 0.:
        final_slot = now_slot_idx
        download_timestamps = np.clip(np.random.rayleigh(0.1, int(cum_download)), a_min=now_timestamp,
                                      a_max=now_timestamp + 5.)
        for p in range(len(download_timestamps)):
            packets = np.append(packets, np.array([[download_timestamps[p], -1]]), axis=0)

    packets_idx = np.argsort(packets[:, 0])
    packets = packets[packets_idx]
    packets[:, 0] = packets[:, 0] - packets[0, 0]

    if is_dump:
        dump(packets, now_dir, now_file_name)

    final_end = max((final_slot + 1) * const.TIME_SLOT, np.minimum(const.CUTOFF_TIME, times[-1]))

    return len(packets), total_pkt, final_end, np.minimum(const.CUTOFF_TIME, times[-1])


def parallel(para_list, n_jobs=15):
    pool = mp.Pool(n_jobs)
    overheads = tqdm.tqdm(pool.imap(generate_defense, para_list), total=len(para_list))
    pool.close()

    return overheads


if __name__ == '__main__':

    website_to_set = np.load(anonymity_dir + 'website_to_set' + anonymity_suffix + '.npy', allow_pickle=True).item()
    shrunk_super_matrices = np.load(anonymity_dir + 'shrunk_super_matrices' + anonymity_suffix + '.npy', allow_pickle=True)
    PMF_upload = pickle.load(open(anonymity_dir + 'PMF_upload' + anonymity_suffix + '.pkl', 'rb'))
    PMF_download = pickle.load(open(anonymity_dir + 'PMF_download' + anonymity_suffix + '.pkl', 'rb'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process_data = []

    for i in range(const.MONITORED_SITE_NUM):
        for j in range(const.MONITORED_INST_NUM):
            file_name = '{}-{}'.format(i, j)
            website_idx = i

            anonymity_sets = website_to_set[website_idx]
            anonymity_set_idx = anonymity_sets[j % len(anonymity_sets)]

            shrunk_super_matrix = np.ceil(shrunk_super_matrices[anonymity_set_idx])
            np.set_printoptions(threshold=np.inf)
            sampled_slots_upload = sample(PMF_upload[anonymity_set_idx], alpha_upload, tam_len=const.TAM_LENGTH)
            sampled_slots_download = sample(PMF_download[anonymity_set_idx], alpha_download, tam_len=const.TAM_LENGTH)
            print(len(sampled_slots_upload), len(sampled_slots_download))
            process_data.append((shrunk_super_matrix[0, 0, :, :], out_dir, file_dir + file_name, sampled_slots_upload,
                                 sampled_slots_download, is_dump))

    if const.OPEN_WORLD == 1:
        for i in range(const.UNMONITORED_SITE_NUM):
            file_name = str(i)
            website_idx = const.MONITORED_SITE_NUM

            anonymity_sets = website_to_set[random.randint(0, const.MONITORED_SITE_NUM - 1)]
            anonymity_set_idx = anonymity_sets[random.randint(0, len(anonymity_sets) - 1)]
            shrunk_super_matrix = np.ceil(shrunk_super_matrices[anonymity_set_idx])

            sampled_slots_upload = sample(PMF_upload[anonymity_set_idx], alpha_upload, tam_len=const.TAM_LENGTH)
            sampled_slots_download = sample(PMF_download[anonymity_set_idx], alpha_download,
                                            tam_len=const.TAM_LENGTH)

            process_data.append((shrunk_super_matrix[0, 0, :, :], out_dir, file_dir + file_name, sampled_slots_upload,
                                 sampled_slots_download, is_dump))

    overheads = parallel(process_data, n_jobs=n_jobs)
    final_band, pre_band, final_time, pre_time = zip(*overheads)

    print(np.sum(np.array(final_band)) / np.sum(np.array(pre_band)) - 1,
          np.sum(np.array(final_time)) / np.sum(np.array(pre_time)) - 1)
