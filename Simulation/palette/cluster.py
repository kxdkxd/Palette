import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
import const
from utils.my_utils import get_PMF


labels = [0] * const.MONITORED_SITE_NUM


# get traces for each website
def getDict(dataset, website_indices):
    website_dict = {}
    for i in range(len(dataset)):
        if website_indices[i] not in website_dict.keys():
            website_dict[website_indices[i]] = dataset[i: i + 1]
        else:
            website_dict[website_indices[i]] = np.append(website_dict[website_indices[i]], dataset[i: i + 1], axis=0)

    return website_dict


# load TAM dataset
def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    print(train_X.shape, train_y.shape)

    return train_X, train_y


def get_matrix(dataset_path):
    train_X_ori, train_y_ori = load_data(dataset_path)

    website_dict = getDict(train_X_ori, train_y_ori)

    super_matrices_websites = np.zeros((const.MONITORED_SITE_NUM, 2, const.TAM_LENGTH))

    # build the super-matrix for each website using the corresponding traces
    for website in sorted(website_dict.keys()):
        super_matrix_website = np.max(website_dict[website], axis=0)
        super_matrices_websites[website] = super_matrix_website

    return super_matrices_websites


def update_matrix(matrix_1, matrix_2):
    updated_matrix = np.maximum(matrix_1, matrix_2)

    return updated_matrix


def website_clustering(partition, super_matrices, k=5):
    if len(partition) == 0:
        return [], []

    partition_set = []
    centers = np.empty((0, 2, const.TAM_LENGTH), float)
    partition = np.sort(partition)
    tar_label = np.random.choice(partition, 1)[0]
    visited = [0] * const.MONITORED_SITE_NUM
    visited[tar_label] = 1
    partition_set.append([tar_label])
    centers = np.append(centers, super_matrices[tar_label:tar_label + 1], axis=0)
    node = 0
    cnt = 1

    while cnt < len(partition):

        if len(partition_set[node]) == k:
            if len(partition) - cnt < k:
                break
            node += 1
            max_dis = -1
            max_idx = -1
            for i in partition:
                if visited[i] == 1:
                    continue
                for ct in centers:
                    dis = np.linalg.norm(super_matrices[i] - ct)
                    if dis > max_dis:
                        max_dis = dis
                        max_idx = i

            if max_idx == -1:
                break
            else:
                partition_set.append([max_idx])
                centers = np.append(centers, super_matrices[max_idx][np.newaxis, :], axis=0)
                visited[max_idx] = 1
                cnt += 1

        min_dis = 1e9
        min_idx = -1

        for i in partition:
            if visited[i] == 1:
                continue
            if min_dis > np.linalg.norm(super_matrices[i] - centers[node]):
                min_dis = np.linalg.norm(super_matrices[i] - centers[node])
                min_idx = i

        if min_idx == -1:
            break
        else:
            visited[min_idx] = 1
            partition_set[node].append(min_idx)
            centers[node] = update_matrix(centers[node], super_matrices[min_idx])
            cnt += 1

    for i in partition:
        if visited[i] == 0:
            min_dis = 1e9
            min_idx = -1
            for idx, ct in enumerate(centers):
                dis = np.linalg.norm(super_matrices[i] - ct)
                if dis < min_dis:
                    min_dis = dis
                    min_idx = idx

            partition_set[min_idx].append(i)
            centers[min_idx] = update_matrix(centers[min_idx], super_matrices[i])

    return partition_set, centers


def drawLine(x, y, colors, label):
    plt.plot(x, y, color=colors, label=label)


def draw_plot(super_matrices, anonymity_set, tam_len):
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'pink', 'orange', 'purple', 'brown', 'gray']
    for i in range(len(anonymity_set)):
        drawLine(range(0, tam_len), super_matrices[anonymity_set[i]][0, :tam_len], colors[i % 10],
                 label=anonymity_set[i])
        drawLine(range(0, tam_len), -super_matrices[anonymity_set[i]][1, :tam_len], colors[i % 10],
                 label=anonymity_set[i])
    plt.show()


if __name__ == '__main__':

    # list of generated anonymity sets
    total_sets = []
    # list of generated super-matrices, the super-matrix of total_sets[i] is super_matrices[i]
    super_matrices = []
    # mapping of website to anonymity set index
    website_to_set = {}

    # First step: build the super-matrix for each website using the corresponding traces
    super_matrices_websites = get_matrix(const.DATASET_PATH + const.TRAIN_DATA_FILE)

    website_indices = np.array([i for i in range(const.MONITORED_SITE_NUM)])
    np.random.shuffle(website_indices)

    partition_1 = website_indices
    partition_2 = []
    for i in range(const.ROUND):
        anonymity_sets_fir, super_matrices_fir = website_clustering(partition_1, super_matrices_websites,
                                                                    const.SET_SIZE)
        anonymity_sets_sec, super_matrices_sec = website_clustering(partition_2, super_matrices_websites,
                                                                    const.SET_SIZE)

        partition_1 = []
        partition_2 = []

        for anonymity_set, super_matrix in zip(anonymity_sets_fir, super_matrices_fir):
            super_matrix = np.where(super_matrix == 0, 1, super_matrix)
            for website in anonymity_set:
                if len(partition_1) < len(partition_2):
                    partition_1.append(website)
                else:
                    partition_2.append(website)
            if anonymity_set in total_sets:
                continue
            total_sets.append(anonymity_set)
            super_matrices.append(super_matrix)

        for anonymity_set, super_matrix in zip(anonymity_sets_sec, super_matrices_sec):
            super_matrix = np.where(super_matrix == 0, 1, super_matrix)
            for website in anonymity_set:
                if len(partition_1) < len(partition_2):
                    partition_1.append(website)
                else:
                    partition_2.append(website)
            if anonymity_set in total_sets:
                continue
            total_sets.append(anonymity_set)
            super_matrices.append(super_matrix)

        partition_1 = np.array(partition_1)
        partition_2 = np.array(partition_2)

    print(total_sets)

    for idx, anonymity_set in enumerate(total_sets):
        for website in anonymity_set:
            if website in website_to_set.keys():
                website_to_set[website].append(idx)
            else:
                website_to_set[website] = [idx]

    print(website_to_set)

    # you can visualize the super-matrix of websites in each anonymity set
    for per_set in total_sets:
        draw_plot(super_matrices_websites, per_set, const.TAM_LENGTH)

    # PMF estimation
    PMF_upload, PMF_download = get_PMF(const.DATASET_PATH + const.TRAIN_DATA_FILE, website_to_set, len(total_sets),
                                       const.TAM_LENGTH)

    pickle.dump(PMF_upload, open('cluster_result/PMF_upload_{}_{}_{}'.format(str(const.TAM_LENGTH), str(const.SET_SIZE),
                                                                             str(const.ROUND)) + '.pkl', 'wb'))

    PMF_upload = list(map(lambda arr: arr.tolist(), PMF_upload))
    with open('json/PMF_upload_{}_{}_{}'.format(str(const.TAM_LENGTH), str(const.SET_SIZE),
                                                                             str(const.ROUND)) + '.json', 'w') as f:
        json.dump(PMF_upload, f)

    pickle.dump(PMF_download,
                open('cluster_result/PMF_download_{}_{}_{}'.format(str(const.TAM_LENGTH), str(const.SET_SIZE),
                                                                   str(const.ROUND)) + '.pkl', 'wb'))
    PMF_download = list(map(lambda arr: arr.tolist(), PMF_download))
    with open('json/PMF_download_{}_{}_{}'.format(str(const.TAM_LENGTH), str(const.SET_SIZE),
                                                          str(const.ROUND)) + '.json', 'w') as f:
        json.dump(PMF_download, f)

    pickle.dump(total_sets,
                open('cluster_result/total_set_{}_{}_{}'.format(str(const.TAM_LENGTH), str(const.SET_SIZE),
                                                                str(const.ROUND)) + '.pkl', 'wb'))
    np.save('cluster_result/super_matrices_{}_{}_{}'.format(str(const.TAM_LENGTH), str(const.SET_SIZE),
                                                            str(const.ROUND)) + '.npy', super_matrices)
    np.save('cluster_result/website_to_set_{}_{}_{}'.format(str(const.TAM_LENGTH), str(const.SET_SIZE),
                                                            str(const.ROUND)) + '.npy', website_to_set)

    with open('website_to_set.json', 'w') as f:
        f.write(str(website_to_set))

