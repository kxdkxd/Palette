# encoding: utf8
import argparse
import json
import pickle
import random
import numpy as np
import torch
import torch.utils.data as Data
import const
from utils.my_utils import get_from_set, load_data_tensor


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
eps = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description='Refine Super-Matrix')

    parser.add_argument('--anonymity_dir', default='cluster_result/', type=str)
    parser.add_argument('--anonymity_suffix', default='_1000_' + str(const.SET_SIZE) + '_1', type=str)
    parser.add_argument('--batch_size', default=32 * 8, type=int,
                        help="batch_size = 32 * 8 when k = 30 and # of traces for each site = 1000. When you set "
                             "different parameters, adjust the batch_size by scale")
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()

    return args


def refine(file_path, args):
    train_data_file = const.TRAIN_DATA_FILE
    test_data_file = const.TEST_DATA_FILE
    seed = args.seed
    anonymity_dir = args.anonymity_dir
    anonymity_suffix = args.anonymity_suffix
    lr = args.lr
    batch_size = args.batch_size
    Epoch = args.num_epochs

    random.seed(seed)

    total_set = pickle.load(open(anonymity_dir + 'total_set' + anonymity_suffix + '.pkl', 'rb'))

    super_matrices = np.load(anonymity_dir + 'super_matrices' + anonymity_suffix + '.npy', allow_pickle=True)
    print(super_matrices)
    train_X, train_y = load_data_tensor(file_path + train_data_file)
    test_X, test_y = load_data_tensor(file_path + test_data_file)

    shrunk_super_matrices = []

    for idx, st in enumerate(total_set):

        ct = torch.from_numpy(super_matrices[idx][np.newaxis, np.newaxis, :]).type(torch.FloatTensor).to(device)
        new_train_X, new_train_y, train_label_mapping = get_from_set(train_X, train_y, st, out_dim=const.TAM_LENGTH)
        new_test_X, new_test_y, test_label_mapping = get_from_set(test_X, test_y, st, out_dim=const.TAM_LENGTH,
                                                                  use_mapping=False)

        train_dataset = Data.TensorDataset(new_train_X, new_train_y)
        test_dataset = Data.TensorDataset(new_test_X, new_test_y)

        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size * 5, shuffle=True, num_workers=0)

        weight = torch.randn(2, const.TAM_LENGTH).to(device)
        bias = torch.zeros(2, const.TAM_LENGTH).to(device)

        adv_weight = weight.clone().detach()
        adv_bias = bias.clone().detach()

        adv_weight.requires_grad = True
        adv_bias.requires_grad = True

        th = torch.tensor([[10.], [10.]]).to(device)
        th = torch.where(ct > th, th, ct)

        optimizer = torch.optim.Adam([adv_weight, adv_bias], lr=lr)

        for epoch in range(Epoch):
            for batch_idx, (tr_x, _) in enumerate(train_loader):
                tr_x = tr_x.to(device)
                optimizer.zero_grad()

                adv_weight.requires_grad = True
                adv_bias.requires_grad = True

                trans_weight = torch.sigmoid(adv_weight)
                trans_bias = th * torch.sigmoid(adv_bias)

                pruned = torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0)

                loss = - 0.5 * torch.mean(torch.clamp(pruned * torch.sign(tr_x) - tr_x, max=0)) \
                       + torch.mean(torch.clamp(pruned * torch.sign(tr_x) - tr_x, min=10.))

                loss.backward()
                optimizer.step()

            # test_loss = 0
            # for batch_idx, (ts_x, _) in enumerate(test_loader):
            #     ts_x = ts_x.to(device)
            #     adv_weight.requires_grad = False
            #     adv_bias.requires_grad = False
            #
            #     trans_weight = torch.sigmoid(adv_weight)
            #     trans_bias = th * torch.sigmoid(adv_bias)
            #
            #     pruned = torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0)
            #
            #     loss = - 0.5 * torch.mean(torch.clamp(pruned * torch.sign(ts_x) - ts_x, max=0)) \
            #            + torch.mean(torch.clamp(pruned * torch.sign(ts_x) - ts_x, min=10.))
            #     test_loss += loss.item()

            # print('Epoch: {} Test Loss: {:.6f}'.format(epoch, test_loss / len(test_loader)))

        trans_weight = torch.sigmoid(adv_weight)
        trans_bias = th * torch.sigmoid(adv_bias)

        pruned_ct = torch.ceil(torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0))
        shrunk_super_matrices.append(pruned_ct.detach().cpu().numpy().astype(np.int16))

    np.save('cluster_result/shrunk_super_matrices' + anonymity_suffix + '.npy', shrunk_super_matrices,
            allow_pickle=True)

    shrunk_super_matrices = np.squeeze(shrunk_super_matrices).astype(int)
    with open('json/shrunk_super_matrices' + anonymity_suffix + '.json', 'w') as f:
        json.dump(shrunk_super_matrices.tolist(), f)


if __name__ == '__main__':
    args = parse_args()
    refine(const.DATASET_PATH, args)
