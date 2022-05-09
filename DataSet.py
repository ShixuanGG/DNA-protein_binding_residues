import os

import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader


def ReadFeature(feature_root_dir):
    with open(feature_root_dir) as fr:
        feature = fr.readlines()

    for i in range(len(feature)):
        feature[i] = feature[i].strip().split(' ')
        for j in range(len(feature[i])):
            feature[i][j] = float(feature[i][j])

    return feature


def ReadLabel(label_root_dir):
    with open(label_root_dir) as fr:
        label = fr.readlines()

    for i in range(len(label)):
        label[i] = list(label[i].strip())
        for j in range(len(label[i])):
            if(label[i][j] == '0'):
                label[i][j] = 0.0
            else:
                label[i][j] = 1.0

    return label

def Fill(seqs,filler):
    seq_len = [len(seq) for seq in seqs]
    number = len(seqs)
    max_len = max(seq_len)

    dest_seqs = []

    for i in range(number):
        pad_size = max_len - seq_len[i]
        dest_seqs.append(seqs[i] + [filler for j in range(pad_size)])

    return dest_seqs, max_len


class Dataset(data.Dataset):

    def __init__(self, feature_dir,label_dir):

        self.feature = ReadFeature(feature_dir)
        self.labels = ReadLabel(label_dir)
        #self.len_max = len_max

        if(len(self.feature) != len(self.labels)):
            self.number = -1
            print('###InputError###')
        else:
            self.number = len(self.labels)

    # def Fill(self, seqs, filler):
    #     seq_len = [len(seq) for seq in seqs]
    #     number = len(seqs)
    #     max_len = max(seq_len)
    #
    #     dest_seqs = []
    #
    #     src_mask = torch.fill([max_len,max_len],0)
    #     for i in range(number):
    #         pad_size = max_len - seq_len[i]
    #
    #
    #         dest_seqs.append(seqs[i] + [filler for j in range(pad_size)])
    #
    #     return dest_seqs, max_len

    def __getitem__(self, item):
        # feature, _ = self.Fill(self.feature, 0.0)
        # label, self.maxlen = self.Fill(self.labels, 0)

        x = self.feature[item]
        lab = self.labels[item]
        lab = torch.tensor(lab)
        #label = label.permute(1,0)
        x = torch.tensor(x)
        x = x.reshape([1, -1 , 26])
        seq_len = len(lab)


        return x,lab


    def __len__(self):
        return self.number

    #def max_len(self):



if  __name__ == '__main__':
    dataset = Dataset('./Protein_DNA/feature_combine/train_norm.dat',
                            './Protein_DNA/feature_combine/train_label.dat')
    x = dataset.max_len()
    print(x)
    #print(x1.size(1))
    #dataloader = DataLoader(dataset,64)
