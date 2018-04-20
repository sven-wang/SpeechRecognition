import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
from torch.utils.data import DataLoader

data_path = './'


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_longtensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).long()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
    # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


# Input: list of transcript strings
# Returns: char2idx map and list of chars
def char_map(filename):
    max_len = 0
    transcripts = np.load(filename)
    charset = set()
    for s in transcripts:
        max_len = max(max_len, len(s))
        for c in s:
            charset.add(c)
    charlist = sorted(list(charset))
    charlist.insert(0, '<s/o>')
    charmap = {}
    for (i, c) in enumerate(charlist):
        charmap[c] = i
    print('charlist', len(charlist))
    print('max_len', max_len)
    return charmap, charlist


# input: list of transcript strings
# Returns: 2 np array of shape (batch, padded_len+1) input, target
# def char2idx(transcripts, padded_len, charmap):
#     batch_size = transcripts.shape[0]
#     input = np.zeros((batch_size, padded_len+1))
#     target = np.zeros((batch_size, padded_len + 1))
#
#     for i, each in enumerate(transcripts):
#         idxlist = [charmap[c] for c in each]
#         input[i, 1: len(each)+1] = idxlist
#         target[i, :len(each)] = idxlist
#
#     return input, target

def char2idx(transcript, padded_len, charmap):
    input = np.zeros( padded_len+1)
    target = np.zeros(padded_len + 1)

    idxlist = [charmap[c] for c in transcript]
    input[1: len(transcript)+1] = idxlist
    target[:len(transcript)] = idxlist

    return input, target


# Input: utterance, Yinput, Ytarget
def my_collate(batch):
    batch_size = len(batch)

    tuples = [(tup[0].shape[0], tup[0], tup[1], tup[2]) for tup in batch]
    tuples.sort(key=lambda x: x[0], reverse=True)  # sort in descending order

    max_len = tuples[0][0]
    # print('max length:', max_len)
    max_trans_len = max([len(tup[1]) for tup in batch])

    padded_utters = np.zeros((batch_size, max_len, 40))
    padded_Yinput = np.zeros((batch_size, max_trans_len))
    padded_Ytarget = np.zeros((batch_size, max_trans_len))

    frame_lens = []
    transcript_lens = []

    for i in range(batch_size):
        frames, Yinput, Ytarget = tuples[i][1:]

        # print(frames.shape)
        # print(Yinput.shape)

        act_len = frames.shape[0]
        trans_len = Yinput.shape[0]

        padded_utters[i, :act_len, :] = frames
        padded_Yinput[i, :trans_len] = Yinput
        padded_Ytarget[i, : trans_len] = Ytarget

        frame_lens.append(act_len)
        transcript_lens.append(trans_len)

    return to_tensor(padded_utters), frame_lens, \
           to_longtensor(np.array(padded_Yinput)), to_longtensor(np.array(padded_Ytarget)), transcript_lens


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, charmap):
        if dataset == 'train':
            utterances = np.load(data_path + 'train.npy')
            transcripts = np.load(data_path + 'train_transcripts.npy')
        elif dataset == 'dev':
            utterances = np.load(data_path + 'dev.npy')
            transcripts = np.load(data_path + 'dev_transcripts.npy')

        self.X = utterances

        if transcripts is not None:
            self.Yinput = []
            self.Ytarget = []
            for i in range(transcripts.shape[0]):
                input, target = char2idx(transcripts[i], len(transcripts[i]), charmap)
                self.Yinput.append(input)
                self.Ytarget.append(target)
                # print(input.shape)
        else:
            # dummy labels
            self.Yinput = np.zeros(len(utterances))
            self.Ytarget = np.zeros(len(utterances))

    def __getitem__(self, index):
        eight_multiples = len(self.X[index]) // 8 * 8
        return self.X[index][:eight_multiples], np.array(self.Yinput[index]), np.array(self.Ytarget[index])

    def __len__(self):
        return len(self.X)


class TestDataset(torch.utils.data.Dataset):

    def __init__(self):

        utterances = np.load(data_path + 'test.npy')
        self.X = utterances

    def __getitem__(self, index):
        eight_multiples = len(self.X[index]) // 8 * 8
        return self.X[index][:eight_multiples], eight_multiples

    def __len__(self):
        return len(self.X)














