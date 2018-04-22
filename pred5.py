import model
from dataloader import MyDataset, char_map, TestDataset, my_collate
import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
from torch.autograd import Variable
import sys

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


def train_model(batch_size, epochs, learn_rate, name, listener_state, speller_state):

    charmap, charlist = char_map(data_path + 'train_transcripts.npy')
    print('char count', len(charlist))
    char_count = len(charlist)

    train_dataset = MyDataset('train', charmap)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, collate_fn=my_collate)

    test_dataset = TestDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create the LAS network
    listener = model.Encoder(input_dim=40, hidden_dim=256, attention_dim=128)
    speller = model.Decoder(len(charlist), hidden_dim=256, attention_dim=128)

    # [optional] load state dicts
    if listener_state and speller_state:
        listener.load_state_dict(torch.load(listener_state))
        speller.load_state_dict(torch.load(speller_state))

    loss_fn = nn.CrossEntropyLoss(reduce=False)

    # todo: correct??
    LAS_params = list(listener.parameters()) + list(speller.parameters())
    optim = torch.optim.Adam(LAS_params, lr=learn_rate, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.8)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        listener = listener.cuda()
        speller = speller.cuda()

    # test
    index = 0
    fout = open(name+'.csv', 'w')
    fout.write('Id,Predicted\n')
    for utterance, frame_lens in test_dataloader:
        # input: np array
        # print('Yinput', Yinput.size())

        # forward
        key, value, = listener(to_variable(utterance), frame_lens.numpy().tolist())
        pred_seq = speller(key, value, None, None, False, frame_lens.numpy().tolist())
        pred_seq = pred_seq.cpu().data.numpy()  # B, L, 33

        for b in range(pred_seq.shape[0]):
            trans_dist = pred_seq[b,:,:]

            transcript = ''.join(charlist[np.argmax(trans_dist[i, :])] for i in range(trans_dist.shape[0]))

            fout.write('%d,%s\n' % (index, transcript))
            index += 1


listener_state = None
speller_state = None

if len(sys.argv) == 3:
    listener_state = sys.argv[1]
    speller_state = sys.argv[2]

train_model(batch_size=1, epochs=0, learn_rate=1e-3, name='try5',
            listener_state=listener_state, speller_state=speller_state)