import model
from dataloader import MyDataset, char_map, TestDataset, my_collate
import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
from torch.autograd import Variable
import Levenshtein as L
import math

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


def weights_init(layer):
    class_name = layer.__class__.__name__
    range = 0.1
    if class_name == 'LSTM':
        print(class_name)
        # Initialize LSTM weights
        # range = 1.0 / math.sqrt(hidden_dim)
        torch.nn.init.uniform(layer.weight_ih_l0, -range, range)
        torch.nn.init.uniform(layer.weight_hh_l0, -range, range)
    elif class_name == 'LSTMCell':
        print(class_name)
        torch.nn.init.uniform(layer.weight_ih, -range, range)
        torch.nn.init.uniform(layer.weight_hh, -range, range)


def train_model(batch_size, epochs, learn_rate, name, tf_rate, listener_state, speller_state):

    charmap, charlist = char_map(data_path + 'train_transcripts.npy')
    print('char count', len(charlist))
    char_count = len(charlist)

    train_dataset = MyDataset('train', charmap)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, collate_fn=my_collate)

    dev_dataset = MyDataset('dev', charmap)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=1,
                                                   shuffle=False, collate_fn=my_collate)

    test_dataset = TestDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create the LAS network
    listener = model.Encoder(input_dim=40, hidden_dim=256, attention_dim=128)
    speller = model.Decoder(char_count, hidden_dim=256, attention_dim=128, tf_rate=tf_rate)

    # Initialize weights
    listener.apply(weights_init)
    speller.apply(weights_init)

    # [optional] load state dicts
    if listener_state and speller_state:
        listener.load_state_dict(torch.load(listener_state))
        speller.load_state_dict(torch.load(speller_state))

    loss_fn = nn.CrossEntropyLoss(reduce=False)

    LAS_params = list(listener.parameters()) + list(speller.parameters())
    optim = torch.optim.Adam(LAS_params, lr=learn_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.6)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        listener = listener.cuda()
        speller = speller.cuda()
        loss_fn = loss_fn.cuda()

    for epoch in range(epochs):

        losses = []
        count = -1

        total = len(train_dataset) / batch_size
        interval = total // 10

        scheduler.step()

        for (utterance, frame_lens, Yinput, Ytarget, transcript_lens) in train_dataloader:

            actual_batch_size = len(frame_lens)
            count += 1
            optim.zero_grad()  # Reset the gradients

            # forward
            key, value = listener(to_variable(utterance), frame_lens)
            pred_seq = speller(key, value, to_variable(Yinput), Yinput.size(-1), True, frame_lens)
            # print('pred_seq', pred_seq.size())  # B, L, 33
            # print('Ytarget', Ytarget.size())  # B, L

            pred_seq = pred_seq.resize(pred_seq.size(0) * pred_seq.size(1), char_count)

            # create the transcript mask
            transcript_mask = np.zeros((actual_batch_size, max(transcript_lens)))
            # print('max', max(transcript_lens))

            for i in range(actual_batch_size):
                transcript_mask[i, :transcript_lens[i]] = np.ones(transcript_lens[i])
            transcript_mask = to_variable(to_tensor(transcript_mask)).resize(actual_batch_size * max(transcript_lens))

            # loss
            loss = loss_fn(pred_seq, to_variable(Ytarget).resize(Ytarget.size(0) * Ytarget.size(1)))
            loss = torch.sum(loss * transcript_mask) / actual_batch_size

            # backword
            loss.backward()
            loss_np = loss.data.cpu().numpy()
            losses.append(loss_np)

            # clip gradients
            torch.nn.utils.clip_grad_norm(listener.parameters(), 0.25)  # todo: tune???
            torch.nn.utils.clip_grad_norm(speller.parameters(), 0.25)

            # UPDATE THE NETWORK!!!
            optim.step()

            if count % interval == 0:
                print('Train Loss: %.2f  Progress: %d%%' % (loss_np[0], count * 100 / total))

        print("### Epoch {} Loss: {:.4f} ###".format(epoch, np.asscalar(np.mean(losses))))

        torch.save(listener.state_dict(), '%s-listener-e%d' % (name, epoch))
        torch.save(speller.state_dict(), '%s-speller-e%d' % (name, epoch))

        # # validation
        edit_distances = []
        for (utterance, frame_lens, Yinput, Ytarget, transcript_lens) in dev_dataloader:

            actual_batch_size = len(frame_lens)
            count += 1
            assert actual_batch_size == 1

            # forward
            key, value = listener(to_variable(utterance), frame_lens)
            pred_seq = speller(key, value, to_variable(Yinput), Yinput.size(-1), False, frame_lens)  # input prev pred
            prediction = torch.max(pred_seq, dim=2)[1]

            edit_distance = compare(charlist, prediction.cpu().data.numpy(), Ytarget.numpy())
            edit_distances.append(edit_distance)
            # print(edit_distance)

        print("Epoch {} validation edit distance: {:.4f}".format(epoch, np.asscalar(np.mean(edit_distances))))

    # test
    index = 0
    fout = open(name+'.csv', 'w')
    fout.write('Id,Predicted\n')
    for utterance, frame_lens in test_dataloader:
        # input: np array
        # print('Yinput', Yinput.size())

        # forward
        key, value = listener(to_variable(utterance), frame_lens.numpy().tolist())
        pred_seq = speller(key, value, None, None, False, frame_lens.numpy().tolist())
        pred_seq = pred_seq.cpu().data.numpy()  # B, L, 33

        for b in range(pred_seq.shape[0]):
            trans_dist = pred_seq[b,:,:]

            transcript = ''.join(charlist[np.argmax(trans_dist[i, :])] for i in range(trans_dist.shape[0]))

            fout.write('%d,%s\n' % (index, transcript))
            index += 1


def compare(label_map, pred, target):
    pred_list = pred[0]
    target_list = target[0]

    pred_sent = ""
    for character in pred_list:
        if character == 0:
            break
        pred_sent += label_map[character]

    target_sent = ""
    for character in target_list:
        if character == 0:
            break
        target_sent += label_map[character]

    return float(L.distance(pred_sent, target_sent))

import sys

listener_state = None
speller_state = None

if len(sys.argv) == 3:
    listener_state = sys.argv[1]
    speller_state = sys.argv[2]

train_model(batch_size=32, epochs=0, learn_rate=1e-3, name='try9', tf_rate=0.4,
            listener_state=listener_state, speller_state=speller_state)

'''
try2: added attention masking => overfit after epoch 5, best score 23
try3: added teacher force 0.9
try4: initialize 3 differnent initial states for decoder lstm cells => loss pattern no change 
try5: move key/val computation out of attention => no change 
try8: teacher force 0.6, lr decay => not as good as 0.5
try9: add init to lstm weights => no change
try10:try teacher force rate 0.4

'''
