import model
from dataloader import MyDataset, char_map, TestDataset, my_collate
import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
from torch.autograd import Variable


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

    train_dataset = MyDataset('dev', charmap)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, collate_fn=my_collate)

    test_dataset = TestDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create the LAS network
    listener = model.Encoder(input_dim=40, hidden_dim=256)
    speller = model.Decoder(char_count, hidden_dim=256, attention_dim=128)

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
        loss_fn = loss_fn.cuda()

    for epoch in range(epochs):

        losses = []
        count = -1

        total = len(train_dataset) / batch_size
        interval = total // 4

        # scheduler.step()

        for (utterance, frame_lens, Yinput, Ytarget, transcript_lens) in train_dataloader:

            actual_batch_size = len(frame_lens)
            count += 1
            optim.zero_grad()  # Reset the gradients

            # forward
            listener_feature = listener(to_variable(utterance), frame_lens)
            pred_seq = speller(listener_feature, to_variable(Yinput), Yinput.size(-1), True, frame_lens)
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
        #
        # losses = []
        #
        # for (padded_utters, concat_labels, frame_lens, phoneme_lens) in dev_dataloader:
        #     optim.zero_grad()  # Reset the gradients
        #
        #     prediction = my_net(pack_padded_sequence(to_variable(padded_utters.float()), frame_lens))
        #
        #     loss = loss_fn(prediction, Variable(concat_labels.int()),
        #                    Variable(to_tensor(np.array(frame_lens)).int()), Variable(phoneme_lens.int()))
        #     loss_np = loss.data.cpu().numpy() / batch_size
        #     losses.append(loss_np)
        #     # print(losses)
        # val_loss = np.asscalar(np.mean(losses))
        #
        # print("### Validation Loss: {:.4f} ###\n".format(val_loss))
        #
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_file = '%s-state' % name  # overwrite previous best file
        #     torch.save(my_net.state_dict(), best_file)



    # test
    index = 0
    fout = open(name+'.csv', 'w')
    fout.write('Id,Predicted\n')
    for utterance, frame_lens in test_dataloader:
        # input: np array
        # print('Yinput', Yinput.size())

        # forward
        listener_feature = listener(to_variable(utterance), frame_lens.numpy().tolist())
        pred_seq = speller(listener_feature, None, None, False, frame_lens.numpy().tolist())
        pred_seq = pred_seq.cpu().data.numpy()  # B, L, 33

        for b in range(pred_seq.shape[0]):
            trans_dist = pred_seq[b,:,:]

            transcript = ''.join(charlist[np.argmax(trans_dist[i, :])] for i in range(trans_dist.shape[0]))

            fout.write('%d,%s\n' % (index, transcript))
            index += 1


import sys

listener_state = None
speller_state = None

if len(sys.argv) == 3:
    listener_state = sys.argv[1]
    speller_state = sys.argv[2]

train_model(batch_size=32, epochs=10, learn_rate=1e-3, name='try2',
            listener_state=listener_state, speller_state=speller_state)