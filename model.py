import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class pBLSTM(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim):
        super(pBLSTM, self).__init__()

        self.BLSTM = torch.nn.LSTM(input_feature_dim * 2, hidden_dim,
                                   num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, input):

        # unpack
        input_val, frame_lens = pad_packed_sequence(input, batch_first=True)
        batch_size, max_length, feature_dim = input_val.size()
        # print(input_val.size())

        # half time resolution
        input_val = input_val.contiguous().view([batch_size, max_length // 2, feature_dim * 2])

        # pack
        frame_lens = [length // 2 for length in frame_lens]
        input_val = pack_padded_sequence(input_val, frame_lens, batch_first=True)
        output, hidden = self.BLSTM(input_val)

        return output, hidden


# **Encoder**
# is a BiLSTM followed by 3 pyramid BiLSTMs. Each has 256 hidden dimensions per direction.
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super(Encoder, self).__init__()
        self.BLSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.pBLSTM1 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pBLSTM2 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pBLSTM3 = pBLSTM(hidden_dim * 2, hidden_dim)

        self.key_linear = nn.Linear(hidden_dim * 2, attention_dim)  # output from bLSTM
        self.value_linear = nn.Linear(hidden_dim * 2, attention_dim)  # output from bLSTM

    def forward(self, input, frame_lens):
        input = pack_padded_sequence(input, frame_lens, batch_first=True)
        output, h = self.BLSTM(input)
        output, h = self.pBLSTM1(output)
        output, h = self.pBLSTM2(output)
        output, h = self.pBLSTM3(output)
        output, _ = pad_packed_sequence(output, batch_first=True)

        key = ApplyPerTime(self.key_linear, output).transpose(1, 2)
        value = ApplyPerTime(self.value_linear, output)  # (N, L, 128)
        return key, value


# **Decoder**
# - Single embedding layer from input characters to hidden dimension.
# - Input to LSTM cells is previous context, previous states, and current character.
# - 3 LSTM cells
# - On top is a linear layer for the query projection (done in Attention class)
# - The results of the query and the LSTM state are passed into a single hidden layer MLP for the character projection
# - The last layer of the character projection and the embedding layer have tied weights
class Decoder(nn.Module):
    def __init__(self, char_count, hidden_dim, attention_dim, tf_rate):
        # assert speller_hidden_dim == listener_hidden_dim
        super(Decoder, self).__init__()
        concat_dim = hidden_dim + attention_dim

        self.embedding = nn.Embedding(num_embeddings=char_count, embedding_dim=hidden_dim)
        self.cell1 = nn.LSTMCell(input_size=concat_dim, hidden_size=hidden_dim)
        self.cell2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.cell3 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.attention = Attention(attention_dim, hidden_dim) # 128, 256

        # character projection
        self.mlp = nn.Linear(concat_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.character_projection = nn.Linear(hidden_dim, char_count)

        # tie embedding and project weights
        self.character_projection.weight = self.embedding.weight

        self.softmax = nn.LogSoftmax(dim=-1)  # todo: remove??

        self.hidden_dim = hidden_dim
        self.char_count = char_count

        # initial states
        self.h00 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)
        self.h01 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)
        self.h02 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)
        self.c00 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)
        self.c01 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)
        self.c02 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)

        self.tf_rate = tf_rate

    # listener_feature (N, T, 256)
    # Yinput (N, L )
    def forward(self, key, value, Yinput, max_len, training, frame_lens):

        # Create a binary mask for attention (N, L/8)
        frame_lens = np.array(frame_lens) // 8
        attention_mask = np.zeros((len(frame_lens), 1, np.max(frame_lens)))  # N, 1, L/8
        for i in range(len(frame_lens)):
            attention_mask[i, 0, :frame_lens[i]] = np.ones(frame_lens[i])
        attention_mask = to_variable(to_tensor(attention_mask))

        # INITIALIZATION
        batch_size = key.size()[0]  # train: N; test: 1

        _, context = self.attention(key, value, self.h02.expand(batch_size, self.hidden_dim).contiguous(), attention_mask)
        # common initial hidden and cell states for LSTM cells
        prev_h = (self.h00.expand(batch_size, self.hidden_dim).contiguous(),
                  self.h01.expand(batch_size, self.hidden_dim).contiguous(),
                  self.h02.expand(batch_size, self.hidden_dim).contiguous())
        prev_c = (self.c00.expand(batch_size, self.hidden_dim).contiguous(),
                  self.c01.expand(batch_size, self.hidden_dim).contiguous(),
                  self.c02.expand(batch_size, self.hidden_dim).contiguous())

        pred_seq = None
        pred_idx = to_variable(torch.zeros(batch_size).long())  # size [N] batch size = 1 for test

        if not training:
            max_len = 500

        for step in range(max_len):

            # 0.9 prob to feed the ground truth as input
            teacher_force = True if np.random.random_sample() < self.tf_rate else False

            # label_embedding from Y input or previous prediction
            if training and teacher_force:
                label_embedding = self.embedding(Yinput[:, step])
            else:
                label_embedding = self.embedding(pred_idx.squeeze()) # make sure size [N]
            # print('label_embedding', label_embedding.size()) # (N, 256)

            rnn_input = torch.cat([label_embedding, context], dim=-1)
            pred, context, attention, prev_h, prev_c = \
                self.forward_step(rnn_input, key, value, prev_h, prev_c, attention_mask)
            pred = pred.unsqueeze(1)

            # debug
            # print('context', context[0, 10: 18])

            # label index for the next loop
            pred_idx = torch.max(pred, dim=2)[1]  # argmax size [1, 1]
            if not training and pred_idx.cpu().data.numpy() == 0:
                break  # end of sentence

            # add to the prediction if not eos
            if pred_seq is None:
                pred_seq = pred
            else:
                pred_seq = torch.cat([pred_seq, pred], dim=1)

        return pred_seq

    def forward_step(self, concat, key, value, prev_h, prev_c, attention_mask):

        h1, c1 = self.cell1(concat, (prev_h[0], prev_c[0]))
        h2, c2 = self.cell2(h1, (prev_h[1], prev_c[1]))
        h3, c3 = self.cell3(h2, (prev_h[2], prev_c[2]))

        attention, context = self.attention(key, value, h3, attention_mask)
        concat = torch.cat([h3, context], dim=1)  # (N, speller_dim + values_dim)

        projection = self.character_projection(self.relu(self.mlp(concat)))
        pred = self.softmax(projection)  # todo: remove??
        # print('### pred', pred.size())

        return pred, context, attention, (h1, h2, h3), (c1, c2, c3)


# helper function for
def ApplyPerTime(input_module, input_x):

    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size,time_steps,-1)


# """
# - Mel vectors (N, L, 40)
# - Keys: (N, L, A)
# - Values: (N, L, B)
# - Decoder produces query (N, A)
# - Perform bmm( query(N, 1, A), keys(N, A, L)) = (N,1,L) this is the energy for each sample and each place in the utterance
# - Softmax over the energy (along the utterance length dimension) to create the attention over the utterance
# - Perform bmm( attention(N, 1, L), values(N,L,B)) = (N, 1, B) to produce the context (B = 256)
# - A: key/query length
# INPUTS:
# - encoder_feature: (N,L,B) where B=256 => keys(N, L, A) => keys(N, A, L)
# - decoder_state: (N, B) => query (N, A) unsqueeze=> (N, 1, A)
class Attention(nn.Module):
    def __init__(self, A=128, hidden_dim=256):
        super(Attention, self).__init__()
        # self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # along the time dimension, which is the last one
        self.query_linear = nn.Linear(hidden_dim, A)

    def forward(self, key, value, decoder_state, attention_mask):

        query = self.query_linear(decoder_state).unsqueeze(1)  # query (N, A) => (N, 1, A)
        energy = torch.bmm(query, key)  # (N,1,L)
        attention = self.softmax(energy)  # (N,1,L)

        # mask attention todo: correct???
        # print('=== Attention ===')
        # print(attention)
        attention = attention * attention_mask
        attention = attention / torch.sum(attention, dim=-1).unsqueeze(2)  # (N,1,L) / (N, 1, 1) = (N,1,L)

        # print('=== Attention after mask and renormalization ===')
        # print(attention)

        context = torch.bmm(attention, value)  # (N, 1, B)
        context = context.squeeze(dim=1)  # (N, B)
        # print(encoder_feature.size(1))
        return attention, context


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
