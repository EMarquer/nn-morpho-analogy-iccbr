import torch
import torch.nn as nn


class CNNEmbedding(nn.Module):
    def __init__(self, voc_size, char_emb_size, use_gru_output=False):
        ''' Character level CNN word embedding.
        
        It produces an output of length 80 by applying filters of different sizes on the input.
        It uses 16 filters for each size from 2 to 6.
        
        Arguments:
        voc_size -- the maximum number to find in the input vectors
        char_emb_size -- the size of the character vectors
        '''
        super().__init__()

        self.voc_size = voc_size
        self.char_emb_size = char_emb_size

        self.embedding = nn.Embedding(self.voc_size, self.char_emb_size)

        self.conv2 = nn.Conv1d(self.char_emb_size, 16, 2, padding = 0)
        self.conv3 = nn.Conv1d(self.char_emb_size, 16, 3, padding = 0)
        self.conv4 = nn.Conv1d(self.char_emb_size, 16, 4, padding = 1)
        self.conv5 = nn.Conv1d(self.char_emb_size, 16, 5, padding = 2)
        self.conv6 = nn.Conv1d(self.char_emb_size, 16, 6, padding = 3)

        self.use_rnn_output = use_gru_output
        if self.use_rnn_output:
            self.rnn2 = nn.GRU(16, 8, batch_first=True, bidirectional=True, num_layers=1)
            self.rnn3 = nn.GRU(16, 8, batch_first=True, bidirectional=True, num_layers=1)
            self.rnn4 = nn.GRU(16, 8, batch_first=True, bidirectional=True, num_layers=1)
            self.rnn5 = nn.GRU(16, 8, batch_first=True, bidirectional=True, num_layers=1)
            self.rnn6 = nn.GRU(16, 8, batch_first=True, bidirectional=True, num_layers=1)

    def get_emb_size(self) -> int:
        if self.use_rnn_output:
            filters_2 = self.rnn2.hidden_size * 2
            filters_3 = self.rnn3.hidden_size * 2
            filters_4 = self.rnn4.hidden_size * 2
            filters_5 = self.rnn5.hidden_size * 2
            filters_6 = self.rnn6.hidden_size * 2
        else:
            filters_2 = self.conv2.out_channels
            filters_3 = self.conv3.out_channels
            filters_4 = self.conv4.out_channels
            filters_5 = self.conv5.out_channels
            filters_6 = self.conv6.out_channels
        return filters_2 + filters_3 + filters_4 + filters_5 + filters_6

    def forward(self, word):
        # Embedds the word and set the right dimension for the tensor
        unk = word<0
        word[unk] = 0
        word = self.embedding(word)
        word[unk] = 0
        word = torch.transpose(word, 1,2)

        # Apply each conv layer -> torch.Size([batch_size, 16, whatever])
        size2 = self.conv2(word)
        size3 = self.conv3(word)
        size4 = self.conv4(word)
        size5 = self.conv5(word)
        size6 = self.conv6(word)

        if self.use_rnn_output:
            size2, _ = self.rnn2(size2.transpose(1,2))
            size3, _ = self.rnn3(size3.transpose(1,2))
            size4, _ = self.rnn4(size4.transpose(1,2))
            size5, _ = self.rnn5(size5.transpose(1,2))
            size6, _ = self.rnn6(size6.transpose(1,2))
            size2 = size2.transpose(1,2)
            size3 = size3.transpose(1,2)
            size4 = size4.transpose(1,2)
            size5 = size5.transpose(1,2)
            size6 = size6.transpose(1,2)

        # Get the max of each channel -> torch.Size([batch_size, 16])
        maxima2 = torch.max(size2, dim = -1)
        maxima3 = torch.max(size3, dim = -1)
        maxima4 = torch.max(size4, dim = -1)
        maxima5 = torch.max(size5, dim = -1)
        maxima6 = torch.max(size6, dim = -1)

        # Concatenate the 5 vectors to get 1 -> torch.Size([batch_size, 80])
        output = torch.cat([maxima2[0], maxima3[0], maxima4[0], maxima5[0], maxima6[0]], dim = -1)

        return output
