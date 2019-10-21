import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, pretrained_list, encoder_rnn_size, encoder_num_layers, word_dropout=0.5):
        super(Encoder, self).__init__()

        self.encoder_rnn_size = encoder_rnn_size
        self.encoder_num_layers = encoder_num_layers
        pretrained_tensor = torch.FloatTensor(pretrained_list)
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_tensor, freeze = False)
        #nn.init.constant(self.Other_class, 0.)
        self.Other_class = torch.nn.Parameter(torch.zeros(1))
        self.Other_class.require_grad = True
        #Create input dropout parameter
        self.word_dropout = nn.Dropout(word_dropout)
        self.rnn = nn.LSTM(input_size=pretrained_tensor.size(1),
                           hidden_size=encoder_rnn_size,
                           num_layers=encoder_num_layers,
                           batch_first=True,
                           bidirectional=True)

    def return_0class(self):
        return self.Other_class

    def forward(self, input, index, lengths):
        '''
        :param_input: [batch_size, seq_len] tensor
        :return context of input setences with shape of [batch_size, latent_variable_size]
        '''
        #print("Batch size\t"+str(input.size(0)))
        max_length = input.size(1)
        #print("Len"+str(max_length))
        
        lengths, perm_idx = lengths.sort(0, descending=True)
        input = input[perm_idx]
        encoder_input = self.word_embedding(input)
        [batch_size, seq_len, _] = encoder_input.size()
        #print(batch_size)
        encoder_input = self.word_dropout(encoder_input)

        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
            encoder_input, lengths, True)
        #Unfold rnn with zero initial state and get its final state from the last layer
        rnn_out, (_, final_state) = self.rnn(packed_words, None)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, 
                                                             batch_first=True, total_length=max_length)

        #final_state = final_state.view(self.encoder_num_layers, 2, batch_size, self.encoder_rnn_size)[-1]
        #final_state = rnn_out.view(batch_size, seq_len, 2, self.encoder_rnn_size)[:, index[0]]
        final_state = rnn_out.view(batch_size, seq_len, 2, self.encoder_rnn_size)[torch.arange(batch_size), index, :, :]
        h_1, h_2 = final_state[:, 0], final_state[:, 1]
        final_state = torch.cat([h_1, h_2], 1)
        _, unperm_idx = perm_idx.sort(0)
        final_state = final_state[unperm_idx]
        #print(final_state.size())
        return final_state
