import torch.nn as nn
import torch
from data.Vocab import *


class StructuredEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(StructuredEncoder, self).__init__()
        self.config = config
        self.GRU = nn.GRUCell(input_size=config.word_dims + config.relation_dims,
                              hidden_size=config.gru_hiddens)

        self.padding = nn.Parameter(torch.zeros([1, 1, config.gru_hiddens]),
                                    requires_grad=False)

        self.rel_embedding = nn.Embedding(num_embeddings=vocab.rel_size,
                                          embedding_dim=config.relation_dims,
                                          padding_idx=Vocab.ROOT)

        self.drop_in = nn.Dropout(config.dropout_gru_hidden)

        nn.init.orthogonal_(self.GRU.weight_hh)
        nn.init.orthogonal_(self.GRU.weight_ih)
        nn.init.orthogonal_(self.rel_embedding.weight)
        self.hiddens = None

    def forward(self, cur_step, input_represents, last_arc_index, last_rel_index, edu_lengths):
        batch_size = len(edu_lengths)
        if cur_step is 0:
            # first step
            self.hiddens = None
            padding_hidden = self.padding.repeat(batch_size, 1, 1)
            return padding_hidden
        elif cur_step is 1:
            padding_hidden = self.padding.repeat(batch_size, 1, 1)
            self.hiddens = padding_hidden
            return self.hiddens
        else:
            #
            rel_embeds = self.rel_embedding(last_rel_index)
            gru_input = torch.cat([input_represents, rel_embeds], dim=-1)
            gru_input = self.drop_in(gru_input)
            _, _, hidden_size = self.hiddens.size()
            select_index = last_arc_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hidden_size)
            hx = torch.gather(self.hiddens, dim=1, index=select_index).squeeze(1)
            hidden = self.GRU(gru_input, hx)
            hidden = hidden.unsqueeze(1)
            self.hiddens = torch.cat([self.hiddens, hidden], dim=1)
        return self.hiddens


