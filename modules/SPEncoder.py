import torch.nn as nn

class SPEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(SPEncoder, self).__init__()
        self.sp_embeddings = nn.Embedding(num_embeddings=config.max_sp_size, embedding_dim=config.word_dims)

        self.edu_GRU = nn.GRU(input_size=config.word_dims,
                              hidden_size=config.gru_hiddens // 2,
                              num_layers=config.gru_layers,
                              bidirectional=True, batch_first=True)
        self.hidden_drop = nn.Dropout(config.dropout_gru_hidden)
    
    def forward(self, speakers, edu_lengths):
        sp_embeddings = self.sp_embeddings(speakers)
        gru_input = nn.utils.rnn.pack_padded_sequence(sp_embeddings, edu_lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.edu_GRU(gru_input)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.hidden_drop(outputs[0])
        return hidden

