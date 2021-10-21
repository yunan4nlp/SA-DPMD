import torch.nn as nn
from modules.Layer import *
import torch.nn as nn
import math


class StateEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(StateEncoder, self).__init__()
        self.nonlinear1 = NonLinear(input_size=config.word_dims * 2 + config.gru_hiddens * 2 + 3,
                                    hidden_size=config.hidden_size,
                                    activation=nn.Tanh())

        self.nonlinear2 = NonLinear(input_size=config.hidden_size,
                                    hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                    activation=nn.Tanh())

        nn.init.kaiming_uniform_(self.nonlinear1.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')
        nn.init.kaiming_uniform_(self.nonlinear2.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')

        self.use_structure = config.use_structure

    def forward(self, cur_step, structured_hidden, edu_represents, edu_outputs, edu_lengths, feats):
        batch_size, max_edu_len, _ = edu_outputs.size()

        edu_output_i = edu_outputs[:, cur_step, :].unsqueeze(1).repeat(1, max_edu_len, 1)
        edu_represent_i = edu_represents[:, cur_step, :].unsqueeze(1).repeat(1, max_edu_len, 1)

        if self.use_structure:
            _, struct_len, hidden_size = structured_hidden.size()
            if max_edu_len > struct_len:
                padding = torch.zeros([batch_size, max_edu_len - struct_len, hidden_size]).type(structured_hidden.type())
                structured_hidden = torch.cat([structured_hidden, padding], dim=1)
            state_input = torch.cat([edu_output_i, edu_outputs, edu_represent_i, structured_hidden, feats], dim=-1)
        else:
            state_input = torch.cat([edu_output_i, edu_outputs, edu_represent_i, edu_represents, feats], dim=-1)

        hidden = self.nonlinear1(state_input)
        state_hidden = self.nonlinear2(hidden)
        return state_hidden
