from modules.Layer import *
from modules.ScaleMix import *

import torch.nn as nn
import math


class StateEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(StateEncoder, self).__init__()
        self.utt_nonlinear = NonLinear(input_size=config.word_dims * 2 + config.gru_hiddens * 2,
                                    hidden_size=config.hidden_size,
                                    activation=nn.Tanh())

        self.sp_nonlinear = NonLinear(input_size=config.gru_hiddens * 2,
                                    hidden_size=config.hidden_size,
                                    activation=nn.Tanh())

        self.nonlinear2 = NonLinear(input_size=config.hidden_size,
                                    hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                    activation=nn.Tanh())

        self.rescale = ScalarMix(mixture_size=2)

        nn.init.kaiming_uniform_(self.utt_nonlinear.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')
        nn.init.kaiming_uniform_(self.sp_nonlinear.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')
        nn.init.kaiming_uniform_(self.nonlinear2.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')

    def forward(self, global_outputs, sp_outputs):
        batch_size, max_edu_len, _ = global_outputs.size()
        global_outputs = global_outputs.unsqueeze(1).repeat(1, max_edu_len, 1, 1)
        utt_state_input = torch.cat([global_outputs, global_outputs.transpose(1, 2)], dim=-1)
        utt_hidden = self.utt_nonlinear(utt_state_input)

        sp_outputs =  sp_outputs.unsqueeze(1).repeat(1, max_edu_len, 1, 1)
        sp_state_input = torch.cat([sp_outputs, sp_outputs.transpose(1, 2)], dim=-1)
        sp_hidden = self.sp_nonlinear(sp_state_input)

        state_hidden = self.rescale([utt_hidden, sp_hidden])
        state_hidden = self.nonlinear2(state_hidden)
        return state_hidden
