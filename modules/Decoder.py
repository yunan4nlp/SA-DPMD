from modules.StateEncoder import *


class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super(Decoder, self).__init__()
        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_linear = nn.Linear(in_features=config.mlp_arc_size,
                                    out_features=1, bias=True)

        self.rel_linear = nn.Linear(in_features=config.mlp_rel_size,
                                    out_features=vocab.rel_size, bias=True)

        nn.init.kaiming_uniform_(self.arc_linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_uniform_(self.rel_linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.arc_linear.bias)
        nn.init.zeros_(self.rel_linear.bias)

    def forward(self, state_hidden, arc_masks):
        state_hidden_splits = torch.split(state_hidden, split_size_or_sections=100, dim=-1)
        arc_state_hidden = torch.cat(state_hidden_splits[:self.arc_num], dim=-1)
        rel_state_hidden = torch.cat(state_hidden_splits[self.arc_num:], dim=-1)
        arc_logit = self.arc_linear(arc_state_hidden)
        arc_logit = self.mask_relations(arc_logit, arc_masks)
        rel_logit = self.rel_linear(rel_state_hidden)
        return arc_logit, rel_logit

    def mask_relations(self, input, mask):
        input = input.squeeze(-1)
        mask = (mask - 1) * 1e10
        input = input + mask
        return input


