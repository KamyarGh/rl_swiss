import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.pytorch_util import fanin_init

act_dict = {
    'relu': (F.relu, nn.ReLU),
    'tanh': (F.tanh, nn.Tanh),
}


def make_mlp(
    input_dim,
    hidden_dim,
    num_hidden_layers,
    act_nn,
    use_bn=False,
    output_dim=None,
    output_act_nn=None,
    use_rlkit_mlp_hidden_init=False,
    use_rlkit_b_init=False,
    use_rlkit_last_fc_init=False,
    rlkit_last_fc_init_w=3e-3,
    rlkit_b_init_value=0.1,
    ):
    '''
        Makes an "tower" MLP
    '''
    assert num_hidden_layers > 0
    bias = not use_bn
    
    def get_fc(in_dim, out_dim):
        fc = nn.Linear(in_dim, out_dim, bias=bias)
        if use_rlkit_mlp_hidden_init: fanin_init(fc.weight)
        if bias and use_rlkit_b_init: fc.bias.data.fill_(rlkit_b_init_value)
        return fc

    mod_list = nn.ModuleList([get_fc(input_dim, hidden_dim)])
    if use_bn: mod_list.extend([nn.BatchNorm1d(hidden_dim)])
    mod_list.extend([act_nn()])
    for _ in range(num_hidden_layers-1):
        mod_list.extend([get_fc(hidden_dim, hidden_dim)])
        if use_bn: mod_list.extend([nn.BatchNorm1d(hidden_dim)])
        mod_list.extend([act_nn()])
        
    if output_dim is not None:
        fc = nn.Linear(hidden_dim, output_dim)
        if use_rlkit_last_fc_init:
            fc.weight.data.uniform_(-rlkit_last_fc_init_w, rlkit_last_fc_init_w)
            fc.bias.data.uniform_(-rlkit_last_fc_init_w, rlkit_last_fc_init_w)
        mod_list.extend([fc])
        if output_act_nn is not None:
            mod_list.extend([output_act_nn()])

    return nn.Sequential(*mod_list)


class GenericMap(nn.Module):
    '''
        Assuming all inputs and outputs are flat

        if deterministic, output layer has no activation function
        if stochastic, outputs mean and **LOG** diag covariance for a Gaussian
    '''
    def __init__(
        self,
        input_dims,
        output_dims,
        siamese_input=True,
        num_siamese_input_layers=1,
        siamese_input_layer_dim=128,
        num_hidden_layers=1,
        hidden_dim=128,
        siamese_output=True,
        num_siamese_output_layers=1,
        siamese_output_layer_dim=128,
        act='relu',
        use_bn=False,
        deterministic=False,
        use_rlkit_mlp_hidden_init=False,
        use_rlkit_b_init=False,
        use_rlkit_last_fc_init=False,
        rlkit_last_fc_init_w=3e-3
        ):
        super(GenericMap, self).__init__()
        
        self.siamese_input = siamese_input
        self.siamese_output = siamese_output
        self.deterministic = deterministic
        act_fn, act_nn = act_dict[act]
        
        # process the inputs
        if siamese_input:
            assert num_siamese_input_layers > 0
            self.siamese_input_seqs = nn.ModuleList()
            for dim in input_dims:
                self.siamese_input_seqs.extend([
                    make_mlp(
                        dim, siamese_input_layer_dim,
                        num_siamese_input_layers, act_nn,
                        use_bn=use_bn,
                        use_rlkit_mlp_hidden_init=use_rlkit_mlp_hidden_init,
                        use_rlkit_b_init=use_rlkit_b_init,
                        use_rlkit_last_fc_init=use_rlkit_last_fc_init,
                    )
                ])
        
        # pass through common hidden layers
        if siamese_input:
            concat_dim = len(input_dims) * siamese_input_layer_dim
        else:
            concat_dim = sum(input_dims)
        
        assert num_hidden_layers > 0
        self.hidden_seq = make_mlp(
            concat_dim, hidden_dim, num_hidden_layers, act_nn, use_bn=use_bn,
            use_rlkit_mlp_hidden_init=use_rlkit_mlp_hidden_init,
            use_rlkit_b_init=use_rlkit_b_init,
            use_rlkit_last_fc_init=use_rlkit_last_fc_init,
        )

        # compute outputs
        if siamese_output:
            self.siamese_output_seqs = nn.ModuleList()
            for dim in output_dims:
                if not deterministic:
                    mean_seq = make_mlp(
                        hidden_dim, siamese_output_layer_dim,
                        num_siamese_output_layers, act_nn,
                        output_dim=dim, output_act_nn=None,
                        use_bn=use_bn,
                        use_rlkit_mlp_hidden_init=use_rlkit_mlp_hidden_init,
                        use_rlkit_b_init=use_rlkit_b_init,
                        use_rlkit_last_fc_init=use_rlkit_last_fc_init,
                    )
                    log_cov_seq = make_mlp(
                        hidden_dim, siamese_output_layer_dim,
                        num_siamese_output_layers, act_nn,
                        output_dim=dim, output_act_nn=None,
                        use_bn=use_bn,
                        use_rlkit_mlp_hidden_init=use_rlkit_mlp_hidden_init,
                        use_rlkit_b_init=use_rlkit_b_init,
                        use_rlkit_last_fc_init=use_rlkit_last_fc_init,
                    )
                    self.siamese_output_seqs.extend([mean_seq, log_cov_seq])
                else:
                    self.siamese_output_seqs.extend([
                        make_mlp(
                            hidden_dim, siamese_output_layer_dim,
                            num_siamese_output_layers, act_nn,
                            output_dim=dim, output_act_nn=None,
                            use_bn=use_bn,
                            use_rlkit_mlp_hidden_init=use_rlkit_mlp_hidden_init,
                            use_rlkit_b_init=use_rlkit_b_init,
                            use_rlkit_last_fc_init=use_rlkit_last_fc_init,
                        )
                    ])
        else:
            if deterministic:
                self.output_seq = nn.Linear(hidden_dim, sum(output_dims))
            else:
                self.output_mean_seq = nn.Linear(hidden_dim, sum(output_dims))
                self.output_log_cov_seq = nn.Linear(hidden_dim, sum(output_dims))


    def forward(self, inputs):
        '''
            Output is:
                deterministic: a list
                not: a list of lists
        '''
        if self.siamese_input:
            siamese_input_results = list(
                map(
                    lambda z: z[1](z[0]),
                    zip(inputs, self.siamese_input_seqs)
                )
            )
            hidden_input = torch.cat(siamese_input_results, dim=1)
        else:
            hidden_input = torch.cat(inputs, dim=1)
        
        hidden_output = self.hidden_seq(hidden_input)

        if self.siamese_output:
            if self.deterministic:
                outputs = [
                    seq(hidden_output) for seq in self.siamese_output_seqs
                ]
            else:
                outputs = [
                    seq(hidden_output) for seq in self.siamese_output_seqs
                ]
                outputs = [
                    [outputs[2*i], outputs[2*i+1]] for i in range(int(len(outputs)/2))
                ]
        else:
            if self.deterministic:
                outputs = [self.output_seq(hidden_output)]
            else:
                outputs = [[
                    self.output_mean_seq(hidden_output),
                    self.output_log_cov_seq(hidden_output)
                ]]
        
        return outputs
