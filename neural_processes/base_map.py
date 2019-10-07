from neural_processes.generic_map import GenericMap

class BaseMap(GenericMap):
    '''
        Parametrizing the mapping used in Neural Processes

        f = BaseMap(..., deterministic=True)
        y = f(x,z)
        -- or --
        f = BaseMap(..., deterministic=False)
        y_mean, y_std = f(x,z)

        The PETS paper (data efficient model-based RL...)
        gets best results when the models are not
        deterministic

        Assuming all inputs and outputs are flat

        if deterministic, output layer has no activation function
        if stochastic, outputs mean and **LOG** diag covariance for a Gaussian
    '''
    def __init__(
        self,
        z_dim,
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
        deterministic=False,
        use_bn=False
    ):
        
        all_input_dims = [z_dim] + input_dims
        super(BaseMap, self).__init__(
            all_input_dims,
            output_dims,
            siamese_input=siamese_input,
            num_siamese_input_layers=num_siamese_input_layers,
            siamese_input_layer_dim=siamese_input_layer_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            siamese_output=siamese_output,
            num_siamese_output_layers=num_siamese_output_layers,
            siamese_output_layer_dim=siamese_output_layer_dim,
            act=act,
            deterministic=deterministic,
            use_bn=use_bn
        )

    def forward(self, z, inputs):
        '''
            Output is:
                deterministic: a list
                not: a list of lists
        '''
        all_inputs = [z] + inputs
        
        return super(BaseMap, self).forward(all_inputs)


# class MazeConvBaseMap():
#     '''
#     Maps actions to (1, h, w) using a linear map and concats
#     it to the input image which is then processed down with convolutions.
    
#     with upconvolution brought back up to the output size
#     '''
#     def __init__(
#             self,
#             kernel_sizes,
#             num_channels,
#             strides,
#             paddings,
#             hidden_sizes,
#             input_size,
#             # output_size, outputs are same dims as input_size and a scalar for the reward
#             action_dim,
#             init_w=3e-3,
#             hidden_activation=F.relu,
#             output_activation=identity,
#             hidden_init=ptu.fanin_init,
#             b_init_value=0.1,
#     ):
#         self.save_init_params(locals())
#         super().__init__()

#         self.kernel_sizes = kernel_sizes
#         self.num_channels = num_channels
#         self.strides = strides
#         self.paddings = paddings
#         self.hidden_activation = hidden_activation
#         self.output_activation = output_activation
#         self.convs = []
#         self.fcs = []

#         in_c = input_size[0]
#         in_h = input_size[1]
#         for k, c, s, p in zip(kernel_sizes, num_channels, strides, paddings):
#             conv = nn.Conv2d(in_c, c, k, stride=s, padding=p)
#             hidden_init(conv.weight)
#             conv.bias.data.fill_(b_init_value)
#             self.convs.append(conv)

#             out_h = int(math.floor(
#                 1 + (in_h + 2*p - k)/s
#             ))

#             in_c = c
#             in_h = out_h
        
#         in_dim = in_c * in_h * in_h
#         for h in hidden_sizes:
#             fc = nn.Linear(in_dim, h)
#             in_dim = h
#             hidden_init(fc.weight)
#             fc.bias.data.fill_(b_init_value)
#             self.fcs.append(fc)

#         self.last_fc = nn.Linear(in_dim, output_size)
#         self.last_fc.weight.data.uniform_(-init_w, init_w)
#         self.last_fc.bias.data.uniform_(-init_w, init_w)

#     def forward(self, input, return_preactivations=False):
#         h = input
#         for conv in self.convs:
#             h = conv(h)
#             h = self.hidden_activation(h)
#         h = h.view(h.size(0), -1)
#         for i, fc in enumerate(self.fcs):
#             h = fc(h)
#             h = self.hidden_activation(h)
#         preactivation = self.last_fc(h)
#         output = self.output_activation(preactivation)
#         if return_preactivations:
#             return output, preactivation
#         else:
#             return output