import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp, identity
from rlkit.torch import pytorch_util as ptu

from copy import deepcopy


class ThreeWayResNetAIRLDisc(ResNetAIRLDisc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_fc = nn.Linear(kwargs['hid_dim'], 3)


class AntLinClassDisc(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act='relu',
        use_bn=True,
        clamp_magnitude=10.0,
        z_dim=None
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude

        self.mod_list = nn.ModuleList([nn.Linear(input_dim, hid_dim)])
        if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())

        for i in range(num_layer_blocks - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
            self.mod_list.append(hid_act_class())
        
        self.mod_list.append(nn.Linear(hid_dim, 1))
        self.model = nn.Sequential(*self.mod_list)

        self.obs_processor = AntLinClassObsGating(z_dim=z_dim)


    def forward(self, obs_batch, act_batch, z_batch=None):
        obs_batch = self.obs_processor(obs_batch, False, z_batch)

        if act_batch is not None:
            to_concat = [obs_batch, act_batch]
            input_batch = torch.cat(to_concat, dim=1)
        else:
            raise NotImplementedError()
        output = self.model(input_batch)
        output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return output


class AntLinClassObsGating(PyTorchModule):
    def __init__(self, clamp_magnitude=10.0, z_dim=0):
        self.save_init_params(locals())
        super().__init__()

        self.z_dim = z_dim
        self.clamp_magnitude = clamp_magnitude
        assert clamp_magnitude > 0.0

        C_EMB_HID = 128
        self.mlp = nn.Sequential(
            nn.Linear(8 + z_dim, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, 1)
        )
    

    def forward(self, obs_batch, wrap_absorbing, z_batch=None):
        assert z_batch is not None
        assert not wrap_absorbing

        ant_obs = obs_batch[:,:-12]
        target_0 = obs_batch[:,-12:-10]
        target_1 = obs_batch[:,-10:-8]
        classification_batch = obs_batch[:,-8:]

        logits = self.mlp(torch.cat([classification_batch, z_batch], dim=-1))
        logits = torch.clamp(logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        gate = F.sigmoid(logits)

        obs_batch = torch.cat(
            [
                ant_obs,
                gate * target_0 + (1.0 - gate) * target_1
            ],
            dim=-1
        )
        return obs_batch


class MlpGAILDisc(Mlp):
    def __init__(self, *args, clamp_magnitude=10.0, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)

        assert clamp_magnitude > 0.
        self.clamp_magnitude = clamp_magnitude

    def forward(self, obs_batch, act_batch):
        input_batch = torch.cat([obs_batch, act_batch], dim=1)
        output = super().forward(input_batch)
        output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return output


class ResnetDisc(PyTorchModule):
    def __init__(
        self,
        hidden_size,
        n_layers,
        output_size,
        input_size,
        hidden_activation=F.tanh,
        output_activation=identity
    ):
        self.save_init_params(locals())
        super().__init__()

        b_init_value = 0.1
        hidden_init = ptu.fanin_init
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.mod_list = nn.ModuleList()
        fc = nn.Linear(input_size, hidden_size)
        hidden_init(fc.weight)
        fc.bias.data.fill_(b_init_value)
        self.mod_list.append(fc)

        for _ in range(n_layers - 1):
            fc = nn.Linear(hidden_size, hidden_size)
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.mod_list.append(fc)
        
        fc = nn.Linear(hidden_size, output_size)
        hidden_init(fc.weight)
        fc.bias.data.fill_(b_init_value)
        self.mod_list.append(fc)

        print(self.mod_list)
    

    def forward(self, obs_batch, act_batch):
        input_batch = torch.cat([obs_batch, act_batch], dim=1)
        x = input_batch
        x = self.mod_list[0](x)
        x = self.hidden_activation(x)

        for i in range(1, len(self.mod_list)-1):
            y = self.mod_list[i](x)
            y = self.hidden_activation(x)
            x = x + y
        x = self.mod_list[-1](x)
        x = self.output_activation(x)

        return x


class SingleColorFetchCustomDisc(PyTorchModule):
    def __init__(self, clamp_magnitude=10.0):
        self.save_init_params(locals())
        super().__init__()

        self.clamp_magnitude = clamp_magnitude
        assert clamp_magnitude > 0.0

        C_EMB_HID = 64
        self.color_embed_mlp = nn.Sequential(
            nn.Linear(3, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, 1)
        )
        OBJ_EMB_HID = 128
        OBJ_EMB_DIM = 64
        self.object_state_embed_mlp = nn.Sequential(
            nn.Linear(6, OBJ_EMB_HID),
            nn.BatchNorm1d(OBJ_EMB_HID),
            nn.ReLU(),
            nn.Linear(OBJ_EMB_HID, OBJ_EMB_HID),
            nn.BatchNorm1d(OBJ_EMB_HID),
            nn.ReLU(),
            nn.Linear(OBJ_EMB_HID, OBJ_EMB_DIM)
        )
        FINAL_HID = 64
        self.final_mlp = nn.Sequential(
            nn.Linear(OBJ_EMB_DIM + 4 + 4, FINAL_HID),
            nn.BatchNorm1d(FINAL_HID),
            nn.ReLU(),
            nn.Linear(FINAL_HID, FINAL_HID),
            nn.BatchNorm1d(FINAL_HID),
            nn.ReLU(),
            nn.Linear(FINAL_HID, 1)
        )
    

    def forward(self, obs_batch, act_batch):
        obj_0_state = torch.cat([obs_batch[:,:3], obs_batch[:,6:9]], dim=-1)
        obj_1_state = torch.cat([obs_batch[:,3:6], obs_batch[:,9:12]], dim=-1)
        obj_0_color = obs_batch[:,12:15]
        obj_1_color = obs_batch[:,15:18]
        gripper_obs = obs_batch[:,-4:]

        color_0_embed = self.color_embed_mlp(obj_0_color)
        color_1_embed = self.color_embed_mlp(obj_1_color)
        color_logits = color_0_embed - color_1_embed
        color_logits = torch.clamp(color_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        color_gate = F.sigmoid(color_logits)
        # color_logits = torch.cat([color_0_embed, color_1_embed], dim=-1)
        # color_gates = F.softmax(color_logits, dim=-1)

        state_0_embed = self.object_state_embed_mlp(obj_0_state)
        state_1_embed = self.object_state_embed_mlp(obj_1_state)
        gated_embed = color_gate*state_0_embed + (1.0 - color_gate)*state_1_embed
        # gated_embed = color_gates[:,0:1]*state_0_embed + color_gates[:,1:2]*state_1_embed

        concat_final_input = torch.cat([gated_embed, gripper_obs, act_batch], dim=-1)
        disc_logits = self.final_mlp(concat_final_input)

        clamped_disc_logits = torch.clamp(disc_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return clamped_disc_logits


class SecondVersionSingleColorFetchCustomDisc(PyTorchModule):
    def __init__(self, clamp_magnitude=10.0):
        self.save_init_params(locals())
        super().__init__()

        self.clamp_magnitude = clamp_magnitude
        assert clamp_magnitude > 0.0

        C_EMB_HID = 32
        self.color_embed_mlp = nn.Sequential(
            nn.Linear(3, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, 1)
        )
        DISC_HID = 128
        self.disc_part = nn.Sequential(
            nn.Linear(6 + 4 + 4, DISC_HID),
            nn.BatchNorm1d(DISC_HID),
            nn.ReLU(),
            nn.Linear(DISC_HID, DISC_HID),
            nn.BatchNorm1d(DISC_HID),
            nn.ReLU(),
            nn.Linear(DISC_HID, 1)
        )
    

    def forward(self, obs_batch, act_batch):
        obj_0_state = torch.cat([obs_batch[:,:3], obs_batch[:,6:9]], dim=-1)
        obj_1_state = torch.cat([obs_batch[:,3:6], obs_batch[:,9:12]], dim=-1)
        obj_0_color = obs_batch[:,12:15]
        obj_1_color = obs_batch[:,15:18]
        gripper_obs = obs_batch[:,-4:]

        color_0_embed = self.color_embed_mlp(obj_0_color)
        color_1_embed = self.color_embed_mlp(obj_1_color)
        color_logits = color_0_embed - color_1_embed
        color_logits = torch.clamp(color_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        color_gate = F.sigmoid(color_logits)

        gated_obj_state = color_gate*obj_0_state + (1.0 - color_gate)*obj_1_state

        concat_final_input = torch.cat([gated_obj_state, gripper_obs, act_batch], dim=-1)
        disc_logits = self.disc_part(concat_final_input)

        clamped_disc_logits = torch.clamp(disc_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return clamped_disc_logits


class ObsGatingV1(PyTorchModule):
    def __init__(self, clamp_magnitude=10.0, z_dim=0):
        self.save_init_params(locals())
        super().__init__()

        self.z_dim = z_dim
        self.clamp_magnitude = clamp_magnitude
        assert clamp_magnitude > 0.0

        C_EMB_HID = 32
        self.color_embed_mlp = nn.Sequential(
            nn.Linear(3 + z_dim, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, C_EMB_HID),
            nn.BatchNorm1d(C_EMB_HID),
            nn.ReLU(),
            nn.Linear(C_EMB_HID, 1)
        )
    

    def forward(self, obs_batch, wrap_absorbing, z_batch=None):
        obj_0_state = torch.cat([obs_batch[:,:3], obs_batch[:,6:9]], dim=-1)
        obj_1_state = torch.cat([obs_batch[:,3:6], obs_batch[:,9:12]], dim=-1)
        obj_0_color = obs_batch[:,12:15]
        obj_1_color = obs_batch[:,15:18]
        gripper_obs = obs_batch[:,18:22]
        if wrap_absorbing:
            absorbing = obs_batch[:,22:23]

        if z_batch is None:
            color_0_embed = self.color_embed_mlp(obj_0_color)
            color_1_embed = self.color_embed_mlp(obj_1_color)
        else:
            color_0_embed = self.color_embed_mlp(torch.cat([obj_0_color, z_batch], dim=1))
            color_1_embed = self.color_embed_mlp(torch.cat([obj_1_color, z_batch], dim=1))
        color_logits = color_0_embed - color_1_embed
        color_logits = torch.clamp(color_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        color_gate = F.sigmoid(color_logits)

        gated_obj_state = color_gate*obj_0_state + (1.0 - color_gate)*obj_1_state
        if wrap_absorbing:
            concat_obs = torch.cat([gated_obj_state, gripper_obs, absorbing], dim=-1)
        else:
            concat_obs = torch.cat([gated_obj_state, gripper_obs], dim=-1)

        return concat_obs


class ThirdVersionSingleColorFetchCustomDisc(PyTorchModule):
    def __init__(self, clamp_magnitude=10.0, state_only=False, wrap_absorbing=False, z_dim=0):
        self.save_init_params(locals())
        super().__init__()

        self.z_dim = z_dim
        self.state_only = state_only
        self.wrap_absorbing = wrap_absorbing
        self.clamp_magnitude = clamp_magnitude
        assert clamp_magnitude > 0.0

        self.obs_processor = ObsGatingV1(clamp_magnitude=self.clamp_magnitude, z_dim=z_dim)
        DISC_HID = 512
        print('\n\nDISC HID IS %d\n\n' % DISC_HID)
        input_dim = 10 if state_only else 14
        # input_dim = 20 if state_only else 24
        if wrap_absorbing: input_dim += 1
        self.disc_part = nn.Sequential(
            nn.Linear(input_dim, DISC_HID),
            nn.BatchNorm1d(DISC_HID),
            nn.ReLU(),
            nn.Linear(DISC_HID, DISC_HID),
            nn.BatchNorm1d(DISC_HID),
            nn.ReLU(),
            nn.Linear(DISC_HID, DISC_HID),
            nn.BatchNorm1d(DISC_HID),
            nn.ReLU(),
            nn.Linear(DISC_HID, 1)
        )
    
    
    def forward(self, obs_batch, act_batch, z_batch=None):
    # def forward(self, obs_batch, act_batch, next_obs_batch, z_batch=None):
        obs_batch = self.obs_processor(obs_batch, self.wrap_absorbing, z_batch)
        # next_obs_batch = self.obs_processor(next_obs_batch, self.wrap_absorbing, z_batch)
        if self.state_only:
            disc_logits = self.disc_part(obs_batch)
            # disc_logits = self.disc_part(obs_batch, next_obs_batch)
        else:
            concat_input = torch.cat([obs_batch, act_batch], dim=-1)
            # concat_input = torch.cat([obs_batch, act_batch, next_obs_batch], dim=-1)
            disc_logits = self.disc_part(concat_input)
        clamped_disc_logits = torch.clamp(disc_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return clamped_disc_logits


class TransferVersionSingleColorFetchCustomDisc(PyTorchModule):
    def __init__(self, clamp_magnitude=10.0, z_dim=0, gamma=0.99, soft_target_V_tau=0.005):
        self.save_init_params(locals())
        super().__init__()

        self.gamma = gamma
        self.z_dim = z_dim
        self.clamp_magnitude = clamp_magnitude
        assert clamp_magnitude > 0.0

        self.obs_processor = ObsGatingV1(clamp_magnitude=self.clamp_magnitude, z_dim=z_dim)
        
        # R_HID = 64
        # print('\n\nR HID IS %d\n\n' % R_HID)
        # input_dim = 10
        # self.r_part = nn.Sequential(
        #     nn.Linear(input_dim, R_HID),
        #     nn.BatchNorm1d(R_HID),
        #     nn.ReLU(),
        #     nn.Linear(R_HID, R_HID),
        #     nn.BatchNorm1d(R_HID),
        #     nn.ReLU(),
        #     nn.Linear(R_HID, 1)
        # )

        # V_HID = 128
        # print('\n\nR HID IS %d\n\n' % V_HID)
        # input_dim = 10
        # V_net = nn.Sequential(
        #     nn.Linear(input_dim, V_HID),
        #     nn.BatchNorm1d(V_HID),
        #     nn.ReLU(),
        #     nn.Linear(V_HID, V_HID),
        #     nn.BatchNorm1d(V_HID),
        #     nn.ReLU(),
        #     nn.Linear(V_HID, 1)
        # )

        R_HID = 256
        print('\n\nR HID IS %d\n\n' % R_HID)
        input_dim = 10
        self.r_part = nn.Sequential(
            nn.Linear(input_dim, R_HID),
            nn.BatchNorm1d(R_HID),
            nn.ReLU(),
            nn.Linear(R_HID, R_HID),
            nn.BatchNorm1d(R_HID),
            nn.ReLU(),
            nn.Linear(R_HID, R_HID),
            nn.BatchNorm1d(R_HID),
            nn.ReLU(),
            nn.Linear(R_HID, 1)
        )

        V_HID = 256
        print('\n\nR HID IS %d\n\n' % V_HID)
        input_dim = 10
        V_net = nn.Sequential(
            nn.Linear(input_dim, V_HID),
            nn.BatchNorm1d(V_HID),
            nn.ReLU(),
            nn.Linear(V_HID, V_HID),
            nn.BatchNorm1d(V_HID),
            nn.ReLU(),
            nn.Linear(V_HID, V_HID),
            nn.BatchNorm1d(V_HID),
            nn.ReLU(),
            nn.Linear(V_HID, 1)
        )

        self.V_part = V_net
        self.target_V_part = deepcopy(V_net)
        self.soft_target_V_tau = soft_target_V_tau
    
    
    def forward(self, obs_batch, act_batch, z_batch=None, pol_log_prob=None, next_obs_batch=None):
        pol_log_prob = torch.clamp(pol_log_prob, min=-10.0, max=10.0)

        obs_batch = self.obs_processor(obs_batch, False, z_batch)
        next_obs_batch = self.obs_processor(next_obs_batch, False, z_batch)

        r = self.r_part(obs_batch)
        V_s = self.V_part(obs_batch)
        V_s_prime = self.target_V_part(next_obs_batch).detach()
        shaping = self.gamma*V_s_prime - V_s
        f = r + shaping

        disc_logits = f - pol_log_prob
        clamped_disc_logits = torch.clamp(disc_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return clamped_disc_logits, r, shaping, V_s


    def _update_target_V_part(self):
        ptu.soft_update_from_to(self.V_part, self.target_V_part, self.soft_target_V_tau)







    #     self.V_part = V_net
    #     # this is a hack so it's not added as a submodule
    #     self.target_V_part = [deepcopy(V_net)]
    #     self.soft_target_V_tau = soft_target_V_tau
    

    # def cuda(self, *args, **kwargs):
    #     super().cuda(*args, **kwargs)
    #     self.target_V_part[0].cuda()
    

    # def forward(self, obs_batch, act_batch, z_batch=None, pol_log_prob=None, next_obs_batch=None):
    #     obs_batch = self.obs_processor(obs_batch, False, z_batch)
    #     next_obs_batch = self.obs_processor(next_obs_batch, False, z_batch)

    #     r = self.r_part(obs_batch)
    #     V_s = self.V_part(obs_batch)
    #     V_s_prime = self.target_V_part[0](next_obs_batch).detach()
    #     shaping = self.gamma*V_s_prime - V_s
    #     f = r + shaping

    #     disc_logits = f - pol_log_prob
    #     clamped_disc_logits = torch.clamp(disc_logits, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
    #     return clamped_disc_logits, r, shaping


    # def _update_target_V_part(self):
    #     ptu.soft_update_from_to(self.V_part, self.target_V_part[0], self.soft_target_V_tau)
