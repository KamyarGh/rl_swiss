import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp, identity
from rlkit.torch import pytorch_util as ptu

from copy import deepcopy

class ObsGating(PyTorchModule):
    def __init__(self, clamp_magnitude=10.0, z_dim=0, use_bn=True):
        self.save_init_params(locals())
        super().__init__()

        self.z_dim = z_dim
        self.clamp_magnitude = clamp_magnitude
        self.use_bn = use_bn
        assert clamp_magnitude > 0.0

        C_EMB_HID = 32
        self.color_embed_list = nn.ModuleList([nn.Linear(3 + z_dim, C_EMB_HID)])
        if use_bn:
            self.color_embed_list.append(nn.BatchNorm1d(C_EMB_HID))
        self.color_embed_list.extend([nn.ReLU(), nn.Linear(C_EMB_HID, C_EMB_HID)])
        if use_bn:
            self.color_embed_list.append(nn.BatchNorm1d(C_EMB_HID))
        self.color_embed_list.extend([nn.ReLU(), nn.Linear(C_EMB_HID, 1)])
        self.color_embed_mlp = nn.Sequential(*self.color_embed_list)
    

    def forward(self, obs_batch, wrap_absorbing, z_batch=None, return_color_logits=False):
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

        if return_color_logits:
            return concat_obs, color_logits
        return concat_obs


class TFuncForFetch(PyTorchModule):
    def __init__(
        self,
        T_clamp_magnitude=3.3,
        gating_clamp_magnitude=10.0,
        state_only=False,
        wrap_absorbing=False,
        D_c_repr_dim=0,
        z_dim=0
    ):
        self.save_init_params(locals())
        super().__init__()

        self.D_c_repr_dim = D_c_repr_dim
        self.z_dim = z_dim
        self.state_only = state_only
        self.wrap_absorbing = wrap_absorbing
        self.T_clamp_magnitude = T_clamp_magnitude
        self.gating_clamp_magnitude = gating_clamp_magnitude
        assert T_clamp_magnitude > 0.0
        assert gating_clamp_magnitude > 0.0

        self.D_c_repr_obs_processor = ObsGating(clamp_magnitude=self.gating_clamp_magnitude, z_dim=D_c_repr_dim)
        self.z_obs_processor = ObsGating(clamp_magnitude=self.gating_clamp_magnitude, z_dim=z_dim)
        DISC_HID = 512
        print('\n\nDISC HID IS %d\n\n' % DISC_HID)
        # concat the two processed obs and the actions
        input_dim = 2*10 if state_only else 2*10+4
        if wrap_absorbing: input_dim += 1
        self.mlp = nn.Sequential(
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
    
    
    def forward(self, obs_batch, act_batch, D_c_repr_batch=None, z_batch=None, return_color_logits=False):
        if return_color_logits:
            D_c_repr_obs_batch, D_c_color_logits = self.D_c_repr_obs_processor(obs_batch, self.wrap_absorbing, D_c_repr_batch, return_color_logits)
            z_obs_batch, z_color_logits = self.z_obs_processor(obs_batch, self.wrap_absorbing, z_batch, return_color_logits)
        else:
            D_c_repr_obs_batch = self.D_c_repr_obs_processor(obs_batch, self.wrap_absorbing, D_c_repr_batch)
            z_obs_batch = self.z_obs_processor(obs_batch, self.wrap_absorbing, z_batch)
        obs_batch = torch.cat([D_c_repr_obs_batch, z_obs_batch], dim=-1)
        if self.state_only:
            outputs = self.mlp(obs_batch)
        else:
            concat_input = torch.cat([obs_batch, act_batch], dim=-1)
            outputs = self.mlp(concat_input)
        clamped_outputs = torch.clamp(outputs, min=-1.0*self.T_clamp_magnitude, max=self.T_clamp_magnitude)

        if return_color_logits:
            return clamped_outputs, D_c_color_logits, z_color_logits
        return clamped_outputs


class OnlyDcTFuncForFetch(PyTorchModule):
    def __init__(
        self,
        T_clamp_magnitude=3.3,
        gating_clamp_magnitude=10.0,
        state_only=False,
        wrap_absorbing=False,
        D_c_repr_dim=0
    ):
        self.save_init_params(locals())
        super().__init__()

        self.D_c_repr_dim = D_c_repr_dim
        self.state_only = state_only
        self.wrap_absorbing = wrap_absorbing
        self.T_clamp_magnitude = T_clamp_magnitude
        self.gating_clamp_magnitude = gating_clamp_magnitude
        assert T_clamp_magnitude > 0.0
        assert gating_clamp_magnitude > 0.0

        self.D_c_repr_obs_processor = ObsGating(clamp_magnitude=self.gating_clamp_magnitude, z_dim=D_c_repr_dim)
        DISC_HID = 512
        print('\n\nDISC HID IS %d\n\n' % DISC_HID)
        # concat the two processed obs and the actions
        input_dim = 10 if state_only else 10+4
        if wrap_absorbing: input_dim += 1
        self.mlp = nn.Sequential(
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
    
    
    def forward(self, obs_batch, act_batch, D_c_repr_batch=None, return_color_logits=False):
        if return_color_logits:
            D_c_repr_obs_batch, D_c_color_logits = self.D_c_repr_obs_processor(obs_batch, self.wrap_absorbing, D_c_repr_batch, return_color_logits)
        else:
            D_c_repr_obs_batch = self.D_c_repr_obs_processor(obs_batch, self.wrap_absorbing, D_c_repr_batch)
        if self.state_only:
            outputs = self.mlp(D_c_repr_obs_batch)
        else:
            concat_input = torch.cat([D_c_repr_obs_batch, act_batch], dim=-1)
            outputs = self.mlp(concat_input)
        clamped_outputs = torch.clamp(outputs, min=-1.0*self.T_clamp_magnitude, max=self.T_clamp_magnitude)

        if return_color_logits:
            return clamped_outputs, D_c_color_logits
        return clamped_outputs
