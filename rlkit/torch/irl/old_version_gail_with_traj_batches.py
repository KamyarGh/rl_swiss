from collections import OrderedDict

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_irl_algorithm import TorchIRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic

# FOR EASY FETCH ENV ----------------------------------------------------------
# acts_max = Variable(ptu.from_numpy(np.array([0.11622048, 0.11837779, 1., 0.05])), requires_grad=False)
# acts_min = Variable(ptu.from_numpy(np.array([-0.11406593, -0.11492375, -0.48009082, -0.005])), requires_grad=False)

# obs_max = np.array([ 1.35211534e+00,  7.59012039e-01,  8.74170327e-01,  1.35216868e+00,
# 7.59075514e-01,  8.65117304e-01,  9.99349991e-03,  9.97504859e-03,
# -5.73782252e-04,  5.14756901e-02,  5.14743797e-02,  3.06240725e-03,
# 1.60782802e-02,  9.09377515e-03,  1.45024249e-03,  1.55772198e-03,
# 1.27349030e-02,  2.10399698e-02,  3.87118880e-03,  1.10660038e-02,
# 2.63549517e-03,  3.08370689e-03,  2.64278933e-02,  2.67708565e-02,
# 2.67707824e-02])
# obs_min = np.array([ 1.32694457e+00,  7.39177494e-01,  4.25007763e-01,  1.33124808e+00,
# 7.39111105e-01,  4.24235324e-01, -9.98595942e-03, -9.98935859e-03,
# -1.10015137e-01,  2.55108763e-06, -8.67902630e-08, -2.71974527e-03,
# -9.63782682e-03, -4.56146656e-04, -1.68586348e-03, -1.55750811e-03,
# -7.64317184e-04, -2.08764492e-02, -3.56580593e-03, -1.05306888e-02,
# -3.47314426e-03, -3.00819907e-03, -1.27082374e-02, -3.65293252e-03,
# -3.65292508e-03])
# goal_max = np.array([1.35216868, 0.75907551, 0.87419374])
# goal_min = np.array([1.33124808, 0.73911111, 0.42423532])
# observation_max = Variable(ptu.from_numpy(np.concatenate((obs_max, goal_max), axis=-1)), requires_grad=False)
# observation_min = Variable(ptu.from_numpy(np.concatenate((obs_min, goal_min), axis=-1)), requires_grad=False)

# SCALE = 0.99
# -----------------------------------------------------------------------------

# FOR SUPER EASY FETCH ENV ----------------------------------------------------

# -----------------------------------------------------------------------------


def concat_trajs(trajs):
    new_dict = {}
    for k in trajs[0].keys():
        if isinstance(trajs[0][k], dict):
            new_dict[k] = concat_trajs([t[k] for t in trajs])
        else:
            new_dict[k] = np.concatenate([t[k] for t in trajs], axis=0)
    return new_dict


class GAILWithTrajBatches(TorchIRLAlgorithm):
    '''
        This is actually AIRL / DAC, sorry!
        
        I did not implement the reward-wrapping mentioned in
        https://arxiv.org/pdf/1809.02925.pdf though
    '''
    # FOR SUPER EASY FETCH
    # acts_max = Variable(ptu.from_numpy(np.array([0.24968111, 0.24899998, 0.24999904, 0.01499934])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24993695, -0.24931063, -0.24999953, -0.01499993])), requires_grad=False)
    # observation_max = Variable(ptu.from_numpy(np.array([0.0152033 , 0.01572069, 0.00401832, 0.02023052, 0.03041435,
    #     0.20169743, 0.05092416, 0.05090878, 0.01017929, 0.01013457])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-1.77039428e-02, -1.64070528e-02, -1.10015137e-01, -2.06485778e-02,
    #     -2.99603855e-02, -3.43990285e-03,  0.00000000e+00, -8.67902630e-08,
    #     -9.50872658e-03, -9.28206220e-03])), requires_grad=False)
    # SCALE = 0.99
    # ------------------------------
    # FOR FETCH (with Max-Ent Demos)
    # observation_max = Variable(ptu.from_numpy(np.array([0.14997844, 0.14999457, 0.0066419 , 0.2896332 , 0.29748688,
    #    0.4510363 , 0.05095725, 0.05090321, 0.01027833, 0.01043796])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-0.14985769, -0.14991582, -0.11001514, -0.29275747, -0.28962639,
    #    -0.01673591, -0.00056493, -0.00056452, -0.00953662, -0.00964976])), requires_grad=False)
    # acts_max = Variable(ptu.from_numpy(np.array([0.24999679, 0.24999989, 0.24999854, 0.01499987])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24999918, -0.24999491, -0.24998883, -0.01499993])), requires_grad=False)
    # SCALE = 0.99
    # ------------------------------
    # FOR IN THE AIR FETCH EASY
    # observation_max = Variable(ptu.from_numpy(np.array([ 0.04999746,  0.04979575,  0.00102964,  0.09834792,  0.10275888,
    #     0.2026911 ,  0.05087222,  0.05089798,  0.01014106,  0.01024989])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([ -4.97249838e-02,  -4.99201765e-02,  -1.10015137e-01,
    #     -9.57695575e-02,  -9.56882197e-02,  -2.95093730e-03,
    #      0.00000000e+00,  -8.67902630e-08,  -9.48171330e-03,
    #     -9.57788163e-03])), requires_grad=False)
    # acts_max = Variable(ptu.from_numpy(np.array([ 0.24997477,  0.24999408,  0.24999995,  0.01499998])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24999714, -0.24999004, -0.24999967, -0.01499985])), requires_grad=False)
    # SCALE = 0.99

    # FOR IN THE AIR FETCH EASY LARGER RANGE
    # acts_max = Variable(ptu.from_numpy(np.array([0.24999906, 0.2499996 , 0.24999867, 0.01499948])), requires_grad=False)
    # acts_min = Variable(ptu.from_numpy(np.array([-0.24999676, -0.2499984 , -0.24999669, -0.01499992])), requires_grad=False)
    # observation_max = Variable(ptu.from_numpy(np.array([0.14967261, 0.14953164, 0.00056922, 0.28737584, 0.29375757,
    # 0.30215514, 0.05092484, 0.05089244, 0.01006456, 0.01010476])), requires_grad=False)
    # observation_min = Variable(ptu.from_numpy(np.array([-1.49660926e-01, -1.49646858e-01, -1.10015137e-01, -2.82999770e-01,
    # -2.85085491e-01, -4.58114691e-03,  0.00000000e+00, -8.67902630e-08,
    # -9.47718257e-03, -9.47846722e-03])), requires_grad=False)
    # SCALE = 0.99
    
    def __init__(
            self,
            env,
            policy,
            discriminator,

            policy_optimizer,
            expert_replay_buffer,

            num_trajs_per_update=8,
            batch_size=1024,

            disc_lr=1e-3,
            disc_optimizer_class=optim.Adam,

            use_grad_pen=True,
            grad_pen_weight=10,

            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            **kwargs
    ):
        assert disc_lr != 1e-3, 'Just checking that this is being taken from the spec file'
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        
        # FOR IN THE AIR FETCH EASY LARGER Z RANGE
        # self.acts_max = Variable(ptu.from_numpy(np.array([0.24995736, 0.2499716 , 0.24999983, 0.01499852])), requires_grad=False)
        # self.acts_min = Variable(ptu.from_numpy(np.array([-0.24989959, -0.24995068, -0.2499989 , -0.01499998])), requires_grad=False)
        # self.observation_max = Variable(ptu.from_numpy(np.array([0.0499439 , 0.04998455, 0.00098634, 0.09421162, 0.10457129,
        # 0.3022664 , 0.05094975, 0.05090175, 0.01024486, 0.01029508])), requires_grad=False)
        # self.observation_min = Variable(ptu.from_numpy(np.array([-4.98090099e-02, -4.97771561e-02, -1.10015137e-01, -9.60775777e-02,
        # -1.03508767e-01, -3.50153560e-03,  0.00000000e+00, -8.67902630e-08,
        # -9.47353981e-03, -9.62584145e-03])), requires_grad=False)
        # self.SCALE = 0.99

        # FOR IN THE AIR FETCH EASY LARGER X-Y RANGE
        # self.acts_max = Variable(ptu.from_numpy(np.array([0.24999749, 0.2499975 , 0.2499998 , 0.01499951])), requires_grad=False)
        # self.acts_min = Variable(ptu.from_numpy(np.array([-0.24999754, -0.24999917, -0.24999704, -0.01499989])), requires_grad=False)
        # self.observation_max = Variable(ptu.from_numpy(np.array([0.14953716, 0.14865454, 0.00155898, 0.28595684, 0.27644423,
        # 0.20200016, 0.05094223, 0.05082468, 0.01033346, 0.0103368 ])), requires_grad=False)
        # self.observation_min = Variable(ptu.from_numpy(np.array([-1.49931348e-01, -1.49895902e-01, -1.10015137e-01, -2.80037372e-01,
        # -2.82756899e-01, -3.44387360e-03,  0.00000000e+00, -8.67902630e-08,
        # -9.53356933e-03, -9.71619128e-03])), requires_grad=False)
        # self.SCALE = 0.99

        # FOR IN THE AIR FETCH EASY LARGER OBJECT RANGE
        self.acts_max = Variable(ptu.from_numpy(np.array([0.24999844, 0.24999035, 0.24999848, 0.01499987])), requires_grad=False)
        self.acts_min = Variable(ptu.from_numpy(np.array([-0.24999948, -0.24999969, -0.24999971, -0.01499985])), requires_grad=False)
        self.observation_max = Variable(ptu.from_numpy(np.array([0.14981718, 0.14922823, 0.00105448, 0.19316468, 0.20144443,
            0.20205348, 0.05088978, 0.05087405, 0.01012868, 0.01011336])), requires_grad=False)
        self.observation_min = Variable(ptu.from_numpy(np.array([-1.49439076e-01, -1.49636276e-01, -1.10015137e-01, -1.99832936e-01,
            -1.96645722e-01, -3.35041414e-03,  0.00000000e+00, -8.67902630e-08,
            -9.49761703e-03, -9.71219664e-03])), requires_grad=False)
        self.SCALE = 0.99

        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            expert_replay_buffer=expert_replay_buffer,
            policy_optimizer=policy_optimizer,
            **kwargs
        )

        self.discriminator = discriminator
        self.rewardf_eval_statistics = None
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
        )

        self.num_trajs_per_update = num_trajs_per_update
        self.traj_len = 65
        self.batch_size = batch_size

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(self.batch_size, 1),
                torch.zeros(self.batch_size, 1)
            ],
            dim=0
        )
        self.bce_targets = Variable(self.bce_targets)
        if ptu.gpu_enabled():
            self.bce.cuda()
            self.bce_targets = self.bce_targets.cuda()
        
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight


    def _normalize_obs(self, observation):
        return observation

        observation = (observation - self.observation_min) / (self.observation_max - self.observation_min)
        observation *= 2 * self.SCALE
        observation -= self.SCALE
        return observation


    def _normalize_acts(self, action):
        return action

        action = (action - self.acts_min) / (self.acts_max - self.acts_min)
        action *= 2 * self.SCALE
        action -= self.SCALE
        return action


    def get_expert_batch(self, batch_size):
        # print('\nExpert')
        # print(len(self.expert_replay_buffer._traj_endpoints))
        # print('\nReplay')
        # print(len(self.replay_buffer._traj_endpoints))

        batch = self.expert_replay_buffer.sample_trajs(self.num_trajs_per_update, keys=['observations', 'actions'])
        for b in batch:
            b['observations'] = b['observations'][:self.traj_len]
            b['actions'] = b['actions'][:self.traj_len]

        batch = concat_trajs(batch)
        if self.batch_size < self.traj_len * self.num_trajs_per_update:
            idx = np.random.choice(self.traj_len * self.num_trajs_per_update, size=self.batch_size, replace=False)
            for k in batch:
                batch[k] = batch[k][idx]
        elif self.batch_size > self.traj_len * self.num_trajs_per_update:
            raise Exception

        batch = np_to_pytorch_batch(batch)
        batch['observations'] = self._normalize_obs(batch['observations'])
        batch['actions'] = self._normalize_acts(batch['actions'])

        if self.wrap_absorbing:
            if isinstance(batch['observations'], np.ndarray):
                obs = batch['observations']
                assert len(obs.shape) == 2
                batch['observations'] = np.concatenate((obs, np.zeros((obs.shape[0],1))), -1)
                if 'next_observations' in batch:
                    next_obs = batch['next_observations']
                    batch['next_observations'] = np.concatenate((next_obs, np.zeros((next_obs.shape[0],1))), -1)
            else:
                raise NotImplementedError()
        return batch
    

    def get_policy_batch(self, batch_size):
        batch = self.replay_buffer.sample_trajs(self.num_trajs_per_update)
        batch = concat_trajs(batch)

        if self.batch_size < self.traj_len * self.num_trajs_per_update:
            idx = np.random.choice(self.traj_len * self.num_trajs_per_update, size=self.batch_size, replace=False)
            for k in batch:
                batch[k] = batch[k][idx]
        elif self.batch_size > self.traj_len * self.num_trajs_per_update:
            raise Exception
        
        return np_to_pytorch_batch(batch)


    def _do_reward_training(self):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        expert_batch = self.get_expert_batch(self.num_trajs_per_update)
        expert_obs = expert_batch['observations']
        expert_actions = expert_batch['actions']

        policy_batch = self.get_policy_batch(self.num_trajs_per_update)
        policy_obs = policy_batch['observations']
        policy_actions = policy_batch['actions']

        obs = torch.cat([expert_obs, policy_obs], dim=0)
        actions = torch.cat([expert_actions, policy_actions], dim=0)

        disc_logits = self.discriminator(obs, actions)
        disc_preds = (disc_logits > 0).type(torch.FloatTensor)
        disc_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = Variable(torch.rand(self.batch_size, 1))
            if ptu.gpu_enabled(): eps = eps.cuda()
            
            interp_obs = eps*expert_obs + (1-eps)*policy_obs
            interp_obs.detach()
            interp_obs.requires_grad = True
            interp_actions = eps*expert_actions + (1-eps)*policy_actions
            interp_actions.detach()
            interp_actions.requires_grad = True
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs, interp_actions).sum(),
                inputs=[interp_obs, interp_actions],
                # grad_outputs=torch.ones(exp_specs['batch_size'], 1).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True
            )
            total_grad = torch.cat([gradients[0], gradients[1]], dim=1)
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()

            disc_loss = disc_loss + gradient_penalty * self.grad_pen_weight

        disc_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.rewardf_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.rewardf_eval_statistics = OrderedDict()
            self.rewardf_eval_statistics['Disc Loss'] = np.mean(ptu.get_numpy(disc_loss))
            self.rewardf_eval_statistics['Disc Acc'] = np.mean(ptu.get_numpy(accuracy))
            self.rewardf_eval_statistics['Grad Pen'] = np.mean(ptu.get_numpy(gradient_penalty))
            self.rewardf_eval_statistics['Grad Pen W'] = np.mean(self.grad_pen_weight)


    def _do_policy_training(self):
        policy_batch = self.get_policy_batch(self.batch_size)
        # If you compute log(D) - log(1-D) then you just get the logits
        policy_batch['rewards'] = self.discriminator(policy_batch['observations'], policy_batch['actions'])
        self.policy_optimizer.train_step(policy_batch)

        self.rewardf_eval_statistics['Disc Rew Mean'] = np.mean(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Std'] = np.std(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Max'] = np.max(ptu.get_numpy(policy_batch['rewards']))
        self.rewardf_eval_statistics['Disc Rew Min'] = np.min(ptu.get_numpy(policy_batch['rewards']))
    

    # if the discriminator has batch norm we have to use the following
    # def _do_policy_training(self):
    #     # this is a hack right now to avoid problems when using batchnorm
    #     # since if we use batchnorm the statistics of the disc will be messed up
    #     # if we only evaluate using policy samples
    #     expert_batch = self.get_expert_batch(self.policy_optim_batch_size)
    #     expert_obs = expert_batch['observations']
    #     expert_actions = expert_batch['actions']

    #     policy_batch = self.get_policy_batch(self.policy_optim_batch_size)
    #     policy_obs = policy_batch['observations']
    #     policy_actions = policy_batch['actions']

    #     obs = torch.cat([expert_obs, policy_obs], dim=0)
    #     actions = torch.cat([expert_actions, policy_actions], dim=0)

    #     # If you compute log(D) - log(1-D) then you just get the logits
    #     policy_batch['rewards'] = self.discriminator(obs, actions)[self.policy_optim_batch_size:]
    #     self.policy_optimizer.train_step(policy_batch)


    @property
    def networks(self):
        return [self.discriminator] + self.policy_optimizer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_optimizer.get_snapshot())
        return snapshot


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
