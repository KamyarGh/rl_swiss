import numpy as np
from gym.core import Env
from gym.spaces import Box


class WrappedAbsorbingEnv(Env):
    def __init__(self, env_to_wrap):
        self._env = env_to_wrap
        
        self.action_space = self._env.action_space
        env_obs_space = self._env.observation_space
        self.obs_len = env_obs_space.shape[0]+1
        if isinstance(env_obs_space, Box):
            if len(env_obs_space.shape) > 1:
                raise NotImplementedError()
            self.observation_space = Box(
                low=min(min(env_obs_space.low), 0.), high=max(max(env_obs_space.high), 1.),
                shape=(self.obs_len,), dtype=env_obs_space.dtype
            )
        else:
            raise NotImplementedError()
        

    def step(self, *args, **kwargs):
        obs, rew, done, info = self._env.step(*args, **kwargs)
        actual_obs = np.append(obs, [0.0])
        if done:
            obs = np.zeros(self.obs_len)
            obs[-1] = 1.0
            info['actual_next_obs'] = actual_obs
        else:
            obs = actual_obs

        return obs, rew, done, info

    
    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)
        actual_obs = np.append(obs, [0.0])
        return actual_obs


    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)
    

    def close(self, *args, **kwargs):
        self._env.close(*args, **kwargs)
    

    def seed(self, *args, **kwargs):
        self._env.seed(*args, **kwargs)
