import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.timestep = 0

    def run(self, di):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            di_reward = 0
            for dt in range(di):
                # Given observations, take action and value (V(s))
                # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
                if dt == 0:
                    self.di_obs = np.copy(self.obs)
                    self.di_actions, self.di_values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)
                    # import pdb; pdb.set_trace()    

                obs, rewards, dones, _ = self.env.step(self.di_actions)
                di_reward += rewards * (self.gamma ** dt)

                # Take actions in env and look the results

                self.states = states
                self.dones = dones
                # self.dones = list(map(lamda x, y: x or y, self.dones, dones))
                if self.dones[0]:
                    self.timestep = 0
                    self.obs[0] = self.obs[0]*0
                    break
                self.obs = obs

                self.timestep += 1
            # Append the experiences
            mb_obs.append(np.copy(self.di_obs))
            mb_actions.append(self.di_actions)
            mb_values.append(self.di_values)
            mb_dones.append(self.dones)
            mb_rewards.append(di_reward)

        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]


        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values
