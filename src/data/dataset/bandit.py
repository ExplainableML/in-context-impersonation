from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from gymnasium.envs.registration import register
from torch.distributions import Normal
from torch.utils.data import Dataset


class Sams2armedbandit(gym.Env):
    def __init__(
        self,
        num_actions,
        mean_var,
        reward_var,
        no_trials,
        batch_size=1,
        decimals=1,
        sample_from_hd=False,
    ):
        super().__init__()
        # make trials public
        self.no_trials = no_trials

        self._num_actions = num_actions
        self._no_trials = no_trials
        self._batch_size = batch_size
        self._device = None
        self._reward_std = np.sqrt(reward_var)
        self._mean_std = np.sqrt(mean_var)
        self._decimals = decimals
        self._sample_from_hd = sample_from_hd

        self.action_space = spaces.Discrete(num_actions)
        # Dummy space
        self.observation_space = spaces.Box(0, 1, shape=(1,), dtype=int)

        # init
        self.R_chosen_list = []

    def reset(self, means=None, priors="default"):
        self.t = 0
        if means is not None:
            means = torch.from_numpy(means)
            self.mean1 = means[0]
            self.mean2 = means[1]
        elif self._sample_from_hd:
            self.mean1 = torch.Tensor(
                [int(pd.read_csv(self._sample_from_hd)["mu1"].sample())]
            )
            self.mean2 = torch.Tensor(
                [int(pd.read_csv(self._sample_from_hd)["mu2"].sample())]
            )
        else:
            self.mean1 = Normal(0, self._mean_std).sample()
            self.mean2 = Normal(0, self._mean_std).sample()
        self.R1s = Normal(
            self.mean1 * np.ones(self._no_trials), self._reward_std
        ).sample()
        self.R2s = Normal(
            self.mean2 * np.ones(self._no_trials), self._reward_std
        ).sample()
        self._rewards = torch.stack((self.R1s, self.R2s), dim=1)

        # Priors for Kalman fiter belief updating
        if priors == "default":
            self._exp_rew1 = self._exp_rew2 = 0
            self._exp_var1 = self._exp_var2 = self._mean_std**2
        else:
            self._exp_rew1 = self._exp_rew2 = priors[0]
            self._exp_var1 = self._exp_var2 = priors[1]

        # Data gathering for probit regression
        self.V = []
        self.RU = []
        self.TU = []
        self.action1 = []
        self.Regret_list = []
        self.reward = []

    def step(self, action):
        if action == 0:
            self.action1.append(True)
        elif action == 1:
            self.action1.append(False)
        else:
            raise Exception(f"LLM has given an invalid action: {action}")
        # Action1
        self.V.append(self._exp_rew1 - self._exp_rew2)
        self.RU.append(np.sqrt(self._exp_var1) - np.sqrt(self._exp_var2))
        self.TU.append(np.sqrt(self._exp_var1 + self._exp_var2))
        R_chosen = self._rewards[self.t][action]
        self.reward.append(R_chosen.numpy())

        # Regret = 0 if torch.max(self._rewards[self.t]) == R_chosen else torch.max(self._rewards[self.t]).numpy() - R_chosen.numpy()
        Regret = (
            torch.max(self.mean1, self.mean2) - self.mean1
            if action == 0
            else torch.max(self.mean1, self.mean2) - self.mean2
        )
        self.R_chosen_list.append(R_chosen)
        self.Regret_list.append(Regret.numpy())
        self.t += 1

        # Belief updating based on Gaussian assumptions using Kalman filter
        if action == 0:
            lr = self._exp_var1 / (self._exp_var1 + self._reward_std**2)
            self._exp_var1 -= lr * self._exp_var1
            self._exp_rew1 += lr * (R_chosen.numpy() - self._exp_rew1)
        elif action == 1:
            lr = self._exp_var2 / (self._exp_var2 + self._reward_std**2)
            self._exp_var2 -= lr * self._exp_var2
            self._exp_rew2 += lr * (R_chosen.numpy() - self._exp_rew2)

        done = True if (self.t >= self._no_trials) else False

        return None, R_chosen, done, {}


class Bandit(Dataset):
    def __init__(
        self,
        cache_dir: str,
        train: bool = True,
        num_classes: int = 2,
        num_games: int = 200,
        num_trials: int = 10,
    ) -> None:
        super().__init__()

        register(
            id="Sams2armedbandit-v0",
            entry_point="data.dataset.bandit:Sams2armedbandit",
        )

        self.num_games = num_games
        self.num_trials = num_trials

        self.idx_to_class: dict[int, str] = {
            i: name for i, name in enumerate(["1", "2"])
        }

    def __len__(self):
        return self.num_games

    def __getitem__(self, index) -> Any:
        env = gym.make(
            "Sams2armedbandit-v0",
            no_trials=self.num_trials,
            num_actions=2,
            mean_var=100,
            reward_var=1,
            sample_from_hd=False,
        )

        return {"env": env}


if __name__ == "__main__":
    sc = Bandit()
    item = sc[0]
