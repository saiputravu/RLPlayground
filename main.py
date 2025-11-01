#!/usr/bin/env python
import math
import pickle
import random
from collections import deque, namedtuple
from pprint import pprint
from typing import List, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.envs.classic_control import CartPoleEnv

device = torch.device("mps")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
State = Union[torch.Tensor, None]
Action = torch.Tensor


class ReplayMemory:
    """
    Cyclic buffer to hold episodes during training
    """

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_shape: int, n_actions: int):
        super(DQN, self).__init__()
        self.layers = nn.ParameterList(
            values=[
                nn.Linear(input_shape, 128),
                nn.Linear(128, 128),
                nn.Linear(128, n_actions),
            ]
        )

    def forward(self, x: State) -> Action:
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)


class RLFramework:
    r"""
    The RL learning framework which implements the training loop and plotting logic.

    batch_size:             The number of transitions sampled from the replay buffer.
    gamma:                  The discount factor applied on future rewards.
    eps_start:              The starting value of epsilon in the epsilon greedy approach.
    eps_end:                The final value of epsilon in the epsilon greedy approach.
    eps_decay:              The rate of exponential decay of epsilon, higher means slower decay.
    tau:                    The update rate of the target network.
    learning_rate:          The learning rate of the optimiser.
    replay_memory_capacity: The size of the replay memory.
    criterion:              The objective function to optimise over.
    """

    def __init__(
        self,
        env: gym.Env,
        batch_size: int,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: int,
        tau: float,
        learning_rate: float,
        replay_memory_capacity: int,
        criterion: nn.modules.loss._Loss = nn.SmoothL1Loss(),
    ):
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.env = env

        # The number of actions taken.
        self.steps = 0

        # Tracking the duration of episode lengths.
        self.episodes_durations = []

        self.set_seed()

        state, _ = self.env.reset()
        self.policy_net = DQN(len(state), self.env.action_space.n).to(device)
        self.target_net = DQN(len(state), self.env.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimiser = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(replay_memory_capacity)
        self.criterion = criterion

    def set_seed(self):
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def epsilon_greedy_action_selection(
        self, state: State, no_greedy: bool = False
    ) -> Action:
        if no_greedy:
            sample = 1.0
        else:
            sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * (self.steps / self.eps_decay)
        )
        self.steps += 1

        # EPSILON GREEDY decision
        if sample > eps_threshold:
            with torch.no_grad():
                # .max(1) returns the largest column value of each row.
                # second column on max result -> where max element was found.
                # so we pick action with LARGER expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # The random choice
            return torch.tensor(
                [[self.env.action_space.sample()]], device=device, dtype=torch.long
            )

    def optimise_model(
        self,
    ):
        if len(self.memory) < self.batch_size:
            # Not enough memory, give up.
            return

        # Grab some sample from replay
        transitions = self.memory.sample(self.batch_size)

        # Transpose, from batch-array of Transitions to Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        # Compute a mask of the non-final states and concatenate batch elements
        # Final == terminating state
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute model for Q(s_t, a)
        # We select resultant states batch's columns by actions we would take due to
        # policy.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states. Expected values for actions for
        # non_final_next_states are computed based on "older" target_net. We select
        # their best ward with max(1).values. This is merged based on the mask so
        # that we will have either
        #  1. the expected state value; or
        #  2. 0, if the state is final.
        # Here we take
        # $$
        #    V(s_{t+1}) = max_a Q(s_{t+1}, a)
        # $$
        # This we end up with
        # $$
        #    Q^{\pi} = r + \gamma Q^{\pi}(s', \pi(s'))
        # $$
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # We are computing the loss using huber loss
        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimise the model on the loss
        self.optimiser.zero_grad()
        loss.backward()

        # In-place gradient clipping.
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        # Update parameters by taking a single step.
        self.optimiser.step()

    def train(
        self,
        n_episodes=(
            600
            if torch.cuda.is_available() or torch.backends.mps.is_available()
            else 50
        ),
        max_playout_len=500,
        plot=True,
    ):
        for episode in range(n_episodes):
            # Each episode starts from the beginning
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            for t in range(max_playout_len):
                action = self.epsilon_greedy_action_selection(state)
                observation, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )

                # Give up on termination
                if terminated or truncated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                # Start storing this episode
                reward = torch.tensor([reward], device=device)
                self.memory.push(state, action, next_state, reward)
                state = next_state

                # Learn (one step of optimisation on policy network)
                self.optimise_model()

                # Perform soft update of the target's weight network
                # $$
                #   \theta' := \tau \theta + (1 - \tau) \theta'
                # $$
                pn_state_dict = self.policy_net.state_dict()
                tn_state_dict = self.target_net.state_dict()
                one_minus_tau = 1 - self.tau
                for key in pn_state_dict:
                    tn_state_dict[key] = (
                        pn_state_dict[key] * self.tau
                        + tn_state_dict[key] * one_minus_tau
                    )
                self.target_net.load_state_dict(tn_state_dict)

                if terminated or truncated:
                    self.episodes_durations.append(t + 1)
                    if plot:
                        self.plot_durations()
                    break
        if plot:
            self.plot_durations(show_result=True)
            plt.ioff()
            plt.show()

    def plot_durations(self, show_result: bool = False):
        plt.figure(1)
        durations_t = torch.tensor(self.episodes_durations, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")

        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())

        # Take 100 episode averages and plot those too.
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            # Used as we don't know the first 100 means?
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        # Pause a bit so plots are updated.
        plt.pause(0.001)

    def test(self, human_env: gym.Env, max_playout_len: int = 500):
        state, _ = human_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        run = []
        for t in range(max_playout_len):
            action = self.epsilon_greedy_action_selection(state, no_greedy=True)
            observation, reward, terminated, truncated, info = human_env.step(
                action.item()
            )
            run.append((observation, reward, terminated, truncated, info))
            if terminated or truncated:
                break

            # Update new state.
            state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        pprint(run)
        print(len(run))

    def save_weights(self, savefile: str):
        saved_weights = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }
        with open(savefile, "wb") as f:
            pickle.dump(saved_weights, f)

    def load_weights(self, savefile: str):
        with open(savefile, "rb") as f:
            saved_weights = pickle.load(f)
        self.policy_net.load_state_dict(saved_weights["policy_net"])
        self.target_net.load_state_dict(saved_weights["target_net"])


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


def main():
    _env_docs = CartPoleEnv
    env = gym.make("CartPole-v1")
    rlf = RLFramework(
        env,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        tau=TAU,
        learning_rate=LR,
        replay_memory_capacity=10000,
    )
    # rlf.train(n_episodes=200)
    # rlf.save_weights("saved.pkl")
    rlf.load_weights("saved.pkl")
    rlf.test(gym.make("CartPole-v1", render_mode="human"))


if __name__ == "__main__":
    main()
