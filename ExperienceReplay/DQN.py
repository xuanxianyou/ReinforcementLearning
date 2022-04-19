import torch
import numpy as np
from torch.nn import functional as F


class DQNAgent(object):
    def __init__(self, q_function, optimizer, loss, n_action, buffer, batch_size, start_step, num_steps=4, gamma=0.9,
                 e_greedy=0.1):
        self.q_function = q_function
        self.optimizer = optimizer
        self.loss = loss
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = e_greedy
        self.step_counter = 0
        self.num_steps = num_steps
        self.buffer = buffer
        self.batch_size = batch_size
        self.start_step = start_step

    def decide(self, state):
        """
        规划：贪心选择动作
        :return:
        """
        Q_list = self.q_function(state)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def action(self, state):
        """
        动作：探索与利用过程得到的动作
        :param state:
        :return:
        """
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.n_action)
        else:
            action = self.decide(state)
        return action

    def learn(self, state, action, reward, state_, done):
        """
        批量学习：更新Q函数
        :return:
        """
        self.step_counter += 1
        # 装入buffer
        self.buffer.append((state, action, reward, state_, done))
        # 批量更新
        if len(self.buffer) > self.start_step and self.step_counter % self.num_steps == 0:
            experience = self.buffer.sample(batch_size=self.batch_size)
            states, actions, rewards, state_s, dones = zip(*experience)
            # print(states, actions, rewards, state_s, dones)
            states = torch.FloatTensor([item.detach().tolist() for item in states])
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            state_s = torch.FloatTensor([item.detach().tolist() for item in state_s])
            dones = torch.FloatTensor(dones)
            # TD predict
            q_list = self.q_function(states)
            actions = F.one_hot(actions.to(torch.int64), self.n_action)
            q_predict = (q_list * actions).sum(dim=1)
            # TD target
            q_list = self.q_function(state_s)
            q_target = rewards + (1 - dones) * self.gamma * q_list.max(dim=1)[0] # tensor.max() return (values, indices)
            # 梯度更新
            self.optimizer.zero_grad()
            loss = self.loss(q_predict, q_target)
            loss.backward()
            self.optimizer.step()
