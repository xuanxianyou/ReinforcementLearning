import copy

import torch
import numpy as np
from torch.nn import functional as F


class DDQNAgent(object):
    def __init__(self, q_function, optimizer, loss, n_action, buffer, batch_size, start_step, num_steps=4, gamma=0.9,
                 e_greedy=0.1, sync_step=100, decay=1e-7):
        self.predict = q_function
        self.target = copy.deepcopy(self.predict)
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
        self.sync_step = sync_step
        self.decay = decay

    def decide(self, state):
        """
        规划：贪心选择动作
        :return:
        """
        Q_list = self.predict(state)
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
        # 更新epsilon
        self.epsilon = max(0.001, self.epsilon-self.decay)
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
            states = torch.FloatTensor([item.detach().tolist() for item in states])
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            state_s = torch.FloatTensor([item.detach().tolist() for item in state_s])
            dones = torch.FloatTensor(dones)
            # TD predict
            q_list = self.predict(states)
            actions = F.one_hot(actions.to(torch.int64), self.n_action)
            q_predict = (q_list * actions).sum(dim=1)
            # TD target
            q_list = self.target(state_s)
            q_target = rewards + (1 - dones) * self.gamma * q_list.max(dim=1)[
                0]  # tensor.max() return (values, indices)
            # 梯度更新
            self.optimizer.zero_grad()
            loss = self.loss(q_predict, q_target)
            loss.backward()
            self.optimizer.step()
        # 网络同步
        if self.step_counter % self.sync_step:
            for target_param, predict_param in zip(self.target.parameters(), self.predict.parameters()):
                target_param.data.copy_(predict_param.data)
