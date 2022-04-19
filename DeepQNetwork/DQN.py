import torch
import numpy as np


class DQNAgent(object):
    def __init__(self, q_function, optimizer, loss, n_action, gamma=0.9, e_greedy=0.1):
        self.q_function = q_function
        self.optimizer = optimizer
        self.loss = loss
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = e_greedy

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
        学习：更新Q函数
        :return:
        """
        # TD predict
        q_list = self.q_function(state)
        q_predict = q_list[action]
        # TD target
        q_list = self.q_function(state_)
        if done:
            q_target = torch.FloatTensor([reward])
        else:
            q_target = reward + self.gamma * q_list.max()
        # 梯度更新
        self.optimizer.zero_grad()
        loss = self.loss(q_predict, q_target)
        loss.backward()
        self.optimizer.step()
