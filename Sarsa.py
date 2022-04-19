import time
import numpy as np
import pandas as pd


class SarsaTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.8, e_greedy=0.9):
        np.random.seed(int(time.time()))
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[state]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, state_next, action_next):
        self.check_state_exist(state_next)
        q_predict = self.q_table.loc[state, action]
        if state_next != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_target = reward
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


class SarsaLambdaTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.8, e_greedy=0.9, trace_decay=0.9):
        np.random.seed(int(time.time()))
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def choose_action(self, state):
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[state]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, state_next, action_next):
        self.check_state_exist(state_next)
        q_predict = self.q_table.loc[state, action]
        if state_next != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_target = reward

        # self.eligibility_trace.loc[state, action] += 1
        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state, action] = 1
        self.q_table += self.lr * (q_target - q_predict) * self.eligibility_trace
        self.eligibility_trace *= self.gamma*self.lambda_

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            state_action = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(
                state_action
            )
            self.eligibility_trace = self.eligibility_trace.append(
                state_action
            )
