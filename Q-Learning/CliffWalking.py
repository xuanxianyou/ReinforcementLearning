"""
Q-learning Tabular methods
To solve CliffWalking
"""

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.8, e_greedy=0.95):
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

    def learn(self, state, action, reward, next_state, done):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table.loc[next_state])
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


def learn(env, agent, episodes):
    reward_list = []
    step_list = []
    for i in range(episodes):
        state = env.reset()
        e_step = 0
        e_reward = 0
        while True:
            if i == 999:
                env.render()
            e_step += 1
            action = agent.choose_action(str(state))
            next_state, reward, done, _ = env.step(action)
            agent.learn(str(state), action, reward, str(next_state), done)
            e_reward += reward
            state = next_state
            if done:
                reward_list.append(e_reward)
                step_list.append(e_step)
                print(f"step:{e_step}")
                break
    env.close()
    x = [i for i in range(episodes)]
    plot(x, reward_list, title="rewards of Q Learning")
    plot(x, step_list, color="blue", title="steps of Q Learning")


def plot(X, Y, title, color="red"):
    plt.plot(X, Y, color=color, linestyle="-")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print(f"状态数：{n_states}, 动作数：{n_actions}")
    agent = QLearning(actions=[i for i in range(n_actions)])
    learn(env, agent, 1000)
    print(agent.q_table)
