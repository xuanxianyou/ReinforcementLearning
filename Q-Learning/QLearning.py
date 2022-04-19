import numpy as np
import pandas as pd
import time

# numpy随机数种子
np.random.seed(42)

# local variable
N_STATES = 6  # 状态的数目
ACTIONS = ['left', 'right']  # 可执行的动作
EPSILON = 0.9  # 贪婪度 ε-greedy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.1  # 移动间隔时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table


def choose_action(state, q_table):
    # 获取某一个状态的所有动作
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        # 如果均匀随机大于ε或 所有Action Q值为0
        action = np.random.choice(ACTIONS)
    else:
        # 否则选择可能性最大的动作
        action = state_actions.idxmax()
    return action


def get_env_feedback(state, action):
    """
    只有当 o 移动到了 T, agent才会得到唯一的一个奖励, 奖励值 R=1, 其他情况都没有奖励.
    :param state:
    :param action:
    :return:
    """
    if action == "right":
        if state == N_STATES - 2:
            state_next = "terminal"
            reward = 1
        else:
            state_next = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_next = state
        else:
            state_next = state - 1
    return state_next, reward


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        print()
        time.sleep(FRESH_TIME)


def Q_Learning():
    # 初始化 Q table
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        # 回合
        step_counter = 0
        state = 0  # 回合初始位置
        is_terminated = False  # 是否回合结束
        update_env(state, episode, step_counter)  # 环境更新
        while not is_terminated:
            # 选取动作
            action = choose_action(state, q_table)
            # 执行动作
            state_next, reward = get_env_feedback(state, action)
            print("==========", state, action, "===================")
            q_predict = q_table.loc[state, action]  # 估算的(状态-行为)值
            if state_next != 'terminal':
                q_target = reward + GAMMA * q_table.loc[state_next, :].max()  # 实际的(状态-行为)值 (回合没结束)
            else:
                q_target = reward  # 实际的(状态-行为)值 (回合结束)
                is_terminated = True  # terminate this episode

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)  # q_table 更新
            state = state_next  # 探索者移动到下一个 state

            update_env(state, episode, step_counter + 1)  # 环境更新

            step_counter += 1
    return q_table


class QLearningTable:
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

    def learn(self, state, action, reward, state_next):
        self.check_state_exist(state_next)
        q_predict = self.q_table.loc[state, action]
        if state_next != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_next, :].max()
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

# if __name__ == '__main__':
#     q_table = Q_Learning()
#     print(q_table)
