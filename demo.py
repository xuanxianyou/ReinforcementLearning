import gym
import numpy as np


# MountainCar Demo
# env = gym.make("MountainCar-v0")


# print(env.observation_space)
# print(env.action_space)
# print(env.observation_space.low, env.observation_space.high)
# print(env.action_space.n)

class BespokeAgent:
    def __init__(self):
        pass

    def decide(self, observation):
        """
        决策
        :param observation:
        :return:
        """
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):
        """
        学习
        :param args:
        :return:
        """
        pass


def play(env, agent, render=False, learn=False):
    episode_reward = 0
    # 重置environment
    observation = env.reset()
    while True:
        if render:
            # 渲染界面
            env.render()
        action = agent.decide(observation)
        # env执行一步
        observation_, reward, done, _ = env.step(action)
        episode_reward += reward
        if learn:
            agent.learn()
        if done:
            break
        observation = observation_
    return episode_reward


if __name__ == '__main__':
    # 取出environment
    env = gym.make("MountainCar-v0")
    agent = BespokeAgent()
    env.seed(42)
    episode_reward = play(env, agent, render=True)
    env.close()
    print("回合奖励 = {}".format(np.mean(episode_reward)))
