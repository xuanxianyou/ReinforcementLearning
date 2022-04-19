import gym
import time
from DoubleDQN.train import TrainManger

episodes = 5000


def main():
    env = gym.make("CartPole-v0")
    env.seed(int(time.time()))
    n_observations = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"状态维度: {n_observations}，动作数：{n_actions}")
    trainer = TrainManger(env=env, episodes=episodes)
    for _ in range(episodes):
        total = trainer.train()
        print(f"Total Reward: %d" % total)
        if (_+1) % 100 == 0:
            total = trainer.test()
            print(f"Total Test Reward: %d" % total)
    env.close()


if __name__ == '__main__':
    main()
