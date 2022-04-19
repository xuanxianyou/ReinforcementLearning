import gym
import torch
import numpy as np
import tianshou as ts
from torch import nn


# 构建网络
class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            # np.prod()返回数组元素的乘积
            # inplace = True,会改变输入数据的值,节省反复申请与释放内存的空间与时间
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape))
        )

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = {}
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        # view调整tensor的形状
        logits = self.model(obs.view(batch, -1))
        return logits, state


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # 建立并行环境
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make("CartPole-v0") for _ in range(8)])
    test_envs = ts.env.DummyVectorEnv([lambda : gym.make("CartPole-v0") for _ in range(100)])
    # 神经网络
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    net = Net(state_shape=state_shape, action_shape=action_shape)
    # 优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    # 策略
    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optimizer,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=4,
    )
    # 定义采集器
    train_collector = ts.data.Collector(
        policy=policy,
        env=train_envs,
        buffer=ts.data.VectorReplayBuffer(20000, 10),
        exploration_noise=True
    )
    test_collector = ts.data.Collector(
        policy=policy,
        env=test_envs,
        exploration_noise=True
    )
    # 训练器
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10,
        step_per_epoch=10000,
        step_per_collect=10,
        update_per_step=0.1,
        episode_per_test=100,
        batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold
    )
    print(f'Finished training! Use {result["duration"]}')

    policy.eval()
    policy.set_eps(0.05)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=1, render=1 / 35)