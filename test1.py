import gym
from stable_baselines3 import A2C

env = gym.make("CartPole-v1")

model = A2C(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for i in range(10000):
    action, state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()