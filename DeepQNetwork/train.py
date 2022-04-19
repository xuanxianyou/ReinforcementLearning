import torch
from DeepQNetwork.DQN import DQNAgent
from DeepQNetwork.MLP import MLP


class TrainManger(object):
    def __init__(self, env, episodes=1000, lr=0.01, gamma=0.9, e_greedy=0.1):
        self.env = env
        n_action = env.action_space.n
        n_state = env.observation_space.shape[0]
        q_function = MLP(n_state, n_action)
        optimizer = torch.optim.AdamW(params=q_function.parameters(), lr=lr)
        loss = torch.nn.MSELoss()
        self.agent = DQNAgent(
            q_function=q_function,
            optimizer=optimizer,
            loss=loss,
            n_action=n_action,
            gamma=gamma,
            e_greedy=e_greedy,
        )
        self.episodes = episodes

    def train(self):
        total = 0
        state = self.env.reset()
        # self.env.render()
        while True:
            action = self.agent.action(torch.FloatTensor(state))
            state_, reward, done, _ = self.env.step(action)
            self.agent.learn(torch.FloatTensor(state), action, reward, torch.FloatTensor(state_), done)
            state = state_
            total += reward
            if done:
                break
        return total

    def test(self):
        total = 0
        state = self.env.reset()
        self.env.render()
        while True:
            action = self.agent.decide(torch.FloatTensor(state))
            state_, reward, done, _ = self.env.step(action)
            state = state_
            total += reward
            if done:
                break
        return total
