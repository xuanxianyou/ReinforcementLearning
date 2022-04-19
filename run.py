from maze import Maze
from Sarsa import SarsaLambdaTable


def update():
    for episode in range(100):
        # 重置环境
        state = env.reset()
        # 选择动作
        action = sarsa.choose_action(str(state))
        while True:
            # 渲染环境
            env.render()
            # 执行动作
            state_next, reward, done = env.step(action)
            action_next = sarsa.choose_action(str(state_next))
            print(state, '\t',  action, '\t', reward, '\t', state_next)
            # 训练Agent
            sarsa.learn(str(state), action, reward, str(state_next), action_next)
            # 更新State和action
            state = state_next
            action = action_next
            if done:
                break

    print("Game Over!")
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    sarsa = SarsaLambdaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
