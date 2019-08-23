from env import GrowUp
from Q_table import QLearningTable
import numpy as np

def update():
    for episode in range(10000):
        # initial observation
        observation = 0
        env = GrowUp()
        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(observation, action, reward, observation_)
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                #RL.epsilon += 0.001
                break
        print(reward)

    print(RL.q_table)
    print(reward)
    G = GrowUp()
    print("test")
    for i in range(len(RL.q_table)-1):
        G.step(np.argmax(RL.q_table[i]))
    print(G.score)


    
if __name__ == "__main__":
    env = GrowUp()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    
    update()

    