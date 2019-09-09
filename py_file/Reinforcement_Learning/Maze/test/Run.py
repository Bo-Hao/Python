from Maze_env import Maze
from RL_brain import QLearningTable
import numpy as np 
import copy

def update():
    for episode in range(300):
        # initial observation
        s = env.reset()
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            a = RL.choose_action(s)

            # RL take action and get next observation and reward
            s_, r, done = env.step(a)
            RL.store_memory(s, a, r, s_)
            
            # RL learn from this transition
            if RL.store_times % 50 == 0: 
                RL.target_model = copy.copy(RL.model)
                RL.learn()
                
                RL.epsilon += 0.01
                print('decay', RL.epsilon)
            if  RL.store_times % 200 == 0:
                RL.target_model = copy.copy(RL.model)
            '''if RL.store_times % 100 == 0:
                RL.target_model = copy.copy(RL.model)'''
            # swap observation
            s = s_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


def test():
    s = env.reset()
    while True:
        # fresh env
        env.render()
        # RL choose action based on observation
        a = RL.choose_action(s)

        # RL take action and get next observation and reward
        s_, r, done = env.step(a)
        # swap observation
        s = s_

        # break while loop when end of this episode
        if done:
            break
    print('test over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

    env = Maze()
    env.after(100, test)
    env.mainloop()