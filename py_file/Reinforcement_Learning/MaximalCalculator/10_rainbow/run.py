from __future__ import absolute_import, division, print_function, unicode_literals

from env import Env
from agent import Agent
import numpy as np
import tensorflow as tf 


def update():
    for episode in range(300):
        # initial state
        E = Env()
        s = [0 for i in range(E.final_step)]
        s[0] = 1

        while True:
            # RL choose action based on observation
            a = agent.choose_action(s)
            # RL take action and get next observation and reward
            s, a, r, s_, done = E.step(a)
            print("do:", a, ', at state: ', s, ', get:', r)
            # RL learn from this transition
            
            agent.learn(s, a, r, s_, done) # Multi-steps learning +2 so, N-2
            # swap observation
            s = s_
            # break while loop when end of this episode
            if done:
                #RL.epsilon += 0.001
                break
        if episode %30 == 0:
            agent.dump_model.set_weights(agent.model.get_weights())
            
        
    E = Env()
    print("---------------test---------------")
    for i in range(E.final_step):
        q_list = []
        for i in agent.model.predict([[s]]):
            q_list.append(sum([agent.z[j] * i[0][j] for j in range(agent.atoms)]))
        np.argmax(q_list)
        E.step(np.argmax(q_list))
        print(np.argmax(q_list))
    print(E.score)

if __name__ == "__main__":
    import time
    t = time.time()

    env = Env()
    agent = Agent(actions=list(range(env.n_actions)))
    update()

    print("total time cost: ", time.time() - t)
    