from __future__ import absolute_import, division, print_function, unicode_literals

from env import Env
from Brain import NoisyQ
import numpy as np
import tensorflow as tf 
import copy


def update():
    for episode in range(300):
        # initial state
        s = 0
        E = Env()
        while True:
            # RL choose action based on observation
            action = RL.choose_action(s)
            # RL take action and get next observation and reward
            s_, reward, done = E.step(action)
            print(action, reward)
            # RL learn from this transition
            RL.learn(s, action, reward, s + 2) # Multi-steps learning +2 so, N-2
            # swap observation
            s = s_
            # break while loop when end of this episode
            if done:
                #RL.epsilon += 0.001
                break
        if episode %10 == 0:
            RL.dump_model = copy.copy(RL.model)
            
        
    E = Env()
    print("---------------test---------------")
    RL.m.bias_noisy = False
    RL.m.weight_noisy = False
    for i in range(E.final_step):
        q_table = RL.model.predict([i])
        E.step(np.argmax(q_table))
        print(np.argmax(q_table))
    print(E.score)

if __name__ == "__main__":
    env = Env()
    RL = NoisyQ(actions=list(range(env.n_actions)))
    update()
    
    