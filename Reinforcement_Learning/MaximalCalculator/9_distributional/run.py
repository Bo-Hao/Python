from __future__ import absolute_import, division, print_function, unicode_literals

from env import Env
from distributional import DistributionalRL
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
            a = RL.choose_action(s)
            # RL take action and get next observation and reward
            s, a, r, s_, done = E.step(a)
            print("do:", a, ', get:', r)
            # RL learn from this transition
            multi_step = s_ -1 + 1
            RL.learn(s, a, r, multi_step, done) # Multi-steps learning +2 so, N-2
            # swap observation
            s = s_
            # break while loop when end of this episode
            if done:
                #RL.epsilon += 0.001
                break
        if episode %30 == 0:
            RL.dump_model.set_weights(RL.model.get_weights())
            
        
    E = Env()
    print("---------------test---------------")
    RL.m.bias_noisy = False
    RL.m.weight_noisy = False
    for i in range(E.final_step):
        q_list = []
        for i in RL.model.predict([s]):
            q_list.append(sum([RL.z[j] * i[0][j] for j in range(RL.atoms)]))
        np.argmax(q_list)


        E.step(np.argmax(q_list))
        print(np.argmax(q_list))
    print(E.score)

if __name__ == "__main__":
    import time
    t = time.time()

    env = Env()
    RL = DistributionalRL(actions=list(range(env.n_actions)))
    update()

    print("total time cost: ", time.time() - t)
    