from env import GrowUp
from fix_Q import FixedQ
import numpy as np
import tensorflow.keras as tf 
import copy

def update():
    for episode in range(1000):
        # initial observation
        s = 0
        env = GrowUp()
        while True:

            # RL choose action based on observation
            action = RL.choose_action(s)
            print("action= ", action)
            # RL take action and get next observation and reward
            s_, reward, done = env.step(action)
            # RL learn from this transition
            RL.learn(s, action, reward, s_)
            # swap observation
            s = s_
            # break while loop when end of this episode
            if done:
                #RL.epsilon += 0.001
                break
        RL.dump = copy.copy(RL.fq_model)
        RL.epsilon += 0.001
    

    G = GrowUp()
    print("test")
    for i in range(env.fin_step):
        q_table = RL.fq_model.predict([i])
        G.step(np.argmax(q_table))
    print(G.score)


    
if __name__ == "__main__":
    env = GrowUp()
    RL = FixedQ(actions=list(range(env.n_actions)))
    update()
    print(RL.count)
    
    