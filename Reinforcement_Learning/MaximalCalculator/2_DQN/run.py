from env import GrowUp
from RL_brain import DQN
import numpy as np
import tensorflow.keras as tf 

def update():
    for episode in range(100):
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
            if len(RL.memory) >= 200:

                training = np.array(RL.memory)
                RL.q_model.fit(np.array(training[:, 0]), np.array([i for i in training[:, 1]]), epochs = 5)
                RL.memory = []
            # swap observation
            s = s_
            # break while loop when end of this episode
            if done:
                #RL.epsilon += 0.001
                break

    

    G = GrowUp()
    print("test")
    for i in range(env.fin_step):
        q_table = RL.q_model.predict([i])
        G.step(np.argmax(q_table))
    print(G.score)


    
if __name__ == "__main__":
    env = GrowUp()
    RL = DQN(actions=list(range(env.n_actions)))
    
    update()
    
    