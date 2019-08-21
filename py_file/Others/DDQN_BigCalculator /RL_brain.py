import tensorflow.keras as tf 
import numpy as np

class DQN:
    def __init__(self, actions, learning_rate=0.1, gamma=0.9, e_greedy=0.1):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.q_model = tf.models.Sequential([
            tf.layers.Dense(8, input_shape = (1, ), activation = 'tanh'), 
            tf.layers.Dense(4, activation = 'linear')
        ])
        self.q_model.compile(optimizer='adam',
                      loss='MSE')
    def choose_action(self, s):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_model.predict([s])[0]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose random action
            action = np.random.choice(self.actions)
 
        return action

    def learn(self, s, a, r, s_):
        q_list = self.q_model.predict([s])[0]
        q_predict = q_list[a]
        q_target = r + self.gamma * (max(self.q_model.predict([s_])[0]))
        

        q_list[a] += self.lr * (q_target - q_predict)  # update
        train2 = np.array(q_list).reshape(1, 4)

        self.q_model.fit([s], train2, epochs = 5)

