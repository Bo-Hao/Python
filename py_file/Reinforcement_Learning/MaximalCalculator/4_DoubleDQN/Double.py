import tensorflow.keras as tf 
import numpy as np
import copy 

class DoubleQ:
    def __init__(self, actions, learning_rate=0.1, gamma=0.7, e_greedy=0):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.record = []
        self.count = 0
        
        self.fq_model = tf.models.Sequential([
            tf.layers.Dense(10, input_shape = (1, ), activation = 'linear'), 
            tf.layers.Dense(10, activation = 'linear'),
            tf.layers.Dense(4, activation = 'linear')
        ])

        self.fq_model.compile(optimizer='adam',
                      loss='MSE')
        
        self.dump = copy.copy(self.fq_model)


    def choose_action(self, s):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.fq_model.predict([s])[0]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose random action
            action = np.random.choice(self.actions)
 
        return action

    def learn(self, s, a, r, s_):
        q_list = self.fq_model.predict([s])[0]
        q_predict = q_list[a]
        
        Qvalue = self.dump.predict([s_])[0][np.argmax(q_list)]
        q_target = r + self.gamma * Qvalue
        

        q_list[a] += self.lr * (q_target - q_predict)  # update
        train2 = np.array(q_list).reshape(1, 4)
        self.record.append([s, a, r, s_, q_list, q_predict, q_target])
        self.count += 1


        
        if len(self.record) == 50:
            X_train = np.array(self.record)[:, 0]
            Y_train = np.array([i for i in np.array(self.record)[:, 4]])
            self.fq_model.fit(X_train, Y_train, epochs = 10)
            self.record = []



