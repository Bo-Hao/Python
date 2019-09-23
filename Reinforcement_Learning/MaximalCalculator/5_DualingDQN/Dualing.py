import tensorflow.keras as tf 
import numpy as np
import copy 


class DualingQ:
    def __init__(self, actions, learning_rate=0.1, gamma=0.7, e_greedy=0):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.record = []
        self.count = 0
        
        
        self.model = self.build_model(1, 10, len(actions))
        self.dump_model = copy.copy(self.model)

    def build_model(self, state_size, neurons, action_size):
        state_input = tf.layers.Input(shape=(state_size, ))
        D1 = tf.layers.Dense(neurons, activation='sigmoid')(state_input)

        #連結層
        d1 = tf.layers.Dense(neurons,activation='relu')(D1)
        d2 = tf.layers.Dense(neurons,activation='relu')(d1)
        
        #dueling
        d3_a = tf.layers.Dense(neurons/2, activation='relu')(d2)
        d3_v = tf.layers.Dense(neurons/2, activation='relu')(d2)
        a = tf.layers.Dense(action_size,activation='linear')(d3_a)
        value = tf.layers.Dense(1,activation='linear')(d3_v)
        a_mean = tf.layers.Lambda(lambda x: tf.backend.mean(x, axis=1, keepdims=True))(a)
        advantage = tf.layers.Subtract()([a, a_mean])
        q = tf.layers.Add()([value, advantage])

        #最後compile
        model = tf.models.Model(inputs=state_input, outputs=q)
        model.compile(loss = 'mse', optimizer = tf.optimizers.Adam(lr=0.001))

        return model

    def choose_action(self, s):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.model.predict([s])[0]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose random action
            action = np.random.choice(self.actions)
 
        return action

    def learn(self, s, a, r, s_):
        q_list = self.model.predict([s])[0]
        q_predict = q_list[a]
        
        qvalue = self.dump_model.predict([s_])[0][np.argmax(q_list)]
        q_target = r + self.gamma * qvalue
        

        q_list[a] += self.lr * (q_target - q_predict)  # update
        train2 = np.array(q_list).reshape(1, 4)
        self.record.append([s, a, r, s_, q_list, q_predict, q_target])
        self.count += 1

        if len(self.record) == 50:
            X_train = np.array(self.record)[:, 0]
            Y_train = np.array([i for i in np.array(self.record)[:, 4]])
            self.model.fit(X_train, Y_train, epochs = 10)
            self.record = []



