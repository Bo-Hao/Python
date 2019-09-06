from __future__ import absolute_import, division, print_function, unicode_literals
from env import Env
from agent import Agent
import numpy as np
import tensorflow as tf 
import copy
import time


def update():
    for episode in range(1000):
        # initial state

        E = Env()
        s = E.return_s()
        while True:
            action = agent.choose_action(s)
            s, a, r, s_, done = E.step(action)
            print('do: ', a, "at: ", s, 'get: ', r)
            
            agent.store(s, a, r, s_, done)
            s = s_
            if done:
                print('------------------', r, E.step_num, '------------------')
                break

        if agent.count >= agent.capacity:
            agent.learn()
            agent.count = 0

        if episode % 20 == 0:
            agent.dump_model = copy.copy(agent.model)
            

    print("---------------test---------------")
    E = Env()
    s = E.return_s()
    while True:
        a = np.argmax(agent.model.predict([[s]]))
        s, a, r, s_, done = E.step(a)
        s = s_
        if done:
            break
    print(s, r)


if __name__ == "__main__":
    env = Env()
    agent = Agent(actions=list(range(env.n_actions)))
    update()
    
    