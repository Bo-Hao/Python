import threading
import time 
import queue

'''def p():
    time.sleep(5)

    print('over', threading.current_thread())


print(threading.active_count())
print(threading.enumerate())
thread1 = threading.Thread(target = p)
thread2 = threading.Thread(target = p)
print(threading.active_count())
print(threading.enumerate())

thread1.start()
thread2.start()

print(threading.active_count())
print(threading.enumerate())
print("py over")'''

import scipy.linalg as s 
import numpy as np 

A = [[-1, 0, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]]

P, L, U = s.lu(A)

print(P)
print(L)
print(U)
print(np.linalg.inv(L))