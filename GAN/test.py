import threading
import time 
import queue

def p():
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
print("py over")