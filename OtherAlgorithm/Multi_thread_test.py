import threading 
import time 

class test():
    def __init__(self):
        self.ind = True
        self.i = 1
    def thread1(self):
        while True:
            if self.ind == True:
                print("thread 1 print", self.i)
                self.i += 1
                self.ind = False
            if self.i > 1000:
                break


    def thread2(self):
        while True:
            if self.ind == False:
                print("thread 2 print", self.i)
                self.i += 1
                self.ind = True
            if self.i > 1000:
                break

    def run(self):
        thread1 = threading.Thread(target = self.thread1)
        thread2 = threading.Thread(target = self.thread2)

        thread1.start()
        thread2.start()


if __name__ == "__main__":
    t = time.time()
    t = test()
    t.run()
    print(t - time.time())