import numpy as np
from itertools import combinations

class Taskassignment:
    def __init__(self, task_metric):
        self.task = task_metric
        
    def Hungary(self):
        task = np.array(self.task)
        def step1(task):
            task  = np.array([task[i] - min(task[i]) for i in np.arange(len(task))])
            return np.array(task)  
                
        def step2(task):
            task = np.array([task.T[i] - min(task.T[i]) for i in np.arange(len(task.T))]).T
            return np.array(task)
        
        def step3(task):
            
            zero_metric = [[1 if task[i, j] == 0 else 0 for j in np.arange(task.shape[0])] for i in np.arange(task.shape[1])]
            zero_metric = np.array(zero_metric)
            comb = [num for num in combinations([i for i in np.arange(len(task) * 2)], len(task) - 1)]
            success = []
            for pair in comb:
                tmp = [[1 if task[i, j] == 0 else 0 for j in np.arange(task.shape[0])] for i in np.arange(task.shape[1])]
                tmp = np.array(tmp)
                for i in pair:
                    if i < len(task):
                        tmp[i] = np.array(tmp[i]) * 2
                    else:
                        tmp = tmp.T
                        tmp[i - 4] = tmp[i - 4] * 2
                        tmp = tmp.T
                if sum([list(tmp[i]).count(1) for i in np.arange(tmp.shape[0])]) == 0:
                    success.append(pair)
                    
            if len(success) != 0:
                return success[0]
            else:
                return False

        def step4(task, number_line):
            if number_line == False:
                return task
            else:
                m = []
                for i in np.arange(len(task)):
                    for j in np.arange(len(task)):
                        if i not in number_line and j+4 not in number_line:
                            m.append(task[i, j])
                m = min(m)
                for i in np.arange(len(task)*2):
                    if i >= len(task):
                        if i in number_line:
                            task = task.T
                            task[i - 4] = task[i - 4] + m
                            task = task.T
                    else:
                        if i not in number_line:
                            task[i] = task[i] - m  
                return task
            
        
        
        
        
        task = step2(step1(task))
        number_line = step3(task)
        tor = 1
        while number_line != False and tor <= 50:
            task = step4(task, number_line)
            number_line = step3(task)
            print("---------", tor, '----------')
            tor += 1
            print(task)
    

   


if __name__ == '__main__':
    t = [[90, 75, 75, 80], [35, 85, 55, 65], [125, 95, 90, 105], [45, 110, 95, 115]]
    print(np.array(t))    
    T = Taskassignment(t)
    T.Hungary()