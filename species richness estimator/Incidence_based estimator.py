import numpy as np 
import random 
import math 
from mpmath import *
from scipy.optimize import minimize
mp.dps = 53

    
def C_(a, b):
    
    return factorial(int(a)) / factorial(int(a-b)) / factorial(int(b))


class CMLE():
    def __init__(self):
        self.S = 300
        self.T = 100
        self.t = 10
        
        
        self.create_data()
        self.sample_samples(self.t)
        self.cal_D()
        self.cal_fk()


    def create_data(self):
        population = []
        for i in range(self.T ):
            n = int(np.random.normal(20, 1, 1))
            population.append(np.random.randint(1, self.S, n))
        self.population = population
        
    def sample_samples(self, n):
        samples = random.sample(self.population, n)
        self.samples = samples
        

    def cal_D(self):
        total = []
        for i in range(len(self.samples)):
            total += list(self.samples[i])
        self.D_list = set(total)


    def cal_fk(self):
        fk_list = np.zeros(self.t)
        for d in self.D_list:
            in_it = 0 
            for i in range(len(self.samples)):
                if d in self.samples[i]:
                    in_it += 1
            fk_list[in_it-1] += 1
        self.fk_list = fk_list

    def K(self, alpha, beta):
        term1 = gamma(alpha) * (gamma(beta) / gamma(alpha + beta))
        term2 = gamma(alpha) * (gamma(beta + self.T) / gamma(alpha + beta + self.T))

        K = (term1 - term2)**(-1)
        return K

    def cal_P(self, alpha, beta):
        p_list = []
        K = self.K(alpha, beta)
        for x in range(self.t+1):
            if x == 0:
                term1 = gamma(alpha) * gamma(self.t + beta) / gamma(self.t + alpha + beta)
                term2 = gamma(alpha) * gamma(self.T + beta) / gamma(self.T + alpha + beta)
                p = K * (term1 - term2)
            else: 
                p = K * C_(self.t, x) *gamma(alpha + x)*gamma(self.t + beta - x)/ gamma(self.t + alpha + beta)
                
            p_list.append(p)

        return p_list 


    def cal_Lc(self, alpha, beta):
        D = len(self.D_list)
        p0 = self.p_list[0]
        term_down = 1.
        term_right = 1.

        for k in range(self.t):
            term_down *= math.factorial(self.fk_list[k])
            term_right *= (self.p_list[k+1]/(1 - self.p_list[0]))**(int(self.fk_list[k]))
        
        return math.factorial(D) / term_down * term_right


    def ln_L(self, x, sign = -1): 
        res = 0.0
        p_list = self.cal_P(x[0], x[1])
        fk_list = C.fk_list
        for k in range(self.t):
            res += fk_list[k] * (np.log(float(p_list[k+1])) - np.log(1-float(p_list[0])))

        return sign * res


if __name__ == "__main__":
  
    C = CMLE()
    C.S = 300

    
    C.T = 100
    C.t = 10

    x0 = [1, 1]
    
    Result = minimize(C.ln_L, x0)
    
    print(Result.x, Result.fun, Result.success)

    

    D = len(C.D_list)
    T = C.T
    t = C.t
    #alpha, beta = Result.x[0]/Result.x[0], Result.x[1]/Result.x[0]
    alpha, beta = Result.x[0], Result.x[1]
    term1 = 1 - gamma(alpha + beta) * gamma(T + beta) / gamma(beta) / gamma(T + alpha + beta)
    term2 = 1 - gamma(alpha + beta) * gamma(t + beta) / gamma(beta) / gamma(t + alpha + beta)

    S_CMLE = D * (term1/term2)

    print(S_CMLE,"\n", alpha, beta)



    