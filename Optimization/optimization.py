import numpy as np 
from scipy.optimize import minimize


# single variable  
print("\nSingle Variable")

def f1(x,sign = -1):
    x1 = x[0]
    return sign*(-x1**2+3)

Result3 = minimize(f1, [1])
print(Result3.x)
print(Result3.fun)


# multi variables 
print("\nMulti Variables")

def f2(x, sign = 1):
    x1 = x[0]
    x2 = x[1]
    return sign*(x1**3 - 4 * x1 * x2 + 2 * x2**2)

x0=[1,1]
Result2 = minimize(f2, x0)
print(Result2.x)
print(Result2.fun)

# multi variables with constraints
print("\nMulti Variables with constraints")

def f3(x, sign = 1):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return sign*(x1**2 + x2**2 + x3**2)

def constraint1(x, sign = 1):
    return sign * (x[0] + x[1] - 3)

def constraint2(x, sign = 1):
     return sign * (x[0] + x[2] - 5)

x0=[1,1,1]
b1 = (0, np.inf)
b2 = (0, np.inf)
b3 = (0, np.inf)
bonds = [b1, b2, b3]

con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
cons = [con1, con2]

Result3 = minimize(f3, x0, bounds = bonds, constraints=cons)
print(Result3.x)
print(Result3.fun)
