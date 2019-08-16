import pandas_datareader as pdr 
from SOM import SOM




df1 = pdr.DataReader('AAPL', 'yahoo', '2018-1-1', '2019-1-1')
df2 = pdr.DataReader('MSFT', 'yahoo', '2018-1-1', '2019-1-1')
array1 = df1.values
array2 = df2.values
array = list(array1) + list(array2)


import math
print(math.log(0.05, 0.9))


