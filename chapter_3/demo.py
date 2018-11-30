import numpy as np

 # 原始数据
X = [1, 2, 3, 4, 5, 6]
Y = [2.6, 3.4, 4.7, 5.5, 6.47, 7.8]

 # 用一次多项式拟合，相当于线性拟合

z1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(z1)
print (z1)  #[ 1.          1.49333333]
print (p1)  # 1 x + 1.493