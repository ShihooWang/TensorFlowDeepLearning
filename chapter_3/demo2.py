# 作图显示
import matplotlib.pyplot as plt
import numpy as np
# 原始数据
X = [1, 2, 3, 4, 5, 6]
Y = [2.6, 3.4, 4.7, 5.5, 6.47, 7.8]

# 用一次多项式拟合，相当于线性拟合
z1 = np.polyfit(X, Y, 1)

x = np.arange(1, 7)
y = z1[0] * x + z1[1]
plt.figure()
plt.scatter(X, Y)
plt.plot(x, y)
plt.show()
