import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha = 0.2):
    return (x>=0)*x + (x<0)*alpha*(np.exp(x)-1)


x = np.arange(-5,5,0.1)
y = elu(x)

print(x.shape, y.shape)

plt.plot(x,y)
plt.grid()
plt.show()