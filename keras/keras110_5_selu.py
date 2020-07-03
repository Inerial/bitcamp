import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha = 1.6732632423543772848170429916717):
    return (x>=0)*x + (x<0)*alpha*(np.exp(x)-1)


x = np.arange(-5,5,0.1)
y = selu(x)

print(x.shape, y.shape)

plt.plot(x,y)
plt.grid()
plt.show()