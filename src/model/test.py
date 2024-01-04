import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('tkagg')
x = np.arange(0,20,0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()
