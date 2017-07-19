# download matplotlib
# pip3 install matplotlib

import numpy as np
import matplotlib.pyplot as plot

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plot.plot(x, y1, label="sin")
plot.plot(x, y2, label="cos", linestyle="--")
plot.xlabel("x")
plot.ylabel("y")
plot.title('sin & cos')
plot.show()