import numpy as np
from numpy import pi

# Arange is same as range function in python
a = np.arange(15).reshape(3,5)
print("A arange:",a)

# STATS
print("shape:", a.shape,"\n")
print("ndim:",a.ndim)
print("dtype.name:",a.dtype.name)
print("itemsize:",a.itemsize)
print("size:",a.size)

# Find TYPE
b = np.array([6, 7, 8])
b
print("Type of B:",type(b))

# Create an Array
import numpy as np
n = np.array([2, 3, 4])
b = np.array([(1.5, 2, 3), (4, 5, 6)])

print("NEW ARRAY 1: ",n)
print("NEW ARRAY 2: ",b)

# LINSPACE
lin = np.linspace(0, 2, 9)                   # 9 numbers from 0 to 2
print ("LINSPACE",lin)

x = np.linspace(0, 2 * pi, 100)        # useful to evaluate function at lots of points
# f = np.sin(x)


