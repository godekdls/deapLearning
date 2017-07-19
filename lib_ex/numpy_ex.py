# download numpy
# sudo python3 -m pip install numpy
import numpy as np

x = np.array([1.0, 1.0, 111.0])
print(x) # [   1.    1.  111.]
x = np.arange(1, 4, 1) # 0~(4-1) at intervals of 1 -> [1 2 3]
print(x)

# broadcast
y = np.array([2.0, 4.0, 6.0])
print(x*y) # [  2.   8.  18.]

# n dimensional array
A = np.array([[1,2], [3,4]])
print(A) # [[1 2]
         # [3 4]]
print(A.shape) # (2, 2)
print(A*10) # [[10 20]
            # [30 40]]
print(A[0]) # [1,2]
print(A[0][1]) # [2]

B = A.flatten()
print(B) # [1 2 3 4]
print(B>2) # [False False  True  True]


