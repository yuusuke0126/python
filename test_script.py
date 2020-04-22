import numpy as np

a = range(20)
b = range(10)
len_my_path = len(a)
len_robots_path = len(b)

c = np.zeros((2, len_my_path))

for i in range(len_my_path):
  try:
    c[0,i] = b[i]
  except IndexError as e:
    print(e)
    c[1,i] = b[-1]

print(a,b)
print(c)