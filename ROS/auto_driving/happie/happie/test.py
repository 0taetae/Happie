import numpy as np
import math 

goal =  (-50.032000000000004, -49.797999999999995)
pose = (-50.002, -49.768)

dist = math.sqrt((goal[0] - pose[0]) ** 2 + (goal[0] - pose[1]) ** 2)
print(dist)