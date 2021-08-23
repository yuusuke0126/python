import inspect
import copy
from geometry_msgs.msg import PoseStamped, Pose, Point, Twist

import rospy
import time

class Myclass():
    def __init__(self):
        self.a = 1
        self.b = 'str'
        self.pose = PoseStamped()
        self.pose.header.stamp = rospy.Time.from_sec(time.time())
    pass

def obj2dict(obj0):
    if '__dict__' in dir(obj0) or '.msg' in str(type(obj0)):
        obj = copy.copy(obj0)
        if '__dict__' in dir(obj):
            d = obj.__dict__
        else:
            attributes = inspect.getmembers(obj)
            attributes = [x for x in attributes if not callable(x[1])]
            d = dict([x for x in attributes if not x[0][0].startswith('_')])
        for i in d:
            if '__dict__' in dir(d[i]) or '.msg' in str(type(d[i])):
                d[i] = obj2dict(d[i])
        return d
    else:
        print('error: object cannot convert to dictionary.')
        return obj0

a = Myclass()
b = obj2dict(a)
print(b)