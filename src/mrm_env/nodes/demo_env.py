#!/usr/bin/env python

import rospy
import dynamic_reconfigure.client
import math as m
from time import sleep
from random import uniform


class demo_env(object):
    def __init__(self):
        rospy.init_node('demo_env')
        self.client = dynamic_reconfigure.client.Client("DynReconfServer")
        self.rate = rospy.Rate(1)
    '''
    def callback(self, config):
        rospy.loginfo("Config set to {joint1}, {joint2}, {joint3}, {joint4}, {coord_x}, {coord_y},\
         {coord_z}, {is_change}".format(**config))
    '''
    def set_joint(self, joint_valve):
        self.client.update_configuration({"joint1": joint_valve[0], "joint2": joint_valve[1], "joint3": joint_valve[2],
                                          "joint4": joint_valve[3], "is_change": False})

    def get_coord(self):
        config = self.client.get_configuration(timeout=30)
        #if config['is_change']:
        x = config['coord_x']
        y = config['coord_y']
        z = config['coord_z']
        control = config['is_change']
        return [x, y, z], control
'''
else:
    result = self.get_coord()
    return result
'''

if __name__ == '__main__':
    env = demo_env()
    while 1:
        q1 = uniform(-3.14, 3.14)
        q2 = uniform(-1.57, 1.57)
        q3 = uniform(-1.57, 1.57)
        q4 = uniform(0, 1.57)
        env.set_joint([q1, q2, q3, q4])
        env.rate.sleep()
        obs, flag = env.get_coord()
        print obs, flag
        sleep(10)









