#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
import math as m
from time import sleep
from random import uniform, randint
import numpy as np
from threading import Thread
import tf2_ros
import random
import math as m


class pole_env(Thread):
    observation_space_shape = 9
    action_space_shape = 1
    action_space_high = 16
    action_space_low = -16


    def __init__(self):
        rospy.init_node('pole_env')
        self.pub1 = rospy.Publisher('/pole/joint1_effort_controller/command', Float64, queue_size=10)
        self.pub2 = rospy.Publisher('/pole/joint2_effort_controller/command', Float64, queue_size=10)

        self.rate = rospy.Rate(50) # 50hz future
        self.value = 0
        self.is_stop = False
        self.torque = [0, 0]

        rospy.Subscriber("end_effector", Point, self.callback1)
        rospy.Subscriber("/pole/joint_states", JointState, self.callback2)




        Thread.__init__(self)


    def reset(self):  
        self.step_num = 1      
        self.set_target_point()
        self.old_dist = self.distance()

        return self.get_obs()

    def step(self, action):
        self.set_torque(action)
        reward = self.reward_fun()
        obs = self.get_obs()
        done = False
        self.step_num += 1
        if self.step_num > 250:
            done = True

        return obs, reward, done

    def set_target_point(self):
        fi1 = random.uniform(0, 1.57)
        fi2 = random.uniform(0, 6.28)
        ro = 1.15
        self.target_x = 0#ro * m.sin(fi1) * m.cos(fi2)
        self.target_y = 0#ro * m.sin(fi1) * m.sin(fi2)
        self.target_z = 1.25 #0.1 + ro * m.cos(fi1)

        self.target_xyz = [self.target_x, self.target_y, self.target_z]

    def reward_fun(self):
        new_dist = self.distance()
        reward = self.old_dist - new_dist
        self.old_dist = new_dist
        if self.distance() < 0.05:
            reward += 0.25
        return reward

    def distance(self):
        x1 = self.target_x
        x2 = self.xyz_end[0]
        y1 = self.target_y
        y2 = self.xyz_end[1]
        z1 = self.target_z
        z2 = self.xyz_end[2]
        dist = m.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

        return dist



    def run(self):
        try:
            self.talker()
        except rospy.ROSInterruptException:
                pass

    def talker(self):    



        while not rospy.is_shutdown():                      
            self.pub1.publish(self.torque[0])
            self.pub2.publish(self.torque[1])





            if self.is_stop:
            	break
            self.rate.sleep()


    def callback1(self, point):
    	self.xyz_end = [point.x, point.y, point.z]
    	

    def callback2(self, joint):
        self.pos = list(joint.position)
    	self.vel = list(joint.velocity)
        self.eff = list(joint.effort)
        
    

    def stop_thread(self):
        self.is_stop = True

    def set_torque(self, torque_list):
    	self.torque = [torque_list[0], torque_list[1]]

    def get_obs(self):
        obs = self.xyz_end + self.pos + self.vel + self.eff
        obs = np.array(obs)

        return obs







if __name__ == '__main__':
    env = pole_env()
    env.start()
    env.set_torque([0, -15])
    sleep(1)
    print env.reset()
    while 1:        
        action = [random.uniform(-16,16), random.uniform(-16,16)]
        obs, reward, done = env.step(action) 
        print obs, reward 
        if done:
            break      
        env.rate.sleep()



    
    env.stop_thread()





