#!/usr/bin/env python

import rospy
import dynamic_reconfigure.client
import math as m
from time import sleep
from random import uniform, randint
import numpy as np


class env(object):
    OBSERVATION_SPACE_VALUES = (9,)
    ACTION_SPACE_SIZE = 12

    #action_space = None
    #observation_space = None

    def __init__(self):
        rospy.init_node('environment')
        self.client = dynamic_reconfigure.client.Client("DynReconfServer")
                
        self.num_steps = 1

        self.d_q = m.radians(2)

        self.end_coord_x = None
        self.end_coord_y = None
        self.end_coord_z = None
        self.end_coord = [self.end_coord_x, self.end_coord_y, self.end_coord_z]

        self.ball_coord = []
        self.ball_coord_x = None
        self.ball_coord_y = None
        self.ball_coord_z = None

        self.dist = None
        self.start_dist = None


    
    def step(self, action):

        #while action == [0, 0, 0, 0]:
            #action = [randint(0, 2), randint(0, 2), randint(0, 2), randint(0, 2)]

        self.distance()
        self.start_dist = self.dist
        q1 =self.q1
        q2 =self.q2
        q3 =self.q3
        q4 =self.q4
        if action[0] == 2:
            self.q1 += self.d_q
            if self.q1 > 3.14:
                self.q1 = 3.14
                
        elif action[0] == 1:
            self.q1 -= self.d_q
            if self.q1 < -3.14:
                self.q1 = -3.14
               

        if action[1] == 2:
            self.q2 += self.d_q
            if self.q2 > 1.57:
                self.q2 = 1.57
                
        elif action[1] == 1:
            self.q2 -= self.d_q
            if self.q2 < -1.57:
                self.q2 = -1.57
                
        if action[2] == 2:
            self.q3 += self.d_q
            if self.q3 > 1.57:
                self.q3 = 1.57
                
        elif action[2] == 1:
            self.q3 -= self.d_q
            if self.q3 < 0:
                self.q3 = 0
                
        if action[3] == 2:
            self.q4 += self.d_q
            if self.q4 > 1.57:
                self.q4 = 1.57
                
        elif action[3] == 1:
            self.q4 -= self.d_q
            if self.q4 < 0:
                self.q4 = 0
                

        if not (q1 == self.q1 and q2 == self.q2 and q3 == self.q3 and q4 == self.q4):
            self.set_joint_valve()
            self.get_end_coord() 


        obs = self.get_naiv_obs()
        

        done = False

        self.num_steps += 1

        if self.dist < 0.1:
            done = True

        if self.num_steps > 70:
            done = True

        reward = self.reward_funct(done)        

        return obs, reward, done

    
    def reset(self):

        self.num_steps = 1        
        
        self.set_random_joint_valve()
        self.get_end_coord()
        self.set_random_target_point()
        return self.get_naiv_obs()

    def reward_funct(self, done):
        self.distance()
        reward = 10 * (self.start_dist - self.dist)
        if done:
            reward += (70 - self.num_steps) * 0.2
            
        if self.dist < 0.05:
            reward += 4
        return reward

    
    def set_random_target_point(self):
            ro = uniform(0.35, 1.35)
            teta = uniform(0, m.pi / 2)
            fi = uniform(-m.pi/2, m.pi/2)
            self.ball_coord_x = ro * m.cos(fi) #* m.sin(teta) * m.cos(fi)
            self.ball_coord_y = ro * m.sin(fi) #* m.sin(teta) * m.sin(fi)
            self.ball_coord_z = 0.2 #0.27601 + ro * m.cos(teta)

    def set_random_joint_valve(self):
        q1 = 0#uniform(-1.57, 1.57)
        q2 = 0#uniform(0, 1.57)
        q3 = 0#uniform(0, 1.57)
        q4 = 0#uniform(0, 1.57)
        self.set_joint([q1, q2, q3, q4])

    def set_joint_valve(self):        
        self.set_joint([self.q1, self.q2, self.q3, self.q4])


    def set_target_point(self):
        self.ball_coord = self.target_position
        self.ball_coord_x = self.ball_coord[0]
        self.ball_coord_y = self.ball_coord[1]
        self.ball_coord_z = self.ball_coord[2]  

    def distance(self):
        x1 = self.end_coord_x
        y1 = self.end_coord_y
        z1 = self.end_coord_z
        x2 = self.ball_coord_x
        y2 = self.ball_coord_y
        z2 = self.ball_coord_z

        self.dist = m.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    def get_naiv_obs(self):
        #self.distance()        

        result = np.array([self.q1, self.q2, self.q3, self.q4,
         self.end_coord_x, self.end_coord_y, self.end_coord_z, self.ball_coord_x, self.ball_coord_y])

        return result

    def get_obs(self):
        a = self.get_naiv_obs()     
        b = np.array([3.14, 1.57, 1.57, 1.57, 1.35, 2.7, 1.625, 1.35, 2.7])
        c = np.array([1.57, 0, 0, 0, 0, 1.35, 0, 1.35, 0])
        result = (a+c)/b
        return result



    def set_joint(self, joint_valve):
        self.q1 = joint_valve[0]
        self.q2 = joint_valve[1]
        self.q3 = joint_valve[2]
        self.q4 = joint_valve[3]
        self.client.update_configuration({"joint1": joint_valve[0], "joint2": joint_valve[1],
         "joint3": joint_valve[2], "joint4": joint_valve[3], "is_change": False})

    def get_end_coord(self):
        config = self.client.get_configuration()
        while not config['is_change']:
            config = self.client.get_configuration()
        self.end_coord_x = config['coord_x']
        self.end_coord_y = config['coord_y']
        self.end_coord_z = config['coord_z']
        self.end_coord = [self.end_coord_x, self.end_coord_y, self.end_coord_z]



if __name__ == '__main__':
    env = env()
    env.set_joint([0, 1, 1, 1])
    print env.get_naiv_obs()



    




    
    rospy.spin()
'''

 '''   
