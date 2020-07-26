#!/usr/bin/env python

import rospy
import tf
import roslib
from time import sleep
import geometry_msgs.msg




def main():
    state = [0, 0, 0]
    rospy.init_node('coordinate_reader')
    listener = tf.TransformListener()
    pub = rospy.Publisher('end_effector', geometry_msgs.msg.Point, queue_size=1)
    rate = rospy.Rate(50.0)    
    while not rospy.is_shutdown():
        try:                      
            (trans, rot) = listener.lookupTransform('world', 'end_link', rospy.Time(0))       
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):                        
            continue        

        cmd = geometry_msgs.msg.Point()       
        cmd.x = trans[0]
        cmd.y = trans[1]
        cmd.z = trans[2]

        pub.publish(cmd)
        rate.sleep()



if __name__ == '__main__':
    main()