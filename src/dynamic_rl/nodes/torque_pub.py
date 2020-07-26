#!/usr/bin/env python

import rospy
from std_msgs import Float64

def talker():
    pub = rospy.Publisher('/pole/joint2_effort_controller/command', Float64, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():        
        pub.publish(12)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

