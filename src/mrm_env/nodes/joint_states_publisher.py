#!/usr/bin/env python

import rospy
import dynamic_reconfigure.client
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def get_joint_state(client):    
    config = client.get_configuration()
    q1 = config['joint1']
    q2 = config['joint2']
    q3 = config['joint3']
    q4 = config['joint4']
    return [q1, q2, q3, q4]    

def main():
    rospy.init_node('joint_state_publisher')
    pub = rospy.Publisher('joint_states', JointState, queue_size=1)
    client = dynamic_reconfigure.client.Client("DynReconfServer")
    
    joint = JointState()   

    joint.header = Header()    
    joint.name = ['joint1', 'joint2', 'joint3', 'joint4']
    

    while not rospy.is_shutdown():
      
        joint.header.stamp = rospy.Time.now()
        joint.position = get_joint_state(client) 
        client.update_configuration({"is_change": False}) 
        pub.publish(joint)
        

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
