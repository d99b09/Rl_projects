#!/usr/bin/env python

import rospy
import tf2_ros
import dynamic_reconfigure.client
from time import sleep



def main():
    state = [0, 0, 0]
    rospy.init_node('coordinate_reader')
    client = dynamic_reconfigure.client.Client("DynReconfServer")
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    i = 0
    #rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        try:                      
            trans = tfBuffer.lookup_transform('base_link', 'final_link', rospy.Time())       
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):                        
            #rate.sleep()
            continue        
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        z = trans.transform.translation.z
        new_state = [x, y, z]
        if new_state==state and i<10000:
            i += 1
            #sleep(0.0001)            
            continue
        else:
            #print 'iiiii:', i
            i = 0
            client.update_configuration({"coord_x":x, "coord_y":y, "coord_z":z, "is_change":True})
            state = [x, y, z]            

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass







