#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from mrm_env.cfg import DynReconfConfig

def callback(config, level):
    #rospy.loginfo("""Reconfigure Request: {joint1}, {joint2}, {joint3}, {joint4}, {coord_x}, {coord_y}, {coord_z}, {is_change}""".format(**config))
    return config

if __name__ == "__main__":
    rospy.init_node("DynReconfServer", anonymous=False)

    srv = Server(DynReconfConfig, callback)
    rospy.spin()
