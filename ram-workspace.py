from math import sqrt
import sys
from pymavlink.dialects.v20 import common as mavlink
from pymavlink import mavutil


def main():
    # This is just a dummy connection, the actual script is later on
    connection = mavutil.mavlink_connection("udpin:localhost:8080")

    connection.wait_heartbeat()

    while True:

        msg = connection.recv_msg(type="OBSTACLE_DISTANCE_3D", blocking=True)

        if not msg:
            continue

        if msg.get_type() == "BAD_DATA":
            if mavutil.all_printable(msg.data):
                sys.stdout.write(msg.data)
                sys.stdout.flush()

        # Let's assume we need to move a total of 1 meter to avoid. If the drone doesn't get past the object by that time, then it'll activate this again either way
        AVOIDANCE_DISTANCE = 1

        # We are going to ignore the x value, because we only want it to move up, down, left or right

        y = msg.y
        z = msg.z

        # Invert y and z values to find out where the drone should move
        y = -y
        z = -z

        # Pythagorian theorem
        length = sqrt(y * y + z * y)

        # Normalize the y and z values
        y /= length
        z /= length

        # Multiply it by AVOIDANCE_DISTANCE to find final value for y and z
        y *= AVOIDANCE_DISTANCE
        z *= AVOIDANCE_DISTANCE

        # Send Correction Message

        connection.mav.set_position_target_local_ned_send(
            0,
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            3576,
            0,
            y,
            z,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )


if __name__ == "__main__":
    main()
