#!/usr/bin/env python3

from actionlib import SimpleActionServer
import control_msgs.msg as control_msgs
from controller_manager_msgs.srv import *
import cv_bridge
from franka_msgs.msg import FrankaState, ErrorRecoveryAction
from franka_gripper.msg import *

from pb_ros.topics import joint_state_ur5, ArmCommand_ur5, actionserver_ur5
from pb_ros.ros_wrapper import UR5CameraPlugin
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
from scipy import interpolate
from std_msgs.msg import Header
from threading import Thread
import ipdb

from active_grasp.bbox import to_bbox_msg
from active_grasp.srv import *
from active_grasp.simulation2 import Simulation2
from robot_helpers.ros.conversions import *
from vgn.simulation import apply_noise


alpha = 0.5

class BtSimNode2:
    def __init__(self):
        gui = True #rospy.get_param("~gui")
        print(f"gui = {gui}")
        scene_id = rospy.get_param("~scene")
        vgn_path = rospy.get_param("vgn/model")

        ur5_path = rospy.get_param("bt_sim/ur5_path")

        self.sim = Simulation2(gui, scene_id, vgn_path, ur5_path)
        # ipdb.set_trace()
        
        print(f"ur5_path = {ur5_path}")

        

        print("created pybullet sim WITHOUT PLUGINS")
        self.init_plugins()
        self.advertise_services()

    def init_plugins(self):
        self.plugins = [
            PhysicsPlugin(self.sim),
            CameraPlugin(self.sim.camera),
            UR5RobotStatePlugin(self.sim.ur5.bullet_obj),
            # UR5CameraPlugin(self.sim.ur5.plugin),
        ]
        self.controllers = {
            "position_joint_trajectory_controller": UR5JointTrajectoryControllerPlugin(
                self.sim.ur5.bullet_obj.robot
            ),
            "cartesian_velocity_controller": UR5CartesianVelocityControllerPlugin(
                self.sim.ur5.bullet_obj.robot, self.sim.model
            ),
        }

    def start_plugins(self):
        for plugin in self.plugins + list(self.controllers.values()):
            plugin.thread.start()

    def activate_plugins(self):
        # Original code
        for plugin in self.plugins:
        # my change
        # for plugin in self.plugins + list(self.controllers.values()):
            plugin.activate()

    def deactivate_plugins(self):
        for plugin in self.plugins:
            plugin.deactivate()

    def deactivate_controllers(self):
        for controller in self.controllers.values():
            controller.deactivate()

    def advertise_services(self):
        rospy.Service("seed", Seed, self.seed)
        rospy.Service("reset", Reset, self.reset)
        rospy.Service(
            "/controller_manager/switch_controller",
            SwitchController,
            self.switch_controller,
        )

    def seed(self, req):
        self.sim.seed(req.seed)
        rospy.loginfo(f"Seeded the rng with {req.seed}.")
        return SeedResponse()

    def reset(self, req):
        self.deactivate_plugins()
        self.deactivate_controllers()
        rospy.sleep(1.0)  # TODO replace with a read-write lock
        bbox = self.sim.reset()
        self.activate_plugins()
        return ResetResponse(to_bbox_msg(bbox))

    def switch_controller(self, req):
        for controller in req.stop_controllers:
            self.controllers[controller].deactivate()
        for controller in req.start_controllers:
            self.controllers[controller].activate()
        return SwitchControllerResponse(ok=True)

    def run(self):
        self.start_plugins()
        print("started plugins")
        self.activate_plugins()
        print("activated plugins")

        # ipdb.set_trace()

        # self.sim.ur5.main_loop()
        # self.sim.ur5.bullet_obj.robot.step()
        # ipdb.set_trace()
        # self.sim.ur5.plugin.activate()
        rospy.spin()
        print("finished rospy spin")
       


class Plugin:
    """A plugin that spins at a constant rate in its own thread."""

    def __init__(self, rate):
        self.rate = rate
        self.thread = Thread(target=self.loop, daemon=True)
        self.is_running = False

    def activate(self):
        self.is_running = True

    def deactivate(self):
        self.is_running = False

    def loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.is_running:
                self.update()
            rate.sleep()

    def update(self):
        raise NotImplementedError


class PhysicsPlugin(Plugin):
    def __init__(self, sim):
        super().__init__(sim.rate)
        self.sim = sim

    def update(self):
        self.sim.step()


class UR5RobotStatePlugin(Plugin):
    
    def __init__(self, rosconvert_object, rate=30):
        super().__init__(rate)

        self._pub_jointstate = rospy.Publisher(
            "joint_states", JointState, queue_size=10
        )
        self.obj = rosconvert_object
        # self._pub_jointstate = rospy.Publisher("joint_states",JointState,
        #                                         latch=True,queue_size=1)
    
    def update(self):
        # ipdb.set_trace()
        val = self.obj.getJointState()
        self._pub_jointstate.publish(val)



class CartesianVelocityControllerPlugin(Plugin):
    def __init__(self, arm, model, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.model = model
        topic = rospy.get_param("cartesian_velocity_controller/topic")
        rospy.Subscriber(topic, Twist, self.target_cb)

    def target_cb(self, msg):
        self.dx_d = from_twist_msg(msg)

    def activate(self):
        self.dx_d = np.zeros(6)
        self.is_running = True

    def deactivate(self):
        self.dx_d = np.zeros(6)
        self.is_running = False
        self.arm.set_desired_joint_velocities(np.zeros(7))

    def update(self):
        q, _ = self.arm.get_state()
        J_pinv = np.linalg.pinv(self.model.jacobian(q))
        cmd = np.dot(J_pinv, self.dx_d)
        self.arm.set_desired_joint_velocities(cmd)

class JointTrajectoryControllerPlugin(Plugin):
    def __init__(self, arm, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.dt = 1.0 / self.rate  # TODO this might not be reliable
        self.init_action_server()

    def init_action_server(self):
        name = "position_joint_trajectory_controller/follow_joint_trajectory"
        self.action_server = SimpleActionServer(
            name, control_msgs.FollowJointTrajectoryAction, auto_start=False
        )
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        goal = self.action_server.accept_new_goal()
        self.interpolate_trajectory(goal.trajectory.points)
        self.elapsed_time = 0.0

    def interpolate_trajectory(self, points):
        t, y = np.zeros(len(points)), np.zeros((7, len(points)))
        for i, point in enumerate(points):
            t[i] = point.time_from_start.to_sec()
            y[:, i] = point.positions
        self.m = interpolate.interp1d(t, y)
        self.duration = t[-1]

    def update(self):
        if self.action_server.is_active():
            self.elapsed_time += self.dt
            if self.elapsed_time > self.duration:
                self.action_server.set_succeeded()
                return
            self.arm.set_desired_joint_positions(self.m(self.elapsed_time))

class UR5CartesianVelocityControllerPlugin(Plugin):
    def __init__(self, arm, model, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.model = model
        topic = rospy.get_param("cartesian_velocity_controller/topic")
        rospy.Subscriber(topic, Twist, self.target_cb)

    def target_cb(self, msg):
        self.dx_d = from_twist_msg(msg)

    def activate(self):
        self.dx_d = np.zeros(6)
        self.is_running = True

    def deactivate(self):
        self.dx_d = np.zeros(6)
        self.is_running = False
        self.arm.allJointMotorControl(JointVels=np.zeros(6), controllerType='velocity')

    def update(self):
        q = self.arm.getJointStates()[1]
        # ipdb.set_trace()
        # a,b = q[0], q[2]
        # q[0], q[2] = b,a
        print(f'joint states read by controller to pybullet = {q}')
        J_pinv = np.linalg.pinv(self.model.jacobian(q))
        # ipdb.set_trace()
        cmd = np.dot(J_pinv, self.dx_d)
        print(f'velocity commands = {cmd}')
        self.arm.allJointMotorControl(JointVels=cmd, controllerType='velocity')

class UR5JointTrajectoryControllerPlugin(Plugin):
    def __init__(self, arm, rate=30):
        super().__init__(rate)
        self.arm = arm
        self.dt = 1.0 / self.rate  # TODO this might not be reliable
        self.init_action_server()

    def init_action_server(self):
        name = "ur5_arm_controller/follow_joint_trajectory"
        self.action_server = SimpleActionServer(
            name, control_msgs.FollowJointTrajectoryAction, auto_start=False
        )
        # ipdb.set_trace()
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()
#####################################################
        # self.duration = 100

    def action_goal_cb(self):
        goal = self.action_server.accept_new_goal()
        # print(f"goal={goal}")
        self.interpolate_trajectory(goal.trajectory.points)
        self.elapsed_time = 0.0

    def interpolate_trajectory(self, points):
        t, y = np.zeros(len(points)), np.zeros((6, len(points)))
        for i, point in enumerate(points):
            t[i] = point.time_from_start.to_sec()
            y[:, i] = point.positions
        self.m = interpolate.interp1d(t, y)
        self.duration = t[-1]

    def update(self):
        # ipdb.set_trace()
        # print("came to controller update")
        if self.action_server.is_active():
            self.elapsed_time += self.dt
            if self.elapsed_time > self.duration:
                self.action_server.set_succeeded()
                return
            # self.arm.set_desired_joint_positions_ur5(self.m(self.elapsed_time))
            self.arm.allJointMotorControl(self.m(self.elapsed_time))


class MoveActionPlugin(Plugin):
    def __init__(self, gripper, rate=10):
        super().__init__(rate)
        self.gripper = gripper
        self.dt = 1.0 / self.rate
        self.init_action_server()

    def init_action_server(self):
        name = "/franka_gripper/move"
        self.action_server = SimpleActionServer(name, MoveAction, auto_start=False)
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        self.elapsed_time = 0.0
        goal = self.action_server.accept_new_goal()
        self.gripper.set_desired_width(goal.width)

    def update(self):
        if self.action_server.is_active():
            self.elapsed_time += self.dt
            if self.elapsed_time > 1.0:
                self.action_server.set_succeeded()


class GraspActionPlugin(Plugin):
    def __init__(self, gripper, rate=10):
        super().__init__(rate)
        self.gripper = gripper
        self.dt = 1.0 / self.rate
        self.force = rospy.get_param("~gripper_force")
        self.init_action_server()

    def init_action_server(self):
        name = "/franka_gripper/grasp"
        self.action_server = SimpleActionServer(name, GraspAction, auto_start=False)
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        self.elapsed_time = 0.0
        goal = self.action_server.accept_new_goal()
        self.gripper.set_desired_speed(-0.1, force=self.force)

    def update(self):
        if self.action_server.is_active():
            self.elapsed_time += self.dt
            if self.elapsed_time > 1.0:
                self.action_server.set_succeeded()


class GripperActionPlugin(Plugin):
    """Empty action server to make MoveIt happy"""

    def __init__(self, rate=1):
        super().__init__(rate)
        self.init_action_server()

    def init_action_server(self):
        name = "/franka_gripper/gripper_action"
        self.action_server = SimpleActionServer(
            name, control_msgs.GripperCommandAction, auto_start=False
        )
        self.action_server.register_goal_callback(self.action_goal_cb)
        self.action_server.start()

    def action_goal_cb(self):
        self.action_server.accept_new_goal()

    def update(self):
        if self.action_server.is_active():
            self.action_server.set_succeeded()


class CameraPlugin(Plugin):
    def __init__(self, camera, name="camera", rate=5):
        super().__init__(rate)
        self.camera = camera
        self.name = name
        self.cam_noise = rospy.get_param("~cam_noise", False)
        self.cv_bridge = cv_bridge.CvBridge()
        self.init_publishers()

    def init_publishers(self):
        topic = self.name + "/depth/camera_info"
        self.info_pub = rospy.Publisher(topic, CameraInfo, queue_size=10)
        topic = self.name + "/depth/image_rect_raw"
        self.depth_pub = rospy.Publisher(topic, Image, queue_size=10)
        topic = self.name + "/color/image_raw"
        self.color_pub = rospy.Publisher(topic, Image, queue_size=10)

    def update(self):
        stamp = rospy.Time.now()
        # ipdb.set_trace()
        msg = to_camera_info_msg(self.camera.intrinsic)
        # msg.header.frame_id = "ur5_" + self.name + "_depth_optical_frame"
        msg.header.frame_id = "base_link"
        msg.header.stamp = stamp
        self.info_pub.publish(msg)

        color , depth, _ = self.camera.get_image()

        if self.cam_noise:
            depth = apply_noise(depth)

        msg = self.cv_bridge.cv2_to_imgmsg((1000 * depth).astype(np.uint16))
        msg.header.stamp = stamp
        msg.header.frame_id = "ur5_" + self.name + "_depth_optical_frame"
        # msg.header.frame_id = "base_link"
        self.depth_pub.publish(msg)

        msg = self.cv_bridge.cv2_to_imgmsg(color)
        msg.header.stamp = stamp
        # msg.header.frame_id = "ur5_" + self.name + "_depth_optical_frame"
        msg.header.frame_id = "base_link"
        self.color_pub.publish(msg)





def main():
    rospy.init_node("bt_sim")
    
    print("starting pybullet sim")
    server = BtSimNode2()
    print("created pybullet sim")
    server.run()
    print("successfully running server")


if __name__ == "__main__":
    main()
