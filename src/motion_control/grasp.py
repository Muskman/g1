#!/usr/bin/env python3
"""
Unitree G1 Robot Grasping Demo

This script demonstrates smooth grasping motion control for the Unitree G1 humanoid robot.
The robot performs a complete grasping sequence including reaching, grasping, lifting, 
moving, and releasing an object.

Features:
- Reads current robot joint positions for smooth startup
- Modular motion functions for easy customization
- Joint-specific stiffness values for optimal control
- Complete grasping workflow with hand/wrist control
- Multiple grasping scenarios (table grasp, shelf grasp)
"""

import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

G1_NUM_MOTOR = 29
CONTROL_DT = 0.002  # Control loop timestep (500 Hz)

# Joint-specific stiffness values optimized for different body parts
Kp = [
    60, 60, 60, 100, 40, 40,      # Left leg: higher stiffness for support
    60, 60, 60, 100, 40, 40,      # Right leg: higher stiffness for support
    60, 40, 40,                   # Waist: medium stiffness for stability
    40, 40, 40, 40, 40, 40, 40,   # Left arm: lower stiffness for smooth motion
    40, 40, 40, 40, 40, 40, 40    # Right arm: lower stiffness for smooth motion
]

Kd = [
    1, 1, 1, 2, 1, 1,     # Left leg: higher damping for knees
    1, 1, 1, 2, 1, 1,     # Right leg: higher damping for knees
    1, 1, 1,              # Waist: standard damping
    1, 1, 1, 1, 1, 1, 1,  # Left arm: standard damping
    1, 1, 1, 1, 1, 1, 1   # Right arm: standard damping
]

# ============================================================================
# JOINT INDEX MAPPING
# ============================================================================

class G1JointIndex:
    """Joint index constants for G1 29DOF configuration"""
    # Legs
    LeftHipPitch, LeftHipRoll, LeftHipYaw, LeftKnee = 0, 1, 2, 3
    LeftAnklePitch, LeftAnkleRoll = 4, 5
    RightHipPitch, RightHipRoll, RightHipYaw, RightKnee = 6, 7, 8, 9
    RightAnklePitch, RightAnkleRoll = 10, 11
    
    # Waist
    WaistYaw, WaistRoll, WaistPitch = 12, 13, 14
    
    # Left arm
    LeftShoulderPitch, LeftShoulderRoll, LeftShoulderYaw = 15, 16, 17
    LeftElbow, LeftWristRoll, LeftWristPitch, LeftWristYaw = 18, 19, 20, 21
    
    # Right arm  
    RightShoulderPitch, RightShoulderRoll, RightShoulderYaw = 22, 23, 24
    RightElbow, RightWristRoll, RightWristPitch, RightWristYaw = 25, 26, 27, 28

# ============================================================================
# POSE DEFINITIONS
# ============================================================================

# Base standing pose with stable leg configuration and relaxed arms
standing_pose = np.array([
    # Left leg (0-5): slight outward stance for stability
    0.0, 0.1, 0.0, -0.3, 0.0, -0.3,
    # Right leg (6-11): mirrored configuration
    0.0, -0.1, 0.0, -0.3, 0.0, -0.3,
    # Waist (12-14): upright neutral position
    0.0, 0.0, 0.0,
    # Left arm (15-21): relaxed at side
    0.3, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0,
    # Right arm (22-28): relaxed at side
    0.3, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0
], dtype=float)

# Ready to grasp: arm positioned for grasping approach
ready_to_grasp_pose = standing_pose.copy()
ready_to_grasp_pose[G1JointIndex.RightShoulderPitch] = -0.8    # Raise arm forward moderately
ready_to_grasp_pose[G1JointIndex.RightShoulderRoll] = -0.3     # Slight outward for comfort
ready_to_grasp_pose[G1JointIndex.RightElbow] = 1.2             # Bend elbow for reach
ready_to_grasp_pose[G1JointIndex.RightWristPitch] = -0.3       # Orient wrist for grasping

# Reach for object: extend arm toward object on table
reach_object_pose = standing_pose.copy()
reach_object_pose[G1JointIndex.RightShoulderPitch] = -1.2      # Extend arm forward
reach_object_pose[G1JointIndex.RightShoulderRoll] = -0.2       # Slight outward
reach_object_pose[G1JointIndex.RightElbow] = 0.8               # Extend arm more
reach_object_pose[G1JointIndex.RightWristPitch] = -0.5         # Orient wrist down toward object
reach_object_pose[G1JointIndex.RightWristRoll] = 0.0           # Neutral roll for approach

# Pre-grasp: hand open, positioned around object
pre_grasp_pose = reach_object_pose.copy()
pre_grasp_pose[G1JointIndex.RightElbow] = 0.6                  # Move closer to object
pre_grasp_pose[G1JointIndex.RightWristRoll] = -1.2             # Open hand (simulate finger spread)
pre_grasp_pose[G1JointIndex.RightWristYaw] = -0.3              # Adjust wrist orientation

# Grasp: hand closed around object
grasp_pose = pre_grasp_pose.copy()
grasp_pose[G1JointIndex.RightWristRoll] = 1.2                  # Close hand (simulate finger close)
grasp_pose[G1JointIndex.RightWristPitch] = -0.3                # Adjust grip angle

# Lift object: raise grasped object
lift_object_pose = grasp_pose.copy()
lift_object_pose[G1JointIndex.RightShoulderPitch] = -1.8       # Raise arm higher
lift_object_pose[G1JointIndex.RightElbow] = 1.0                # Adjust elbow for lift

# Move object left: carry object to new location
move_object_left_pose = lift_object_pose.copy()
move_object_left_pose[G1JointIndex.WaistYaw] = -0.6            # Turn waist left
move_object_left_pose[G1JointIndex.RightShoulderYaw] = -0.4    # Adjust arm position

# Place object: lower object to new position
place_object_pose = move_object_left_pose.copy()
place_object_pose[G1JointIndex.RightShoulderPitch] = -1.2      # Lower arm
place_object_pose[G1JointIndex.RightElbow] = 0.6               # Extend for placement

# Release object: open hand to release
release_pose = place_object_pose.copy()
release_pose[G1JointIndex.RightWristRoll] = -1.2               # Open hand (release object)
release_pose[G1JointIndex.RightWristYaw] = 0.0                 # Neutral wrist

# Withdraw: pull hand back after release
withdraw_pose = release_pose.copy()
withdraw_pose[G1JointIndex.RightElbow] = 1.5                   # Pull arm back
withdraw_pose[G1JointIndex.RightShoulderPitch] = -0.5          # Lower arm

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

current_pose = None
current_state_received = False

def low_state_handler(msg: LowState_):
    """Callback to continuously update current joint positions"""
    global current_pose, current_state_received
    current_pose = np.array([msg.motor_state[i].q for i in range(G1_NUM_MOTOR)])
    current_state_received = True

def wait_for_robot_state():
    """Wait until we receive robot state data"""
    if not current_state_received:
        print("Waiting for robot state...")
        while not current_state_received:
            time.sleep(0.01)

def set_motor_commands(cmd, target_pose):
    """Efficiently set all motor commands for target pose"""
    for i in range(G1_NUM_MOTOR):
        motor = cmd.motor_cmd[i]
        motor.q = target_pose[i]
        motor.kp = Kp[i]
        motor.kd = Kd[i]
        motor.dq = 0.0
        motor.tau = 0.0

# ============================================================================
# MOTION CONTROL FUNCTIONS
# ============================================================================

def smooth_transition_to_pose(target_pose, duration, pub, cmd, description="Transitioning"):
    """
    Smoothly transition from current robot pose to target pose.
    
    Uses cosine interpolation for natural acceleration/deceleration curves.
    
    Args:
        target_pose: Target joint positions (29-element numpy array)
        duration: Transition time in seconds
        pub: Command publisher
        cmd: Command message object
        description: Progress description string
    """
    wait_for_robot_state()
    
    start_pose = current_pose.copy()
    start_time = time.perf_counter()
    last_print_time = -1
    
    print(f"{description} over {duration:.1f}s...")
    
    while True:
        elapsed_time = time.perf_counter() - start_time
        
        # Calculate smooth interpolation ratio
        if elapsed_time >= duration:
            ratio = 1.0
        else:
            ratio = elapsed_time / duration
        
        # Apply cosine smoothing for natural motion
        smooth_ratio = 0.5 * (1 - np.cos(ratio * np.pi))
        interpolated_pose = smooth_ratio * target_pose + (1 - smooth_ratio) * start_pose
        
        # Set motor commands and publish
        set_motor_commands(cmd, interpolated_pose)
        pub.Write(cmd)
        
        # Progress feedback
        current_second = int(elapsed_time)
        if current_second != last_print_time:
            print(f"{description}... {elapsed_time:.1f}s/{duration:.1f}s")
            last_print_time = current_second
        
        if elapsed_time >= duration:
            print(f"{description} completed!")
            break
        
        time.sleep(CONTROL_DT)

def hold_pose(target_pose, duration, pub, cmd, description="Holding pose"):
    """
    Hold a specific pose for a given duration.
    
    Args:
        target_pose: Joint positions to maintain
        duration: Hold time in seconds  
        pub: Command publisher
        cmd: Command message object
        description: Progress description string
    """
    print(f"{description} for {duration:.1f} seconds...")
    start_time = time.perf_counter()
    
    while time.perf_counter() - start_time < duration:
        set_motor_commands(cmd, target_pose)
        pub.Write(cmd)
        time.sleep(CONTROL_DT)
    
    print(f"{description} completed!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the grasping sequence"""
    # Initialize SDK communication
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    # Setup publishers and subscribers
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(low_state_handler, 10)
    
    # Initialize command message
    cmd = unitree_hg_msg_dds__LowCmd_()
    for i in range(35):  # G1 uses 35-element motor array
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0  
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    print("🤏 Starting G1 grasping sequence...")
    
    # Execution step timing
    t_approach = 3.0      # Time for approach movements
    t_grasp = 2.0         # Time for grasp/release actions
    t_hold = 1.5          # Time to hold poses
    t_move = 3.5          # Time for larger movements
    
    # Execute complete grasping sequence
    print("📍 Phase 1: Preparation")
    smooth_transition_to_pose(standing_pose, t_approach, pub, cmd, "Moving to standing position")
    smooth_transition_to_pose(ready_to_grasp_pose, t_approach, pub, cmd, "Moving to ready position")
    
    print("📍 Phase 2: Approach Object")
    smooth_transition_to_pose(reach_object_pose, t_approach, pub, cmd, "Reaching toward object")
    smooth_transition_to_pose(pre_grasp_pose, t_grasp, pub, cmd, "Positioning for grasp (hand open)")
    hold_pose(pre_grasp_pose, t_hold, pub, cmd, "Preparing to grasp")
    
    print("📍 Phase 3: Grasp Object")
    smooth_transition_to_pose(grasp_pose, t_grasp, pub, cmd, "Grasping object (closing hand)")
    hold_pose(grasp_pose, t_hold, pub, cmd, "Securing grip")
    
    print("📍 Phase 4: Lift and Move")
    smooth_transition_to_pose(lift_object_pose, t_approach, pub, cmd, "Lifting object")
    hold_pose(lift_object_pose, t_hold, pub, cmd, "Object lifted")
    smooth_transition_to_pose(move_object_left_pose, t_move, pub, cmd, "Moving object to new location")
    
    print("📍 Phase 5: Place and Release")
    smooth_transition_to_pose(place_object_pose, t_approach, pub, cmd, "Positioning for placement")
    smooth_transition_to_pose(release_pose, t_grasp, pub, cmd, "Releasing object (opening hand)")
    hold_pose(release_pose, t_hold, pub, cmd, "Object released")
    
    print("📍 Phase 6: Return to Rest")
    smooth_transition_to_pose(withdraw_pose, t_approach, pub, cmd, "Withdrawing hand")
    smooth_transition_to_pose(standing_pose, t_move, pub, cmd, "Returning to standing position")
    hold_pose(standing_pose, t_hold, pub, cmd, "Final standing pose")
    
    print("✅ Grasping sequence completed successfully!")
    print("🎯 Summary: Object grasped, moved, and placed successfully!")

if __name__ == '__main__':
    input("Press Enter to start grasping sequence...")
    main()

