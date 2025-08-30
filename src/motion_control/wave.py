#!/usr/bin/env python3
"""
Unitree G1 Robot Waving Demo

This script demonstrates smooth motion control for the Unitree G1 humanoid robot.
The robot performs a waving sequence with smooth transitions between poses.

Features:
- Reads current robot joint positions for smooth startup
- Modular motion functions for easy customization
- Joint-specific stiffness values for optimal control
- Smooth oscillation for natural waving motion
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
    # Right arm (22-28): relaxed at side (will be modified for waving)
    0.3, 0.0, 0.0, -0.5, 0.0, 0.0, 0.08
], dtype=float)

# Waving pose: right arm raised and positioned for waving
wave_up_pose = standing_pose.copy()
wave_up_pose[G1JointIndex.RightShoulderPitch] = -2.0  # Raise arm up
wave_up_pose[G1JointIndex.RightShoulderRoll] = -1.0   # Move arm away from body
wave_up_pose[G1JointIndex.RightElbow] = 1.0           # Bend elbow for wave

# Extended wave pose: arm more extended for oscillation
wave_extended_pose = wave_up_pose.copy()
wave_extended_pose[G1JointIndex.RightElbow] = -0.3    # Extend arm outward

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
    """Execute the waving sequence"""
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

    print("🤖 Starting G1 waving sequence...")
    
    # execution step
    t_execution = 2.0

    # Execute waving sequence
    smooth_transition_to_pose(standing_pose, t_execution, pub, cmd, "Moving to standing position")
    smooth_transition_to_pose(wave_up_pose, t_execution, pub, cmd, "Raising right arm")
    smooth_transition_to_pose(wave_extended_pose, t_execution, pub, cmd, "Waving")
    smooth_transition_to_pose(wave_up_pose, t_execution, pub, cmd, "Waving")
    smooth_transition_to_pose(standing_pose, t_execution, pub, cmd, "Returning to standing")
    hold_pose(standing_pose, t_execution, pub, cmd, "Final standing pose")
    
    print("✅ Wave sequence completed successfully!")

if __name__ == '__main__':
    input("Press Enter to start waving sequence...")
    main()