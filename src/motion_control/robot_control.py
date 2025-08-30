#!/usr/bin/env python3
"""
Unitree G1 Robot Combined Motion Demo

This script demonstrates smooth motion control for the Unitree G1 humanoid robot.
The robot can perform three different motion sequences with either hand based on user selection:
1. Waving sequence (left or right hand)
2. Pointing sequence (left or right hand, with directional options)
3. Grasping sequence (left or right hand)

Features:
- Interactive menu system for motion and hand selection
- Dynamic pose generation for both left and right hands
- Reads current robot joint positions for smooth startup
- Modular motion functions for easy customization
- Joint-specific stiffness values for optimal control
- Complete motion workflows for each behavior type
- Mirrored motions with proper joint mapping for natural left/right hand movements
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

# ============================================================================
# POSE GENERATION UTILITIES
# ============================================================================

def create_pose_for_hand(base_pose, hand, modifications):
    """
    Create a pose for specified hand with given joint modifications.
    
    Args:
        base_pose: Base pose to start from (typically standing_pose)
        hand: 'left' or 'right'
        modifications: Dict of joint modifications relative to base pose
    
    Returns:
        Modified pose array
    """
    pose = base_pose.copy()
    
    # Define joint index mappings for each hand
    if hand == 'right':
        joint_map = {
            'shoulder_pitch': G1JointIndex.RightShoulderPitch,
            'shoulder_roll': G1JointIndex.RightShoulderRoll,
            'shoulder_yaw': G1JointIndex.RightShoulderYaw,
            'elbow': G1JointIndex.RightElbow,
            'wrist_roll': G1JointIndex.RightWristRoll,
            'wrist_pitch': G1JointIndex.RightWristPitch,
            'wrist_yaw': G1JointIndex.RightWristYaw
        }
    else:  # left hand
        joint_map = {
            'shoulder_pitch': G1JointIndex.LeftShoulderPitch,
            'shoulder_roll': G1JointIndex.LeftShoulderRoll,
            'shoulder_yaw': G1JointIndex.LeftShoulderYaw,
            'elbow': G1JointIndex.LeftElbow,
            'wrist_roll': G1JointIndex.LeftWristRoll,
            'wrist_pitch': G1JointIndex.LeftWristPitch,
            'wrist_yaw': G1JointIndex.LeftWristYaw
        }
    
    # Apply modifications
    for joint_name, value in modifications.items():
        if joint_name in joint_map:
            pose[joint_map[joint_name]] = value
        elif joint_name == 'waist_yaw':
            # Special handling for waist rotation
            pose[G1JointIndex.WaistYaw] = value
    
    return pose

# ============================================================================
# MOTION POSE GENERATORS
# ============================================================================

def get_waving_poses(hand):
    """Generate waving poses for specified hand"""
    # Mirror shoulder roll for left hand (positive for left, negative for right)
    shoulder_roll = 1.0 if hand == 'left' else -1.0
    
    wave_up_mods = {
        'shoulder_pitch': -2.0,
        'shoulder_roll': shoulder_roll,
        'elbow': 1.0
    }
    
    wave_extended_mods = {
        'shoulder_pitch': -2.0,
        'shoulder_roll': shoulder_roll,
        'elbow': -0.3
    }
    
    wave_up_pose = create_pose_for_hand(standing_pose, hand, wave_up_mods)
    wave_extended_pose = create_pose_for_hand(standing_pose, hand, wave_extended_mods)
    
    return wave_up_pose, wave_extended_pose

def get_pointing_poses(hand, direction):
    """Generate pointing pose for specified hand and direction"""
    # Mirror values for left hand
    if hand == 'left':
        # For left hand, mirror roll values and adjust yaw/waist rotations
        if direction == 'forward':
            mods = {
                'shoulder_pitch': -1.5,
                'shoulder_roll': 0.1,    # Mirror of -0.1
                'elbow': 1,
                'wrist_roll': 1          # Mirror of -1
            }
        elif direction == 'left':
            mods = {
                'shoulder_pitch': 0,
                'shoulder_roll': 1.5,    # Mirror of -1.5
                'shoulder_yaw': 0,
                'elbow': 1.5,
                'wrist_pitch': 0
            }
        elif direction == 'right':
            mods = {
                'shoulder_pitch': -1.5,
                'shoulder_roll': 0.1,    # Mirror of -0.1
                'elbow': 1.5,
                'wrist_pitch': -0.5,
                'waist_yaw': -1.5        # Mirror of 1.5
            }
        elif direction == 'up':
            mods = {
                'shoulder_pitch': -2.5,
                'shoulder_roll': 0.5,    # Mirror of -0.5
                'elbow': 1,
                'wrist_pitch': 0
            }
    else:  # right hand
        if direction == 'forward':
            mods = {
                'shoulder_pitch': -1.5,
                'shoulder_roll': -0.1,
                'elbow': 1,
                'wrist_roll': -1
            }
        elif direction == 'left':
            mods = {
                'shoulder_pitch': 0,
                'shoulder_roll': -1.5,
                'shoulder_yaw': 0,
                'elbow': 1.5,
                'wrist_pitch': 0
            }
        elif direction == 'right':
            mods = {
                'shoulder_pitch': -1.5,
                'shoulder_roll': -0.1,
                'elbow': 1.5,
                'wrist_pitch': -0.5,
                'waist_yaw': 1.5
            }
        elif direction == 'up':
            mods = {
                'shoulder_pitch': -2.5,
                'shoulder_roll': -0.5,
                'elbow': 1,
                'wrist_pitch': 0
            }
    
    return create_pose_for_hand(standing_pose, hand, mods)

def get_grasping_poses(hand):
    """Generate all grasping poses for specified hand"""
    # Mirror shoulder roll values for left hand
    shoulder_roll_sign = 1 if hand == 'left' else -1
    wrist_roll_open = 1.2 if hand == 'left' else -1.2
    wrist_roll_closed = -1.2 if hand == 'left' else 1.2
    waist_turn = 0.6 if hand == 'left' else -0.6  # Turn toward opposite side
    shoulder_yaw_adjust = 0.4 if hand == 'left' else -0.4
    
    # Ready to grasp
    ready_mods = {
        'shoulder_pitch': -0.8,
        'shoulder_roll': shoulder_roll_sign * 0.3,
        'elbow': 1.2,
        'wrist_pitch': -0.3
    }
    
    # Reach for object
    reach_mods = {
        'shoulder_pitch': -1.2,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'elbow': 0.8,
        'wrist_pitch': -0.5,
        'wrist_roll': 0.0
    }
    
    # Pre-grasp (hand open)
    pre_grasp_mods = {
        'shoulder_pitch': -1.2,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'elbow': 0.6,
        'wrist_pitch': -0.5,
        'wrist_roll': wrist_roll_open,
        'wrist_yaw': shoulder_roll_sign * 0.3
    }
    
    # Grasp (hand closed)
    grasp_mods = {
        'shoulder_pitch': -1.2,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'elbow': 0.6,
        'wrist_pitch': -0.3,
        'wrist_roll': wrist_roll_closed,
        'wrist_yaw': shoulder_roll_sign * 0.3
    }
    
    # Lift object
    lift_mods = {
        'shoulder_pitch': -1.8,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'elbow': 1.0,
        'wrist_pitch': -0.3,
        'wrist_roll': wrist_roll_closed,
        'wrist_yaw': shoulder_roll_sign * 0.3
    }
    
    # Move object to opposite side
    move_mods = {
        'shoulder_pitch': -1.8,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'shoulder_yaw': shoulder_yaw_adjust,
        'elbow': 1.0,
        'wrist_pitch': -0.3,
        'wrist_roll': wrist_roll_closed,
        'wrist_yaw': shoulder_roll_sign * 0.3,
        'waist_yaw': waist_turn
    }
    
    # Place object
    place_mods = {
        'shoulder_pitch': -1.2,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'shoulder_yaw': shoulder_yaw_adjust,
        'elbow': 0.6,
        'wrist_pitch': -0.3,
        'wrist_roll': wrist_roll_closed,
        'wrist_yaw': shoulder_roll_sign * 0.3,
        'waist_yaw': waist_turn
    }
    
    # Release object (hand open)
    release_mods = {
        'shoulder_pitch': -1.2,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'shoulder_yaw': shoulder_yaw_adjust,
        'elbow': 0.6,
        'wrist_pitch': -0.3,
        'wrist_roll': wrist_roll_open,
        'wrist_yaw': 0.0,
        'waist_yaw': waist_turn
    }
    
    # Withdraw hand
    withdraw_mods = {
        'shoulder_pitch': -0.5,
        'shoulder_roll': shoulder_roll_sign * 0.2,
        'elbow': 1.5,
        'wrist_pitch': -0.3,
        'wrist_roll': wrist_roll_open,
        'wrist_yaw': 0.0,
        'waist_yaw': waist_turn
    }
    
    # Generate all poses
    poses = {}
    poses['ready'] = create_pose_for_hand(standing_pose, hand, ready_mods)
    poses['reach'] = create_pose_for_hand(standing_pose, hand, reach_mods)
    poses['pre_grasp'] = create_pose_for_hand(standing_pose, hand, pre_grasp_mods)
    poses['grasp'] = create_pose_for_hand(standing_pose, hand, grasp_mods)
    poses['lift'] = create_pose_for_hand(standing_pose, hand, lift_mods)
    poses['move'] = create_pose_for_hand(standing_pose, hand, move_mods)
    poses['place'] = create_pose_for_hand(standing_pose, hand, place_mods)
    poses['release'] = create_pose_for_hand(standing_pose, hand, release_mods)
    poses['withdraw'] = create_pose_for_hand(standing_pose, hand, withdraw_mods)
    
    return poses

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
# MOTION SEQUENCES
# ============================================================================

def execute_waving_sequence(pub, cmd, hand):
    """Execute the waving sequence with specified hand"""
    print(f"🤖 Starting waving sequence with {hand} hand...")
    
    t_execution = 2.0
    
    # Get poses for specified hand
    wave_up_pose, wave_extended_pose = get_waving_poses(hand)
    
    # Execute waving sequence
    smooth_transition_to_pose(standing_pose, t_execution, pub, cmd, "Moving to standing position")
    smooth_transition_to_pose(wave_up_pose, t_execution, pub, cmd, f"Raising {hand} arm")
    smooth_transition_to_pose(wave_extended_pose, t_execution, pub, cmd, "Waving")
    smooth_transition_to_pose(wave_up_pose, t_execution, pub, cmd, "Waving")
    smooth_transition_to_pose(standing_pose, t_execution, pub, cmd, "Returning to standing")
    hold_pose(standing_pose, t_execution, pub, cmd, "Final standing pose")
    
    print(f"✅ {hand.capitalize()} hand wave sequence completed successfully!")

def execute_pointing_sequence(pub, cmd, hand, direction):
    """Execute the pointing sequence with specified hand and direction"""
    print(f"👉 Starting pointing sequence - {hand} hand pointing {direction}...")
    
    t_execution = 3.0
    t_hold = 2.0
    
    # Get pose for specified hand and direction
    target_pose = get_pointing_poses(hand, direction)
    
    # Execute pointing sequence
    smooth_transition_to_pose(standing_pose, t_execution, pub, cmd, "Moving to standing position")
    smooth_transition_to_pose(target_pose, t_execution, pub, cmd, f"{hand.capitalize()} hand pointing {direction}")
    hold_pose(target_pose, t_hold, pub, cmd, f"Holding {direction} point")
    smooth_transition_to_pose(standing_pose, t_execution, pub, cmd, "Returning to standing")
    hold_pose(standing_pose, t_execution, pub, cmd, "Final standing pose")
    
    print(f"✅ {hand.capitalize()} hand pointing {direction} sequence completed successfully!")

def execute_grasping_sequence(pub, cmd, hand):
    """Execute the grasping sequence with specified hand"""
    print(f"🤏 Starting grasping sequence with {hand} hand...")
    
    # Execution step timing
    t_approach = 3.0      # Time for approach movements
    t_grasp = 2.0         # Time for grasp/release actions
    t_hold = 1.5          # Time to hold poses
    t_move = 3.5          # Time for larger movements
    
    # Get all poses for specified hand
    poses = get_grasping_poses(hand)
    move_direction = "right" if hand == 'left' else "left"
    
    # Execute complete grasping sequence
    print("📍 Phase 1: Preparation")
    smooth_transition_to_pose(standing_pose, t_approach, pub, cmd, "Moving to standing position")
    smooth_transition_to_pose(poses['ready'], t_approach, pub, cmd, f"Moving {hand} arm to ready position")
    
    print("📍 Phase 2: Approach Object")
    smooth_transition_to_pose(poses['reach'], t_approach, pub, cmd, "Reaching toward object")
    smooth_transition_to_pose(poses['pre_grasp'], t_grasp, pub, cmd, "Positioning for grasp (hand open)")
    hold_pose(poses['pre_grasp'], t_hold, pub, cmd, "Preparing to grasp")
    
    print("📍 Phase 3: Grasp Object")
    smooth_transition_to_pose(poses['grasp'], t_grasp, pub, cmd, "Grasping object (closing hand)")
    hold_pose(poses['grasp'], t_hold, pub, cmd, "Securing grip")
    
    print("📍 Phase 4: Lift and Move")
    smooth_transition_to_pose(poses['lift'], t_approach, pub, cmd, "Lifting object")
    hold_pose(poses['lift'], t_hold, pub, cmd, "Object lifted")
    smooth_transition_to_pose(poses['move'], t_move, pub, cmd, f"Moving object to the {move_direction}")
    
    print("📍 Phase 5: Place and Release")
    smooth_transition_to_pose(poses['place'], t_approach, pub, cmd, "Positioning for placement")
    smooth_transition_to_pose(poses['release'], t_grasp, pub, cmd, "Releasing object (opening hand)")
    hold_pose(poses['release'], t_hold, pub, cmd, "Object released")
    
    print("📍 Phase 6: Return to Rest")
    smooth_transition_to_pose(poses['withdraw'], t_approach, pub, cmd, f"Withdrawing {hand} hand")
    smooth_transition_to_pose(standing_pose, t_move, pub, cmd, "Returning to standing position")
    hold_pose(standing_pose, t_hold, pub, cmd, "Final standing pose")
    
    print(f"✅ {hand.capitalize()} hand grasping sequence completed successfully!")
    print(f"🎯 Summary: Object grasped with {hand} hand, moved to the {move_direction}, and placed successfully!")

# ============================================================================
# USER INTERFACE
# ============================================================================

def display_main_menu():
    """Display the main motion selection menu"""
    print("\n" + "="*50)
    print("🤖 Unitree G1 Motion Control Center")
    print("="*50)
    print("Select a motion to execute:")
    print("1. 🤖 Waving sequence")
    print("2. 👉 Pointing sequence")
    print("3. 🤏 Grasping sequence")
    print("4. 🚪 Exit")
    print("="*50)

def get_hand_selection():
    """Get hand selection from user"""
    print("\n" + "-"*40)
    print("🖐️ Select hand to use:")
    print("-"*40)
    print("1. Left hand")
    print("2. Right hand")
    print("-"*40)
    
    while True:
        try:
            choice = input("Enter your choice (1-2): ").strip()
            hand_map = {
                '1': 'left',
                '2': 'right'
            }
            
            if choice in hand_map:
                return hand_map[choice]
            else:
                print("❌ Invalid choice. Please enter 1-2.")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input. Please enter 1-2.")

def get_pointing_direction():
    """Get pointing direction from user"""
    print("\n" + "-"*40)
    print("👉 Select pointing direction:")
    print("-"*40)
    print("1. Forward")
    print("2. Left")
    print("3. Right")
    print("4. Up")
    print("-"*40)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            direction_map = {
                '1': 'forward',
                '2': 'left', 
                '3': 'right',
                '4': 'up'
            }
            
            if choice in direction_map:
                return direction_map[choice]
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input. Please enter 1-4.")

def get_user_choice():
    """Get motion choice from user"""
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input. Please enter 1-4.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def initialize_robot_connection():
    """Initialize SDK communication and return pub, sub, cmd objects"""
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

    return pub, sub, cmd

def main():
    """Main execution with user menu system"""
    print("🚀 Initializing Unitree G1 Robot Connection...")
    pub, sub, cmd = initialize_robot_connection()
    print("✅ Robot connection established!")
    
    while True:
        display_main_menu()
        choice = get_user_choice()
        
        if choice == '1':
            print("\n🤖 Waving motion selected!")
            hand = get_hand_selection()
            print(f"\n{hand.capitalize()} hand waving selected!")
            input("Press Enter to start waving sequence...")
            execute_waving_sequence(pub, cmd, hand)
            
        elif choice == '2':
            print("\n👉 Pointing motion selected!")
            hand = get_hand_selection()
            direction = get_pointing_direction()
            print(f"\n{hand.capitalize()} hand pointing {direction} selected!")
            input("Press Enter to start pointing sequence...")
            execute_pointing_sequence(pub, cmd, hand, direction)
            
        elif choice == '3':
            print("\n🤏 Grasping motion selected!")
            hand = get_hand_selection()
            print(f"\n{hand.capitalize()} hand grasping selected!")
            input("Press Enter to start grasping sequence...")
            execute_grasping_sequence(pub, cmd, hand)
            
        elif choice == '4':
            print("\n👋 Exiting G1 Motion Control Center...")
            print("🤖 Thank you for using Unitree G1 Robot!")
            break
        
        # Ask if user wants to continue
        print("\n" + "="*50)
        continue_choice = input("Would you like to perform another motion? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n👋 Exiting G1 Motion Control Center...")
            print("🤖 Thank you for using Unitree G1 Robot!")
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user.")
        print("👋 Exiting G1 Motion Control Center...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("👋 Exiting G1 Motion Control Center...")
