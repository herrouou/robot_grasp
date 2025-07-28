import time
import mujoco
import mujoco.viewer
import numpy as np
import math


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """Configuration parameters for the grasping system"""
    XML_PATH = "scene_right_box.xml"
    
    # Control tolerances
    ERROR_THRESHOLD = 0.001
    FINGER_ERROR_THRESHOLD = 0.00008
    
    # Grasp parameters
    RADIUS = 0.02
    BETA_ANGLE = 0  # degrees
    
    # Lift parameters
    LIFT_HEIGHT = 0.1
    
    # Joint position references for null space control
    FF_JOINT_REF = [0, 1.0, 1.8, 1.8]
    MF_JOINT_REF = [0, 1.0, 1.8, 1.8]
    RF_JOINT_REF = [0, 1.0, 1.8, 1.8]
    TH_JOINT_REF = [1.4, 0, 1.0, 1.8]
    
    # Null space control weights
    ALPHA_FF = 0.3
    ALPHA_MF = 0.3
    ALPHA_RF = 0.3
    ALPHA_TH = 0.2


# =============================================================================
# PID CONTROLLER CLASS
# =============================================================================

class PIDController:
    """PID Controller for position control"""
    
    def __init__(self, kp, ki, kd):
        """
        Initialize PID controller
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0
        self.prev_error = 0

    def compute_control_signals(self, error):
        """
        Compute PID control signals
        
        Args:
            error (np.array): Position error vector
            
        Returns:
            np.array: Control signals
        """
        # Proportional term
        p = self.kp * error

        # Integral term
        self.integral_error += error
        i = self.ki * self.integral_error

        # Derivative term
        derivative_error = error - self.prev_error
        d = self.kd * derivative_error

        # Calculate total control signals
        control_signals = p + i + d

        # Update previous error for next iteration
        self.prev_error = error.copy()

        return control_signals


# =============================================================================
# GRASPING CONTROL SYSTEM
# =============================================================================

class GraspingController:
    """Main grasping control system"""
    
    def __init__(self, model, data):
        """
        Initialize the grasping controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        self.config = Config()
        
        # Initialize control state flags
        self.reset_flags()
        
        # Initialize PID controllers for different phases
        self.init_controllers()
        
        # Calculate target contact positions
        self.calculate_target_positions()
        
        # Get body indices for Jacobian computation
        self.get_body_indices()
        
        # Initialize Jacobian matrices
        self.init_jacobians()

    def reset_flags(self):
        """Reset all control phase flags"""
        self.open_flag = False
        self.init_flag = True
        self.grasp_flag = False
        self.lift_flag = False
        self.ff_flag = False
        self.mf_flag = False
        self.rf_flag = False
        self.th_flag = False

    def init_controllers(self):
        """Initialize PID controllers for different control phases"""
        # Hand opening controller
        self.open_pid = PIDController(4, 0.08, 0.01)
        
        # Initial positioning controller
        self.initial_pid = PIDController(1, 0.001, 0.01)
        
        # Finger controllers
        self.ff_pid = PIDController(1.8, 0.18, 0.01)
        self.mf_pid = PIDController(1.8, 0.2, 0.01)
        self.rf_pid = PIDController(1.8, 0.18, 0.01)
        self.th_pid = PIDController(0.5, 0.1, 0.01)
        
        # Target lifting controller
        self.target_pid = PIDController(0.02, 0.0005, 0.0001)

    def calculate_target_positions(self):
        """Calculate target contact positions for each finger"""
        sin_beta = math.sin(self.config.BETA_ANGLE * math.pi / 180)
        cos_beta = math.cos(self.config.BETA_ANGLE * math.pi / 180)
        
        # Target positions relative to object center
        self.ff_contact_target = np.array([
            -self.config.RADIUS * sin_beta,
            -self.config.RADIUS * cos_beta,
            0.15
        ])
        
        self.mf_contact_target = np.array([
            -self.config.RADIUS * sin_beta,
            -self.config.RADIUS * cos_beta,
            0.10
        ])
        
        self.rf_contact_target = np.array([
            -self.config.RADIUS * sin_beta,
            -self.config.RADIUS * cos_beta,
            0.05
        ])
        
        self.th_contact_target = np.array([
            self.config.RADIUS * sin_beta,
            self.config.RADIUS * cos_beta,
            0.115
        ])

    def get_body_indices(self):
        """Get body indices for Jacobian computation"""
        self.ff_tip_idx = self.model.body('ff_tip').id
        self.mf_tip_idx = self.model.body('mf_tip').id
        self.rf_tip_idx = self.model.body('rf_tip').id
        self.th_tip_idx = self.model.body('th_tip').id
        self.cylinder_object_idx = self.model.body('cylinder_object').id

    def init_jacobians(self):
        """Initialize Jacobian matrices for each finger"""
        self.ff_contact_jacp = np.zeros((3, self.model.nv))
        self.ff_contact_jacr = np.zeros((3, self.model.nv))
        self.mf_contact_jacp = np.zeros((3, self.model.nv))
        self.mf_contact_jacr = np.zeros((3, self.model.nv))
        self.rf_contact_jacp = np.zeros((3, self.model.nv))
        self.rf_contact_jacr = np.zeros((3, self.model.nv))
        self.th_contact_jacp = np.zeros((3, self.model.nv))
        self.th_contact_jacr = np.zeros((3, self.model.nv))

    def control_hand_opening(self):
        """Control phase 1: Open the hand"""
        distance_to_open = np.array([1.2]) - self.data.qpos[15]
        self.data.ctrl[15] = self.open_pid.compute_control_signals(distance_to_open)
        
        if distance_to_open < 0.005:
            self.open_flag = False
            self.init_flag = True

    def control_initial_positioning(self):
        """Control phase 2: Position hand near target object"""
        hand_pos = self.data.body('palm').xpos
        target_pos = self.data.body('cylinder_object').xpos
        
        # Calculate desired offset from target
        distance_to_target = target_pos - hand_pos + np.array([-0.102, -0.02, 0])
        self.data.ctrl[0:3] = self.initial_pid.compute_control_signals(distance_to_target)
        
        # Check if positioning is complete
        if np.sum(distance_to_target**2) < self.config.ERROR_THRESHOLD:
            self.init_flag = False
            self.grasp_flag = True

    def control_grasping(self):
        """Control phase 3: Execute grasping motion"""
        # Get current finger contact positions
        ff_pos = self.data.site('ff_contact').xpos
        mf_pos = self.data.site('mf_contact').xpos
        rf_pos = self.data.site('rf_contact').xpos
        th_pos = self.data.site('th_contact').xpos

        # Calculate position errors for each finger
        ff_error = self.ff_contact_target - ff_pos
        mf_error = self.mf_contact_target - mf_pos
        rf_error = self.rf_contact_target - rf_pos
        th_error = self.th_contact_target - th_pos

        # Check if each finger has reached its target
        if np.sum(ff_error**2) < self.config.FINGER_ERROR_THRESHOLD:
            self.ff_flag = True
        if np.sum(mf_error**2) < self.config.FINGER_ERROR_THRESHOLD:
            self.mf_flag = True
        if np.sum(rf_error**2) < self.config.FINGER_ERROR_THRESHOLD:
            self.rf_flag = True
        if np.sum(th_error**2) < self.config.FINGER_ERROR_THRESHOLD:
            self.th_flag = True

        # Check if all fingers are in contact
        if all([self.ff_flag, self.mf_flag, self.rf_flag, self.th_flag]):
            self.grasp_flag = False
            self.lift_flag = True
            self.target_final_pos = self.data.body("palm").xpos[2] + self.config.LIFT_HEIGHT

        # Compute control signals in task space
        ff_control = self.ff_pid.compute_control_signals(ff_error)
        mf_control = self.mf_pid.compute_control_signals(mf_error)
        rf_control = self.rf_pid.compute_control_signals(rf_error)
        th_control = self.th_pid.compute_control_signals(th_error)

        # Compute Jacobians for each finger
        mujoco.mj_jac(self.model, self.data, self.ff_contact_jacp, self.ff_contact_jacr, 
                     self.data.site('ff_contact').xpos, self.ff_tip_idx)
        mujoco.mj_jac(self.model, self.data, self.mf_contact_jacp, self.mf_contact_jacr, 
                     self.data.site('mf_contact').xpos, self.mf_tip_idx)
        mujoco.mj_jac(self.model, self.data, self.rf_contact_jacp, self.rf_contact_jacr, 
                     self.data.site('rf_contact').xpos, self.rf_tip_idx)
        mujoco.mj_jac(self.model, self.data, self.th_contact_jacp, self.th_contact_jacr, 
                     self.data.site('th_contact').xpos, self.th_tip_idx)

        # Reshape Jacobians
        self.ff_contact_jacp = self.ff_contact_jacp.reshape((3, self.model.nv))
        self.mf_contact_jacp = self.mf_contact_jacp.reshape((3, self.model.nv))
        self.rf_contact_jacp = self.rf_contact_jacp.reshape((3, self.model.nv))
        self.th_contact_jacp = self.th_contact_jacp.reshape((3, self.model.nv))

        # Convert task space control to joint space with null space control
        joint_controls = self.compute_joint_space_control(
            ff_control, mf_control, rf_control, th_control
        )

        # Apply joint space controls
        self.data.ctrl[3:] = joint_controls

    def compute_joint_space_control(self, ff_control, mf_control, rf_control, th_control):
        """
        Convert task space control signals to joint space using pseudo-inverse Jacobian
        with null space control for joint limit avoidance
        """
        # Compute pseudo-inverse Jacobians
        J_pinv_ff = np.linalg.pinv(self.ff_contact_jacp)
        J_pinv_mf = np.linalg.pinv(self.mf_contact_jacp)
        J_pinv_rf = np.linalg.pinv(self.rf_contact_jacp)
        J_pinv_th = np.linalg.pinv(self.th_contact_jacp)

        # Initialize null space vectors
        H_q_ff = np.zeros((self.model.nv, 1))
        H_q_mf = np.zeros((self.model.nv, 1))
        H_q_rf = np.zeros((self.model.nv, 1))
        H_q_th = np.zeros((self.model.nv, 1))

        # Compute null space vectors for joint limit avoidance
        H_q_ff[3:7] = (self.data.qpos[3:7].reshape(4, 1) - 
                       np.array(self.config.FF_JOINT_REF).reshape(4, 1))
        H_q_mf[7:11] = (self.data.qpos[7:11].reshape(4, 1) - 
                        np.array(self.config.MF_JOINT_REF).reshape(4, 1))
        H_q_rf[11:15] = (self.data.qpos[11:15].reshape(4, 1) - 
                         np.array(self.config.RF_JOINT_REF).reshape(4, 1))
        H_q_th[15:19] = (self.data.qpos[15:19].reshape(4, 1) - 
                         np.array(self.config.TH_JOINT_REF).reshape(4, 1))

        # Compute joint space control with null space projection
        ff_joint = (J_pinv_ff @ ff_control - 
                   (self.config.ALPHA_FF * (np.eye(self.model.nv) - 
                    J_pinv_ff @ self.ff_contact_jacp) @ H_q_ff).reshape(self.model.nv,))
        
        mf_joint = (J_pinv_mf @ mf_control - 
                   (self.config.ALPHA_MF * (np.eye(self.model.nv) - 
                    J_pinv_mf @ self.mf_contact_jacp) @ H_q_mf).reshape(self.model.nv,))
        
        rf_joint = (J_pinv_rf @ rf_control - 
                   (self.config.ALPHA_RF * (np.eye(self.model.nv) - 
                    J_pinv_rf @ self.rf_contact_jacp) @ H_q_rf).reshape(self.model.nv,))
        
        th_joint = (J_pinv_th @ th_control - 
                   (self.config.ALPHA_TH * (np.eye(self.model.nv) - 
                    J_pinv_th @ self.th_contact_jacp) @ H_q_th).reshape(self.model.nv,))

        # Extract relevant joint controls
        joint_controls = np.concatenate([
            ff_joint[3:7],
            mf_joint[7:11],
            rf_joint[11:15],
            th_joint[15:19]
        ])

        return joint_controls

    def control_lifting(self):
        """Control phase 4: Lift the grasped object"""
        current_z = self.data.body('palm').xpos[2]
        distance_to_target = self.target_final_pos - current_z
        
        self.data.ctrl[2] = self.target_pid.compute_control_signals(distance_to_target)
        
        if distance_to_target < 0.0001:
            self.lift_flag = False

    def update_control(self):
        """Main control update function - executes appropriate control phase"""
        if self.open_flag:
            self.control_hand_opening()
        elif self.init_flag:
            self.control_initial_positioning()
        elif self.grasp_flag:
            self.control_grasping()
        elif self.lift_flag:
            self.control_lifting()

    def print_debug_info(self):
        """Print debug information about finger positions"""
        print("ff_contact:", self.data.site('ff_contact').xpos)
        print("mf_contact:", self.data.site('mf_contact').xpos)
        print("rf_contact:", self.data.site('rf_contact').xpos)
        print("th_contact:", self.data.site('th_contact').xpos)


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def main():
    """Main simulation function"""
    # Load MuJoCo model and initialize data
    model = mujoco.MjModel.from_xml_path(Config.XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    renderer = mujoco.Renderer(model)

    # Reset simulation data
    mujoco.mj_resetData(model, data)

    # Initialize grasping controller
    controller = GraspingController(model, data)

    # Debug: Print initial contact information
    print("Initial contacts:", data.contact)

    # Launch viewer and run simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_start = time.time()
        last_print_time = step_start
        start = step_start

        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            current_time = time.time()

            # Print debug information every second
            if current_time - last_print_time >= 1:
                print(f"Time: {current_time:.2f}s")
                controller.print_debug_info()
                last_print_time = current_time

            # Update control signals
            controller.update_control()

            # Step simulation
            mujoco.mj_step(model, data)
            renderer.update_scene(data)

            # Update viewer
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.sync()

            # Maintain real-time simulation
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()