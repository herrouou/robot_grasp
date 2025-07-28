#include "allegro_hand_grasp_plugin/allegro_grasp_plugin.h"
#include <pluginlib/class_list_macros.h>
#include <Eigen/Dense>

namespace allegro_        if (i < 3) { // FF, MF, RF
            base_target << -(config_.radius) * sin_beta,  // Add offset to ensure outside surface
                          -(config_.radius) * cos_beta,
                          config_.z_heights[i];
            fingers_[i].target = cyl_pos + base_target;
        } else { // TH
            base_target << (config_.radius) * sin_beta,   // Add offset to ensure outside surface
                          (config_.radius) * cos_beta,
                          config_.z_heights[i];
            fingers_[i].target = cyl_pos + base_target;
        }plugin {


/* ================= ctor / reset ================= */
AllegroGraspPlugin::AllegroGraspPlugin() :
    phase_(OPEN),
    nv_(0),
    // Keep original open position configuration
    open_pos_({0,0,0,0,  0,0,0,0,  0,0,0,0,  1.2,0,0,0}) {
        last_print_ = ros::Time::now();
    }

void AllegroGraspPlugin::reset() {
    phase_ = OPEN;
    last_print_ = ros::Time::now();
    for (auto& finger : fingers_) {
        finger.reached_target = false;
    }
    
    // Reset static variables (for next experiment)
    // Note: This is a simplified handling, real applications may need better state management
    static bool reset_statics = true;
    if (reset_statics) {
        // Here we can reset static variables in graspObject
        // Since we can't access them directly, we rely on initialization flags
    }
}

Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& J, double lambda2 = 1e-6)
{
    Eigen::MatrixXd JJt = J * J.transpose();
    Eigen::MatrixXd inv = (JJt + lambda2 * Eigen::MatrixXd::Identity(JJt.rows(), JJt.cols())).inverse();
    return J.transpose() * inv;
}

/* ================= load ================= */
bool AllegroGraspPlugin::load(const mjModel* m, mjData* d) {
    ROS_INFO("[AllegroGraspPlugin] Loading plugin...");
    nv_ = m->nv;
    // Initialize joint and actuator mapping
    std::vector<std::string> joint_names = {
        "ffj0","ffj1","ffj2","ffj3",
        "mfj0","mfj1","mfj2","mfj3",
        "rfj0","rfj1","rfj2","rfj3",
        "thj0","thj1","thj2","thj3"
    };
    
    std::vector<std::string> act_names = {
        "ffa0","ffa1","ffa2","ffa3",
        "mfa0","mfa1","mfa2","mfa3",
        "rfa0","rfa1","rfa2","rfa3",
        "tha0","tha1","tha2","tha3"
    };


    /* ---- Look up table ---- */
    qpos_id_.clear(); ctrl_id_.clear();
    for (size_t k=0; k<joint_names.size(); ++k) {
        int jid = mj_name2id(m, mjOBJ_JOINT, joint_names[k].c_str());
        int aid = mj_name2id(m, mjOBJ_ACTUATOR, act_names[k].c_str());
        if (jid < 0 || aid < 0) {
            ROS_ERROR("Joint/actuator %s not found", joint_names[k].c_str());
            return false;
        }
        qpos_id_.push_back(m->jnt_qposadr[jid]);
        ctrl_id_.push_back(aid);
    }

    ff_tip_ = mj_name2id(m, mjOBJ_BODY, "ff_tip");
    mf_tip_ = mj_name2id(m, mjOBJ_BODY, "mf_tip");
    rf_tip_ = mj_name2id(m, mjOBJ_BODY, "rf_tip");
    th_tip_ = mj_name2id(m, mjOBJ_BODY, "th_tip");

    ff_site_ = mj_name2id(m, mjOBJ_SITE, "ff_contact");
    mf_site_ = mj_name2id(m, mjOBJ_SITE, "mf_contact");
    rf_site_ = mj_name2id(m, mjOBJ_SITE, "rf_contact");
    th_site_ = mj_name2id(m, mjOBJ_SITE, "th_contact");
    /* ---- Root sliding DOF actuator ---- */
    rootx_ctrl_ = mj_name2id(m, mjOBJ_ACTUATOR, "rootx");
    rooty_ctrl_ = mj_name2id(m, mjOBJ_ACTUATOR, "rooty");
    rootz_ctrl_ = mj_name2id(m, mjOBJ_ACTUATOR, "rootz");
    // Print debug info
    ROS_INFO("rootx_ctrl: %d, rooty_ctrl: %d, rootz_ctrl: %d", rootx_ctrl_, rooty_ctrl_, rootz_ctrl_);
    if (rootx_ctrl_ < 0 || rooty_ctrl_ < 0 || rootz_ctrl_ < 0) {
        ROS_ERROR("root actuator not found");
        return false;
    }

    /* ---- body id ---- */
    palm_body_id_ = mj_name2id(m, mjOBJ_BODY, "palm");
    cyl_body_id_  = mj_name2id(m, mjOBJ_BODY, "cylinder_object");

    std::vector<std::string> finger_names = {"ff", "mf", "rf", "th"};
    for (int i = 0; i < 4; ++i) {
        Finger f;
        f.name = finger_names[i];
        f.tip_body_id = mj_name2id(m, mjOBJ_BODY, (f.name + "_tip").c_str());
        f.site_id     = mj_name2id(m, mjOBJ_SITE, (f.name + "_contact").c_str());
        f.sensor_id   = mj_name2id(m, mjOBJ_SENSOR, (f.name + "_force").c_str());
        f.qpos_start = 4 * i;
        f.ctrl_start = 4 * i;
        f.reached_target = false;
        f.jacp.resize(3, nv_); 
        
        // Fix: Correct initialization method
        if (i == 0 || i == 2) {
            f.pid = std::make_unique<PID>(1, 0.05, 2); // FF, RF   
        } else if(i == 1){
            f.pid = std::make_unique<PID>(1, 0.05, 2);  // MF
        } else {
            f.pid = std::make_unique<PID>(0.5, 0.1, 2);  // TH
        }
        
        fingers_.push_back(std::move(f)); // Use move semantics
    }
    // Initialize Jacobian buffer
    
    jacp_buffer_.resize(3 * nv_);
    
    ROS_INFO("[AllegroGraspPlugin] Plugin loaded successfully");
    return true;
}


void AllegroGraspPlugin::updateFingerTargets(const mjModel* m, const mjData* d) {
    
    Eigen::Vector3d cyl_pos = {0.0, 0.0, 0.0};
    
    const double beta_rad = config_.beta_angle * M_PI / 180.0;
    const double sin_beta = sin(beta_rad);
    const double cos_beta = cos(beta_rad);
    
    // Debug: Output cylinder position comparison
    static int target_debug_counter = 0;
   
    
    for (int i = 0; i < fingers_.size(); ++i) {
        Eigen::Vector3d base_target;
        
        if (i < 3) { // FF, MF, RF
            base_target << -(config_.radius) * sin_beta,  // 增加偏移量，确保在表面外侧
                          -(config_.radius) * cos_beta,
                          config_.z_heights[i];
            fingers_[i].target = cyl_pos + base_target;
        } else { // TH
            base_target << (config_.radius) * sin_beta,  // 增加偏移量，确保在表面外侧
                          (config_.radius) * cos_beta,
                          config_.z_heights[i];
            fingers_[i].target = cyl_pos + base_target;
        }
        

    }
}



/* ================= Four phases ================= */
void AllegroGraspPlugin::openHand(const mjModel* m, mjData* d) {
    // Set all joints to open position (keep original configuration)
    for (int i = 0; i < 16; ++i) {
        d->ctrl[ctrl_id_[i]] = open_pos_[i];
    }
    
    // Check if opening is complete (consider original configuration)
    bool all_open = true;
    for (int i = 0; i < 16; ++i) {
        if (fabs(d->qpos[qpos_id_[i]] - open_pos_[i]) > 0.05) {
            all_open = false;
            break;
        }
    }
    
    if (all_open) {
        phase_ = ALIGN;
        ROS_INFO("Hand opened, moving to ALIGN phase");
    }
}

void AllegroGraspPlugin::alignHand(const mjModel* m, mjData* d) {
    Eigen::Map<const Eigen::Vector3d> palm_pos(d->xpos + 3*palm_body_id_);
    Eigen::Map<const Eigen::Vector3d> cyl_pos(d->xpos + cyl_body_id_ * 3);
    Eigen::Vector3d error = cyl_pos - palm_pos + config_.hand_offset;
    
    // Position control gain
    const double Kp = 2.0;
    d->ctrl[rootx_ctrl_] = d->qpos[0] + Kp * error.x();
    d->ctrl[rooty_ctrl_] = d->qpos[1] + Kp * error.y();
    d->ctrl[rootz_ctrl_] = d->qpos[2] + Kp * error.z();
    
    if (error.norm() < 0.005) {
        phase_ = GRASP;
        ROS_INFO("Hand aligned, moving to GRASP phase");
    }
}

void AllegroGraspPlugin::graspObject(const mjModel* m, mjData* d) {
    // 1. 更新目标位置
    mj_kinematics(m, d);
    updateFingerTargets(m, d);
    
    // 2. 初始化总控制信号
    Eigen::VectorXd tau_total = Eigen::VectorXd::Zero(nv_);
    
    // 3. Process each finger
    bool all_fingers_reached = true;
    for (auto& finger : fingers_) {
        // Get current position and sensor data
        const mjtNum* site_xpos = d->site_xpos + 3 * finger.site_id;
        const mjtNum* tip_body_xpos = d->xpos + 3 * finger.tip_body_id;
        Eigen::Map<const Eigen::Vector3d> tip_pos(site_xpos);
        finger.error = finger.target - tip_pos;
       

        // Check if target is reached
        if (!finger.reached_target) {
            if (finger.error.norm() < config_.finger_error_threshold) {
                finger.reached_target = true;
                ROS_INFO("Finger %s reached target", finger.name.c_str());
            } else {
                all_fingers_reached = false;
            }
        }
        
        // Calculate task space control signal
    
        // Calculate Jacobian matrix - use site Jacobian

        mj_jacSite(m, d, jacp_buffer_.data(), nullptr, finger.site_id);
        finger.jacp = Eigen::Map<RowMat32d>(jacp_buffer_.data(), 3, nv_);

        Eigen::Vector3d scale(10.0, 10.0, 100.0);
        Eigen::VectorXd u = (*finger.pid)(finger.error);
       
        Eigen::MatrixXd J_pinv = pseudoInverse(finger.jacp, 1e-6);
        Eigen::VectorXd qn = Eigen::VectorXd::Zero(nv_);

        // Get reference joint positions
        Eigen::VectorXd q_ref;
        if (finger.name == "ff") q_ref = Eigen::Map<Eigen::VectorXd>(config_.ff_joint_ref.data(), 4);
        else if (finger.name == "mf") q_ref = Eigen::Map<Eigen::VectorXd>(config_.mf_joint_ref.data(), 4);
        else if (finger.name == "rf") q_ref = Eigen::Map<Eigen::VectorXd>(config_.rf_joint_ref.data(), 4);
        else q_ref = Eigen::Map<Eigen::VectorXd>(config_.th_joint_ref.data(), 4);

        // Null space control term
        double alpha;
        int qstart;
        if (finger.name == "ff") {
            qstart = 3;
            alpha = config_.alpha_ff;
        } else if (finger.name == "mf") {
            qstart = 7;
            alpha = config_.alpha_mf;
        } else if (finger.name == "rf") {
            qstart = 11;
            alpha = config_.alpha_rf;
        } else {
            qstart = 15;
            alpha = config_.alpha_th;
        }
        for (int i = 0; i < 4; ++i)
            qn[qstart+i] = d->qpos[qstart+i] - q_ref[i];
       
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nv_, nv_);
        Eigen::VectorXd qdot = J_pinv * u  - alpha * (I - J_pinv * finger.jacp) * qn;;
        tau_total += qdot;


    }
    
    // // 4. Add force control
    // Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    // S(0,0) = 1.0; // Only consider normal force
    
    // for (auto& finger : fingers_) {
    //     if (finger.sensor_id < 0) continue;
        
    //     // Get force sensor data
        
    //     Eigen::Vector3d f_meas(d->sensordata + finger.sensor_id * 3);
        
    //     // Set desired force
    //     double f_des_val;
    //     if (finger.name == "th") {
    //         f_des_val = config_.desired_force_thumb_grasp;
    //     } else {
    //         f_des_val = config_.desired_force_grasp;
    //     }
    //     Eigen::Vector3d f_des(f_des_val, 0, 0);
        
    //     // Calculate force control term
    //     double kf = (finger.name == "th") ? config_.kf_thumb : config_.kf_force;
    //     Eigen::Vector3d force_ctrl = kf * (f_des - f_meas);
        
    //     // Convert to joint space
    //     Eigen::VectorXd tau_force = finger.J.transpose() * S * force_ctrl;
        
        // // Accumulate force control signal to total control vector (only for this finger's joints)
        // for (int k = 0; k < 4; ++k) {
        //     int global_idx = qpos_id_[finger.qpos_start + k];
        //     if (global_idx < tau_total.size()) {
        //         tau_total[global_idx] += tau_force[global_idx];
        //     }
        // }
    // }
    
    // 5. Apply control signal to joints (avoid overwriting rootz control)
    for (int i = 3; i < nv_; ++i)
        d->ctrl[i] = d->qpos[i] + tau_total[i];
    
    // Force maintain Z-axis position - use stronger control gain to resist gravity and disturbances
    static double target_z = -1.0;  // Initialization flag
    if (target_z < -0.5) {  // First time entering GRASP phase, record target height
        target_z = d->qpos[2];
    }
    
    // Use strong PD control to maintain Z position
    double z_error = target_z - d->qpos[2];
    double Kp_z = 5.0;  // Strong position gain

    
    d->ctrl[rootz_ctrl_] = d->qpos[2] + Kp_z * z_error;
    
    // 6. Check if grasping is complete
    if (all_fingers_reached) {
        // Print control signals for joints 3-18
        for (int i = 3; i < 18; ++i) {
            ROS_INFO("Control initial[%d]: %f", i, d->ctrl[i]);
        }
        phase_ = LIFT;
        ROS_INFO("All fingers reached targets, moving to LIFT phase");
    }
}

void AllegroGraspPlugin::liftObject(const mjModel* m, mjData* d) {
    // 1. Set lifting target
    double current_z = d->qpos[2];
    double target_z = 0.15;

    // 2. Position control
    double Kp = 0.5;
    double error_z = target_z - current_z;
    d->ctrl[rootz_ctrl_] = d->qpos[2] + Kp * error_z;
    
    // 3. In LIFT phase, finger control signals remain unchanged, no longer updated
    // Finger control signals from end of GRASP phase will be maintained, only control palm lifting
    
    // 4. Check if lifting is complete
    if (fabs(error_z) < 0.001) {
        ROS_INFO("Lift completed");
        // Here you can add logic after completion
    }
}

/* ================= Main callback ================= */
void AllegroGraspPlugin::controlCallback(const mjModel* m, mjData* d) {
    mj_kinematics(m, d);
    switch (phase_) {
        case OPEN :  openHand   (m,d); break;
        case ALIGN:  alignHand  (m,d); break;
        case GRASP:  graspObject(m,d); break;
        case LIFT :  liftObject (m,d); break;
    }
    if ((ros::Time::now() - last_print_).toSec() > 1.0) {
        ROS_INFO_STREAM("Allegro phase: " << phase_);
        last_print_ = ros::Time::now();
    }
}

} // namespace

PLUGINLIB_EXPORT_CLASS(allegro_hand_grasp_plugin::AllegroGraspPlugin,
                       mujoco_ros::MujocoPlugin)

