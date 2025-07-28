#pragma once
#include <mujoco/mujoco.h>
#include <mujoco_ros/plugin_utils.h>
#include <ros/ros.h>
#include <vector>
#include <string>
#include <Eigen/Dense>

using RowMat32d = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;
namespace allegro_hand_grasp_plugin {

class PID {
public:
    PID(double kp, double ki, double kd)
            : kp_(kp), ki_(ki), kd_(kd),
              integ_(Eigen::VectorXd::Zero(1)),
              prev_(Eigen::VectorXd::Zero(1)) {}

    // Add Vector3d overload
    Eigen::Vector3d operator()(const Eigen::Vector3d& err) {
        if (integ_.size() != 3) {
            integ_.setZero(3);
            prev_ = err;
        }
        integ_ += err;
        Eigen::Vector3d deriv = err - prev_;
        prev_ = err;
        return kp_*err + ki_*integ_ + kd_*deriv;
    }
    
    // Keep original VectorXd version
    Eigen::VectorXd operator()(const Eigen::VectorXd& err) {
        if (integ_.size() != err.size()) {
            integ_.setZero(err.size());
            prev_ = err;
        }
        integ_ += err;
        Eigen::VectorXd deriv = err - prev_;
        prev_ = err;
        return kp_*err + ki_*integ_ + kd_*deriv;
    }
    
private:
    double kp_, ki_, kd_;
    Eigen::Vector3d integ_, prev_;  // Also need corresponding types
};

class AllegroGraspPlugin : public mujoco_ros::MujocoPlugin
{
public:
    AllegroGraspPlugin();

    // MujocoPlugin interface
    bool load   (const mjModel* m, mjData* d) override;
    void controlCallback(const mjModel* m, mjData* d) override;
    void reset  () override;

private:
    enum Phase { OPEN, ALIGN, GRASP, LIFT };
    Phase phase_;

    // Information for each finger
    struct Finger {
        std::string name;         // e.g. "ff"
        int tip_body_id;          // e.g. ff_tip
        int site_id; 
        int sensor_id;            // force sensor
        int qpos_start;           // joint index
        int ctrl_start;           // actuator index
        int dof_id; 
        Eigen::MatrixXd jacp;
        Eigen::Vector3d target;   // Contact target
        Eigen::MatrixXd J;        // Jacobian
        Eigen::Vector3d error;    // Current error
        Eigen::Vector3d force;    // Current sensor force
        std::unique_ptr<PID> pid; 
        bool reached_target;      // Whether target position is reached
    };
    std::vector<Finger> fingers_;  // Four fingers

    /* ---------- ID indices ---------- */
    std::vector<int> qpos_id_;   // 16 joint indices in qpos
    std::vector<int> ctrl_id_;   // Corresponding actuator id
    int rootx_ctrl_{-1}, rooty_ctrl_{-1}, rootz_ctrl_{-1};
    int palm_body_id_{-1}, cyl_body_id_{-1};
    int ff_tip_{-1}, mf_tip_{-1}, rf_tip_{-1}, th_tip_{-1};
    int ff_site_{-1}, mf_site_{-1}, rf_site_{-1}, th_site_{-1};

    // Open position configuration
    std::vector<double> open_pos_;

    /* ---------- Parameters ---------- */
    struct GraspConfig {
        double radius = 0.05;
        double beta_angle = 7.0;  // degrees
        Eigen::Vector3d contact_offset = Eigen::Vector3d(0.02, 0.02, 0);
        std::vector<double> z_heights = {0.15, 0.10, 0.05, 0.115}; // FF, MF, RF, TH
        Eigen::Vector3d hand_offset = Eigen::Vector3d(-0.102, 0.01, 0.02);
        double lift_height = 0.1;
        double finger_error_threshold = 0.005;  
        
        // Reference joint positions
        std::vector<double> ff_joint_ref = {0, 1.0, 1.5, 1.5};
        std::vector<double> mf_joint_ref = {0, 1.0, 1.5, 1.5};
        std::vector<double> rf_joint_ref = {0, 1.0, 1.5, 1.5};
        std::vector<double> th_joint_ref = {1.4, 0, 0.8, 0.8};
        
        // Control gains   
        double alpha_ff = 0.01;
        double alpha_mf = 0.01;
        double alpha_rf = 0.01;
        double alpha_th = 0.01;
        
        // Force control parameters
        double desired_force_grasp = 5.0;
        double desired_force_thumb_grasp = 2.5;
        double desired_force_lift = 0.2;
        double kf_force = 0.1;
        double kf_thumb = 0.05;
    };
    GraspConfig config_;
    int once = 0;

    // Jacobian calculation buffer
    std::vector<mjtNum> jacp_buffer_;
    int nv_{0};
    ros::Time last_print_;

    /* ---------- Phase methods ---------- */
    void openHand   (const mjModel* m, mjData* d);
    void alignHand  (const mjModel* m, mjData* d);
    void graspObject(const mjModel* m, mjData* d);
    void liftObject (const mjModel* m, mjData* d);
    void updateFingerTargets(const mjModel* m, const mjData* d);

    void printDebugInfo(const mjModel* m, mjData* d);
};

} // namespace
