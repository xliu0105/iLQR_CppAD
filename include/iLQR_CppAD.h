#ifndef ILQR_CPPAD_H
#define ILQR_CPPAD_H

// --- Eigen Headers --- //
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

// --- Standard Headers --- //
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <thread>
#include <fstream>

// --- CppAD Headers --- //
#include <cppad/cppad.hpp>

// --- Constants --- //
#define PI 3.14159265358979323846

// --- using --- //
using ADvector = CPPAD_TESTVECTOR(CppAD::AD<double>);
// you can also use CPPAD_TEST_VECTOR: using ADvector = CPPAD_TEST_VECTOR<CppAD::AD<double>>;


// --- Define --- //
// #define CPPAD_TEST_VECTOR CppAD::vector

// --- iLQR Class --- //

class iLQR_CppAD // this class contains a pure virtual function, so it is an abstract class
{
protected:
// --- Variables --- //
    int traj_len_ = 200; // length of the trajectory， default is 200
    double dt_ = 2e-3; // time step, default is 0.002
    int state_dim_; // state dimension
    int ctrl_dim_; // control dimension
    int max_iter_ = 0; // maximum number of iterations, 0 means no limit
    double conv_thresh_ = 1e-2; // convergence threshold, default is 0.01
    double forward_cost_; // forward cost
    double line_search_forward_cost_; // forward cost for line search
    double line_search_alpha_ = 1.0; // line search alpha
    double line_search_c = 0.5;
    double line_search_b = 1e-2;
    double delta_J = 0.0; // change in cost
    // --- CppAD Variables --- //
    CppAD::ADFun<double> f_dynamic_ad_; // dynamic function for CppAD
    CppAD::ADFun<double> stage_cost_ad_; // stage cost function for CppAD
    CppAD::ADFun<double> terminal_cost_ad_; // terminal cost function for CppAD
    // --- Eigen Variables --- //
    Eigen::VectorXd init_state_; // initial state
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> states_traj_; // states trajectory
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> line_search_states_traj_;
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> ctrl_traj_; // control trajectory
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> line_search_ctrl_traj_; // control trajectory
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> d_iLQR_; // Adjustment to the control trajectory
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> K_iLQR_; // feedback gain matrix acquired from the iLQR backward pass
    Eigen::MatrixXd Capital_P_kp1_; // Represents P_(k+1) in iLQR
    Eigen::VectorXd Lowercase_p_kp1_; // Represents p_(k+1) in iLQR


public:
// --- Variables --- //

// --- Functions --- //
    iLQR_CppAD();
    ~iLQR_CppAD();
    void setInitialState(Eigen::VectorXd init_state);
    void setInitialCtrlTraj(std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> &init_ctrl);
    void setTrajLen(int traj_len);
    void setDt(double dt);
    void setMaxIter(int max_iter);
    void setConveregenceThresh(double conv_thresh);
    int getTrajLen();
    int getStateDim();
    virtual Eigen::VectorXd f_dynamic(Eigen::VectorXd &x, Eigen::VectorXd &u) = 0; // pure virtual function, used for the forward dynamics in iLQR
    virtual void f_dynamic_CppAD() = 0;
    virtual void terminal_cost_CppAD() = 0;
    virtual void stage_cost_CppAD() = 0;
    void iLQR_Forward();
    double iLQR_Forward_cost(std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> &states_traj_,std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> &ctrl_traj_);
    void iLQR_Backward();
    void iLQR_Line_Search();
    // void check_Posdef_Condnum(Eigen::MatrixXd &G_uu); // check if the matrix is positive definite and the condition number
    void check_Posdef_Condnum(Eigen::MatrixXd &G_uu);
    bool detectConvergence();
    void iLQR_Optimize();
};

#endif