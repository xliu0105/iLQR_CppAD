#include "iLQR_CppAD.h"

// --- Constants --- //
#define g 9.81

// --- Class Definition --- //
class Mujoco_Acrobot_iLQR : public iLQR_CppAD
{
public:
// --- Variables --- //
    double m1=1.0; // mass of link 1
    double m2=1.0; // mass of link 2
    double l1=1.0; // length of link 1
    double l2=1.0; // length of link 2
    double r1=0.005; // radius of link 1
    double r2=0.005; // radius of link 2
    double I1=m1*l1*l1/3.0; // moment of inertia of link 1
    double I2=m2*l2*l2/3.0; // moment of inertia of link 2
    double stage_cost_x_theta_weight_ = 1.0; // weight for theta in stage cost
    double stage_cost_x_theta_dot_weight_ = 0.1; // weight for theta_dot in stage cost
    double stage_cost_u_weight_ = 0.1; // weight for u in stage cost
    double terminal_cost_x_weight_ = 100.0; // weight for x in terminal cost

// --- Functions --- //
    void f_dynamic_CppAD();
    Eigen::Vector<CppAD::AD<double>,4> f_CppAD_(Eigen::Vector<CppAD::AD<double>,4> &x, Eigen::Vector<CppAD::AD<double>,1> &u);
    void stage_cost_CppAD();
    void terminal_cost_CppAD();
    Eigen::VectorXd f_dynamic(Eigen::VectorXd &x, Eigen::VectorXd &u);
    Eigen::VectorXd f_(Eigen::VectorXd x, Eigen::VectorXd u);
};