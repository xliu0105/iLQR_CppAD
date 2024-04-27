#include "Mujoco_Acrobot.h"

// Acrobot dynamic equation reference: https://blog.csdn.net/weixin_46536094/article/details/123582939
Eigen::Vector<CppAD::AD<double>,4> Mujoco_Acrobot_iLQR::f_CppAD_(Eigen::Vector<CppAD::AD<double>,4> &x, Eigen::Vector<CppAD::AD<double>,1> &u)
{
    Eigen::Vector<CppAD::AD<double>,4> x_next(4);
    Eigen::Matrix<CppAD::AD<double>,2,2> M;
    M(0,0) = I1+I2+m2*l1*l1+2.0*m2*l1*l2/2.0*cos(x(1));
    M(0,1) = I2 + m2*l1*l2/2.0*cos(x(1));
    M(1,0) = I2 + m2*l1*l2/2.0*cos(x(1));
    M(1,1) = I2;
    Eigen::Matrix<CppAD::AD<double>,2,2> C;
    C(0,0) = -2.0*m2*l1*l2/2.0*sin(x(1))*x(3);
    C(0,1) = -m2*l1*l2/2.0*sin(x(1))*x(3);
    C(1,0) = m2*l1*l2/2.0*sin(x(1))*x(2);
    C(1,1) = 0;
    Eigen::Vector<CppAD::AD<double>,2> G;
    G(0) = -m1*g*l1/2.0*sin(x(0))-m2*g*(l1*sin(x(0))+l2/2.0*sin(x(0)+x(1)));
    G(1) = -m2*g*l2/2.0*sin(x(0)+x(1));
    Eigen::Vector<CppAD::AD<double>,2> B;
    B(0) = 0.0;
    B(1) = 1.0;
    Eigen::Vector<CppAD::AD<double>,2> q_double_dot;
    q_double_dot = M.inverse()*(G + B*u - C*Eigen::Vector<CppAD::AD<double>,2>(x(2),x(3)));
    x_next << x(2),x(3),q_double_dot(0),q_double_dot(1);
    return x_next;
}

void Mujoco_Acrobot_iLQR::f_dynamic_CppAD()
{
    ADvector xu_ad(state_dim_+ctrl_dim_);
    for(int i = 0; i < state_dim_+ctrl_dim_; i++)
    {
        xu_ad[i] = 0.1;
    }
    CppAD::Independent(xu_ad);
    Eigen::Vector<CppAD::AD<double>,4> x_ad_eigen;
    Eigen::Vector<CppAD::AD<double>,1> u_ad_eigen;
    for (int i = 0; i < state_dim_; i++)
    {
        x_ad_eigen(i) = xu_ad[i];
    }
    for (int i = 0; i < ctrl_dim_; i++)
    {
        u_ad_eigen(i) = xu_ad[state_dim_+i];
    }
    Eigen::Vector<CppAD::AD<double>,4> K1 = f_CppAD_(x_ad_eigen,u_ad_eigen);
    Eigen::Vector<CppAD::AD<double>,4> a1 = x_ad_eigen+0.5*dt_*K1;
    Eigen::Vector<CppAD::AD<double>,4> K2 = f_CppAD_(a1,u_ad_eigen);
    Eigen::Vector<CppAD::AD<double>,4> a2 = x_ad_eigen+0.5*dt_*K2;
    Eigen::Vector<CppAD::AD<double>,4> K3 = f_CppAD_(a2,u_ad_eigen);
    Eigen::Vector<CppAD::AD<double>,4> a3 = x_ad_eigen+dt_*K3;
    Eigen::Vector<CppAD::AD<double>,4> K4 = f_CppAD_(a3,u_ad_eigen);
    Eigen::Vector<CppAD::AD<double>,4> x_out = x_ad_eigen + (dt_/6.0)*(K1+2.0*K2+2.0*K3+K4);
    ADvector y_ad(state_dim_);
    for (int i = 0; i < state_dim_; i++)
    {
        y_ad[i] = x_out(i);
    }
    f_dynamic_ad_.Dependent(xu_ad,y_ad);

    std::cout << "Create f_Dynamic for CppAD" << std::endl;
    
}

Eigen::VectorXd Mujoco_Acrobot_iLQR::f_(Eigen::VectorXd x, Eigen::VectorXd u)
{
    Eigen::Matrix2d M;
    M(0,0) = I1+I2+m2*l1*l1+2.0*m2*l1*l2/2.0*cos(x(1));
    M(0,1) = I2 + m2*l1*l2/2.0*cos(x(1));
    M(1,0) = I2 + m2*l1*l2/2.0*cos(x(1));
    M(1,1) = I2;
    Eigen::Matrix2d C;
    C(0,0) = -2.0*m2*l1*l2/2.0*sin(x(1))*x(3);
    C(0,1) = -m2*l1*l2/2.0*sin(x(1))*x(3);
    C(1,0) = m2*l1*l2/2.0*sin(x(1))*x(2);
    C(1,1) = 0.0;
    Eigen::Vector2d G;
    G(0) = -m1*g*l1/2.0*sin(x(0))-m2*g*(l1*sin(x(0))+l2/2.0*sin(x(0)+x(1)));
    G(1) = -m2*g*l2/2.0*sin(x(0)+x(1));
    Eigen::Vector2d B;
    B(0) = 0.0;
    B(1) = 1.0;
    Eigen::Vector2d q_double_dot;
    q_double_dot = M.inverse()*(G + B*u -  C*Eigen::Vector2d(x(2),x(3)));
    return Eigen::Vector4d(x(2),x(3),q_double_dot(0),q_double_dot(1));
}

Eigen::VectorXd Mujoco_Acrobot_iLQR::f_dynamic(Eigen::VectorXd &x, Eigen::VectorXd &u)
{
    Eigen::VectorXd K1,K2,K3,K4,x_next;
    K1 = f_(x,u);
    K2 = f_(x+0.5*dt_*K1,u);
    K3 = f_(x+0.5*dt_*K2,u);
    K4 = f_(x+dt_*K3,u);
    x_next = x + (dt_/6.0)*(K1+2.0*K2+2.0*K3+K4);
    return x_next;
}

void Mujoco_Acrobot_iLQR::stage_cost_CppAD()
{
    ADvector xu_ad(state_dim_+ctrl_dim_);
    for(int i = 0; i < state_dim_+ctrl_dim_; i++)
    {
        xu_ad[i] = 0.1;
    }
    CppAD::Independent(xu_ad);
    Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,1> xu_ad_eigen(5);
    CPPAD_TEST_VECTOR<double> goal(5);
    goal[0] = PI;
    goal[1] = 0.0;
    goal[2] = 0.0;
    goal[3] = 0.0;
    goal[4] = 0.0;
    for (int i = 0; i < state_dim_+ctrl_dim_; i++)
    {
        xu_ad_eigen(i) = xu_ad[i]-goal[i];
    }
    Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,Eigen::Dynamic> Q(state_dim_+ctrl_dim_,state_dim_+ctrl_dim_);
    Q = Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,Eigen::Dynamic>::Zero(state_dim_+ctrl_dim_,state_dim_+ctrl_dim_);
    Q.block(0,0,2,2) = stage_cost_x_theta_weight_*Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,Eigen::Dynamic>::Identity(2,2);
    Q.block(2,2,2,2) = stage_cost_x_theta_dot_weight_*Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,Eigen::Dynamic>::Identity(2,2);
    Q.block(state_dim_,state_dim_,ctrl_dim_,ctrl_dim_) = stage_cost_u_weight_*Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,Eigen::Dynamic>::Identity(ctrl_dim_,ctrl_dim_);
    ADvector y_ad(1);
    y_ad[0] = (0.5*xu_ad_eigen.transpose()*Q*xu_ad_eigen)(0,0);
    stage_cost_ad_.Dependent(xu_ad,y_ad);
    std::cout << "Create stage cost for CppAD" << std::endl;
}

void Mujoco_Acrobot_iLQR::terminal_cost_CppAD()
{
    ADvector x_n_ad(state_dim_);
    for(int i = 0; i < state_dim_; i++)
    {
        x_n_ad[i] = 0.1;
    }
    CppAD::Independent(x_n_ad);
    Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,1> x_n_ad_eigen(state_dim_);
    CPPAD_TEST_VECTOR<double> goal(4);
    goal[0] = PI;
    goal[1] = 0.0;
    goal[2] = 0.0;
    goal[3] = 0.0;
    for (int i = 0; i < state_dim_; i++)
    {
        x_n_ad_eigen(i) = x_n_ad[i]-goal[i];
    }
    Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,Eigen::Dynamic> Q(state_dim_,state_dim_);
    Q = terminal_cost_x_weight_*Eigen::Matrix<CppAD::AD<double>,Eigen::Dynamic,Eigen::Dynamic>::Identity(state_dim_,state_dim_);
    ADvector y_ad(1);
    y_ad[0] = (0.5*x_n_ad_eigen.transpose()*Q*x_n_ad_eigen)(0,0);
    terminal_cost_ad_.Dependent(x_n_ad,y_ad);
    std::cout << "Create Terminal cost for CppAD" << std::endl;
}

int main(int argc, char *argv[])
{
    srand(static_cast<unsigned int>(time(0)));
    Mujoco_Acrobot_iLQR acrobot_iLQR;
    acrobot_iLQR.setInitialState(Eigen::VectorXd::Zero(4));
    acrobot_iLQR.setTrajLen(200);
    acrobot_iLQR.setDt(0.05);
    acrobot_iLQR.setMaxIter(1000);
    acrobot_iLQR.setConveregenceThresh(1e-2);
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> init_ctrl;
    init_ctrl.resize(acrobot_iLQR.getTrajLen()-1);
    for(auto & vec : init_ctrl)
    {
        vec = Eigen::VectorXd::Random(1);
    }
    acrobot_iLQR.setInitialCtrlTraj(init_ctrl);
    acrobot_iLQR.f_dynamic_CppAD();
    acrobot_iLQR.stage_cost_CppAD();
    acrobot_iLQR.terminal_cost_CppAD();
    acrobot_iLQR.iLQR_Optimize();
    return 0;
}