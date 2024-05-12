#include "iLQR_CppAD.h"

iLQR_CppAD::iLQR_CppAD()
{
    // Constructor
}

iLQR_CppAD::~iLQR_CppAD()
{
    // Destructor
}

void iLQR_CppAD::setInitialState(Eigen::VectorXd init_state)
{
    this->init_state_ = init_state;
    state_dim_ = init_state.size();
    Lowercase_p_kp1_.resize(state_dim_); // resize the Lowercase_p_kp1_ vector
    Capital_P_kp1_.resize(state_dim_,state_dim_); // resize the Capital_P_kp1_ matrix

}

int iLQR_CppAD::getStateDim()
{
    return this->state_dim_;
}

void iLQR_CppAD::setInitialCtrlTraj(std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> &init_ctrl)
{
    this->ctrl_traj_ = init_ctrl;
    ctrl_dim_ = init_ctrl[0].size();
}

void iLQR_CppAD::setTrajLen(int traj_len)
{
    this->traj_len_ = traj_len;
    states_traj_.reserve(traj_len); // reserve memory for the states trajectory
    ctrl_traj_.reserve(traj_len-1);
    line_search_states_traj_.reserve(traj_len); // reserve memory for the line search states trajectory
    line_search_ctrl_traj_.reserve(traj_len-1);
}

int iLQR_CppAD::getTrajLen()
{
    return this->traj_len_;
}

void iLQR_CppAD::setDt(double dt)
{
    this->dt_ = dt;
}

void iLQR_CppAD::setMaxIter(int max_iter) // if you don't set the max_iter, the default value is 0, which means no limit
{
    this->max_iter_ = max_iter;
}

void iLQR_CppAD::setConveregenceThresh(double conv_thresh)
{
    this->conv_thresh_ = conv_thresh;
}

void iLQR_CppAD::iLQR_Forward()
{
    states_traj_.clear(); // clear the states trajectory
    states_traj_.push_back(init_state_); // push the initial state into the states trajectory
    Eigen::VectorXd u;
    Eigen::VectorXd x;
    Eigen::VectorXd x_next;
    for(int i = 0; i < traj_len_-1; i++)
    {
        u = ctrl_traj_[i]; // control input
        x = states_traj_[i]; // state
        x_next = f_dynamic(x,u); // next state
        states_traj_.push_back(x_next); // push the next state into the states trajectory
    }
}

double iLQR_CppAD::iLQR_Forward_cost(std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> &states_traj_,std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>> &ctrl_traj_)
{
    double _forward_cost_ = 0;
    CPPAD_TEST_VECTOR<double> xu_ad(state_dim_+ctrl_dim_); // create a x and u vector of AD variables
    CPPAD_TEST_VECTOR<double> x_ad(state_dim_); // create a x vector of AD variables
    for(int i = 0; i < traj_len_-1; i++)
    {
        xu_ad[0] = states_traj_[i](0); // assign the values of the states to the x and u AD vector
        xu_ad[1] = states_traj_[i](1);
        xu_ad[2] = states_traj_[i](2);
        xu_ad[3] = states_traj_[i](3);
        xu_ad[4] = ctrl_traj_[i](0);
        _forward_cost_ += (stage_cost_ad_.Forward(0,xu_ad).data()[0]);
    }
    x_ad[0] = states_traj_[traj_len_-1](0); // assign the values of the last state to the x AD vector
    x_ad[1] = states_traj_[traj_len_-1](1);
    x_ad[2] = states_traj_[traj_len_-1](2);
    x_ad[3] = states_traj_[traj_len_-1](3);
    _forward_cost_ += (terminal_cost_ad_.Forward(0,x_ad).data()[0]); // calculate the terminal cost
    return _forward_cost_;
}

void iLQR_CppAD::iLQR_Backward()
{
    delta_J = 0.0;
    // calulate the terminal cost P_N and p_N first
    CPPAD_TEST_VECTOR<double> x_K_ad_(state_dim_); // create a x_N vector of AD variables
    for(int i = 0; i < state_dim_; i++) // assign the values of the last state to the x_N AD vector
    {
        x_K_ad_[i] = states_traj_[traj_len_-1](i);
    }
    CPPAD_TEST_VECTOR<double> Lowercase_p_k_ad_ = terminal_cost_ad_.Jacobian(x_K_ad_); // calculate the Jacobian of the terminal cost
    CPPAD_TEST_VECTOR<double> w(1); // create a weight vector, only have one element
    w[0] = 1.0;
    CPPAD_TEST_VECTOR<double> Capital_P_k_ad_ = terminal_cost_ad_.Hessian(x_K_ad_,w); // calculate the Hessian of the terminal cost
    for(int i = 0; i < state_dim_; i++) // assign the values of the terminal cost to the Capital_P_kp1_ and Lowercase_p_kp1_ variables
    {
        Lowercase_p_kp1_(i) = Lowercase_p_k_ad_[i];
        for(int j = 0; j < state_dim_; j++)
        {
            Capital_P_kp1_(i,j) = Capital_P_k_ad_[i*state_dim_+j];
        }
    }
    // calculate the feedback gain matrix K_iLQR_ and the adjustment to the control trajectory d_iLQR_
    CPPAD_TEST_VECTOR<double> xu_K_ad_(state_dim_ + ctrl_dim_); // create a x and u vector of AD variables
    d_iLQR_.clear(); // clear the d_iLQR_ vector
    K_iLQR_.clear(); // clear the K_iLQR_ vector
    for(int backward_times = traj_len_-2; backward_times>=0 ; backward_times--)
    {
        for(int i = 0; i < state_dim_; i++) // assign the values of the last state to the x_N AD vector
        {
            xu_K_ad_[i] = states_traj_[backward_times](i);
        }
        for(int i = 0; i < ctrl_dim_; i++) // assign the values of the last state to the x_N AD vector
        {
            xu_K_ad_[state_dim_+i] = ctrl_traj_[backward_times](i); // ctrl_traj_ has one element less than states_traj_
        }
        CPPAD_TEST_VECTOR<double> stage_cost_jocabian_ad = stage_cost_ad_.Jacobian(xu_K_ad_); // calculate the Jacobian of the stage cost
        Eigen::Vector<double,Eigen::Dynamic> jocabian_stage_cost_x(state_dim_);
        Eigen::Vector<double,Eigen::Dynamic> jocabian_stage_cost_u(ctrl_dim_);
        for(int i = 0; i < state_dim_; i++) // assign the values of the stage cost Jacobian to the gradient_stage_cost_x variable
        {
            jocabian_stage_cost_x(i) = stage_cost_jocabian_ad[i];
        }
        for(int i = 0; i < ctrl_dim_; i++) // assign the values of the stage cost Jacobian to the gradient_stage_cost_u variable
        {
            jocabian_stage_cost_u(i) = stage_cost_jocabian_ad[state_dim_+i];
        }
        CPPAD_TEST_VECTOR<double> stage_cost_hessian_ad = stage_cost_ad_.Hessian(xu_K_ad_,w); // calculate the Hessian of the stage cost
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Hessian_stage_cost(state_dim_+ctrl_dim_,state_dim_+ctrl_dim_); // create a Hessian_stage_cost matrix
        Eigen::MatrixXd hessian_stage_cost_x;
        Eigen::MatrixXd hessian_stage_cost_u;
        Eigen::MatrixXd hessian_stage_cost_xu;
        Eigen::MatrixXd hessian_stage_cost_ux;
        for(int i=0; i < state_dim_+ctrl_dim_; i++) // assign the values of the stage cost Hessian to the Hessian_stage_cost matrix
        {
            for(int j = 0; j< state_dim_+ctrl_dim_; j++)
            {
                Hessian_stage_cost(i,j) = stage_cost_hessian_ad[i*(state_dim_+ctrl_dim_)+j];
            }
        }
        hessian_stage_cost_x = Hessian_stage_cost.block(0,0,state_dim_,state_dim_);
        hessian_stage_cost_u = Hessian_stage_cost.block(state_dim_,state_dim_,ctrl_dim_,ctrl_dim_);
        hessian_stage_cost_xu = Hessian_stage_cost.block(0,state_dim_,state_dim_,ctrl_dim_);
        hessian_stage_cost_ux = Hessian_stage_cost.block(state_dim_,0,ctrl_dim_,state_dim_);
        CPPAD_TEST_VECTOR<double> f_jocabian_ad = f_dynamic_ad_.Jacobian(xu_K_ad_); // calculate the Jacobian of the dynamics
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> jocabian_f(state_dim_,state_dim_+ctrl_dim_);
        Eigen::MatrixXd jocabian_f_x;
        Eigen::MatrixXd jocabian_f_u;
        for(int i = 0; i < state_dim_; i++) // assign the values of the dynamics Jacobian to the jocabian_f matrix
        {
            for(int j = 0; j < state_dim_+ctrl_dim_; j++)
            {
                jocabian_f(i,j) = f_jocabian_ad[i*(state_dim_+ctrl_dim_)+j];
            }
        }
        jocabian_f_x = jocabian_f.leftCols(state_dim_);
        jocabian_f_u = jocabian_f.rightCols(ctrl_dim_);
        Eigen::VectorXd g_x = jocabian_stage_cost_x + jocabian_f_x.transpose()*Lowercase_p_kp1_; // calculate the gradient of the cost function with respect to x
        Eigen::VectorXd g_u = jocabian_stage_cost_u + jocabian_f_u.transpose()*Lowercase_p_kp1_;
        Eigen::MatrixXd G_xx = hessian_stage_cost_x + jocabian_f_x.transpose()*Capital_P_kp1_*jocabian_f_x; // calculate the Hessian of the cost function with respect to x
        Eigen::MatrixXd G_uu = hessian_stage_cost_u + jocabian_f_u.transpose()*Capital_P_kp1_*jocabian_f_u;
        Eigen::MatrixXd G_xu = hessian_stage_cost_xu + jocabian_f_x.transpose()*Capital_P_kp1_*jocabian_f_u; // calculate the Hessian of the cost function with respect to x and u
        Eigen::MatrixXd G_ux = hessian_stage_cost_ux + jocabian_f_u.transpose()*Capital_P_kp1_*jocabian_f_x;
        this->check_Posdef_Condnum(G_uu); // check if the matrix is positive definite and the condition number
        Eigen::VectorXd d = G_uu.inverse()*g_u; // calculate the adjustment to the control trajectory
        Eigen::MatrixXd K = G_uu.inverse()*G_ux; // calculate the feedback gain matrix
        Capital_P_kp1_ = G_xx + K.transpose()*G_uu*K - G_xu*K - K.transpose()*G_ux;
        Lowercase_p_kp1_ = g_x - K.transpose()*g_u + K.transpose()*G_uu*d - G_xu*d;
        d_iLQR_.push_back(d); // push the adjustment to the control trajectory into the d_iLQR_ vector
        K_iLQR_.push_back(K); // push the feedback gain matrix into the K_iLQR_ vector
        delta_J += (g_u.transpose()*d)(0); // calculate the change in cost
    }
    std::reverse(d_iLQR_.begin(),d_iLQR_.end()); // reverse the d_iLQR_ vector
    std::reverse(K_iLQR_.begin(),K_iLQR_.end()); // reverse the K_iLQR_ vector
}

void iLQR_CppAD::check_Posdef_Condnum(Eigen::MatrixXd &G_uu)
{
    // check if the matrix G_uu is positive definite
    Eigen::LLT<Eigen::MatrixXd> LLTofG_uu; // using LLT decomposition to check if the matrix is positive definite
    while(true)
    {
        LLTofG_uu.compute(G_uu);
        // std::cout <<"G_uu" << G_uu << std::endl;
        if(LLTofG_uu.info() == Eigen::NumericalIssue) // if the matrix is not positive definite
        {
            G_uu = G_uu + 0.1*Eigen::MatrixXd::Identity(G_uu.rows(),G_uu.cols()); // add a small value to the diagonal of the matrix
        }
        else
        {
            break;
        }
    }
    // check the condition number of the matrix
}

void iLQR_CppAD::iLQR_Line_Search()
{
    line_search_alpha_ = 1.0;
    Eigen::VectorXd x;
    while(true)
    {
        line_search_states_traj_.clear();
        line_search_ctrl_traj_.clear();
        x = states_traj_[0];
        line_search_states_traj_.push_back(x);
        for(int i = 0; i < traj_len_-1; i++)
        {
            Eigen::VectorXd u = ctrl_traj_[i] - line_search_alpha_*d_iLQR_[i] - (K_iLQR_[i]*(x - states_traj_[i])); // update the control trajectory
            line_search_ctrl_traj_.push_back(u);
            x=f_dynamic(line_search_states_traj_[i],line_search_ctrl_traj_[i]); // update the states trajectory
            line_search_states_traj_.push_back(x); // update the states trajectory
        }
        line_search_forward_cost_ = iLQR_Forward_cost(line_search_states_traj_,line_search_ctrl_traj_); // calculate the forward cost for the line search
        if((line_search_forward_cost_ < forward_cost_ - line_search_b*line_search_alpha_*delta_J && !std::isnan(line_search_forward_cost_)))
        {
            break;
        }
        else
        {
            line_search_alpha_ = line_search_c*line_search_alpha_;
        }
    }
    forward_cost_ = line_search_forward_cost_;
    states_traj_ = line_search_states_traj_;
    ctrl_traj_ = line_search_ctrl_traj_;
}

bool iLQR_CppAD::detectConvergence()
{
    double max_norm = 0;
    // to be implemented
    for(int i = 0; i < traj_len_-1; i++)
    {
        for(int j = 0; j < ctrl_dim_; j++)
        {
            if(max_norm < abs(d_iLQR_[i](j)))
            {
                max_norm = abs(d_iLQR_[i](j));
            }
        }
    }
    if(max_norm < conv_thresh_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void iLQR_CppAD::iLQR_Optimize()
{
    std::cout << "||||  Start iLQR Optimization  ||||" << std::endl;
    int iter = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    iLQR_Forward();
    forward_cost_ = iLQR_Forward_cost(states_traj_,ctrl_traj_);
    while(max_iter_ == 0 || iter <= max_iter_)
    {
        iLQR_Backward();
        iLQR_Line_Search();
        if(detectConvergence())
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << std::endl << "This is iLQR solver, version 0.0.1, referenced to CMU 16-745 course" << std::endl << std::endl;
            std::cout << "......................iLQR Converged !!!!!........................." << std::endl;
            std::cout << "iLQR has converged at iteration...................................:" << iter << std::endl;
            std::cout << "The total time used for the optimization is.......................:" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count() << " ms" << std::endl << std::endl;
            std::cout << "The final control trajectory is stored in.........................:" << "iLQR_CppAD::ctrl_traj_" <<std::endl;
            std::cout << "The final feedback gain matrix is stored in.......................:" << "iLQR_CppAD::K_iLQR_" << std::endl;
            std::cout << "The final Trajectory states are stored in.........................:" << "iLQR_CppAD::states_traj_" << std::endl;
            std::cout << "traj end state.................: ";
            for(int i = 0;i <state_dim_;i++)
            {
                std::cout << states_traj_[traj_len_-1](i) << " ";
            }
            std::cout << std::endl;
            std::cout << "..................................................................." << std::endl;
            break;
        }
        iter++;
        if(iter%20 == 0)
        {
            std::cout << "The current iteration is....: " << iter << "   " << "The current cost is....: " << forward_cost_ << std::endl;
        }
    }
    if(max_iter_ != 0 && iter >= max_iter_)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << std::endl << "This is iLQR solver, version 0.0.1, referenced to CMU 16-745 course." << std::endl << std::endl;
        std::cout << ".......iLQR Not Converged within the limited iterations !!!!!......" << std::endl;
        std::cout << "iLQR has reached the maximum iteration............................:" << max_iter_ << std::endl;
        std::cout << "The total time used for the optimization is.......................:" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count() << " ms" << std::endl << std::endl;
        std::cout << "The final control trajectory is stored in.........................:" << "iLQR_CppAD::ctrl_traj_" <<std::endl;
        std::cout << "The final feedback gain matrix is stored in.......................:" << "iLQR_CppAD::K_iLQR_" << std::endl;
        std::cout << "The final Trajectory states are stored in.........................:" << "iLQR_CppAD::states_traj_" << std::endl;
        std::cout << "traj end state.................: ";
        for(int i = 0;i <state_dim_;i++)
        {
            std::cout << states_traj_[traj_len_-1](i) << " ";
        }
        std::cout << std::endl;
        std::cout << "..................................................................." << std::endl;
    }
}






