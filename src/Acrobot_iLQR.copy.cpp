#include "Mujoco_Acrobot.h"

Mujoco_Acrobot_iLQR::Mujoco_Acrobot_iLQR()
{
    char error[1000]; // error message for mujoco
    m = mj_loadXML("/home/liu_xu/liuxu_Documents/iLQR_CppAD/env/Acrobot.xml", NULL, error, 1000);
    d = mj_makeData(m);
    if (m) {
        std::cout << "load model successfully." << std::endl;
    }
}

Mujoco_Acrobot_iLQR::~Mujoco_Acrobot_iLQR()
{
    mj_deleteData(d);
    mj_deleteModel(m);
}


void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

// Acrobot dynamic equation reference: https://blog.csdn.net/weixin_46536094/article/details/123582939
Eigen::Vector<CppAD::AD<double>,4> Mujoco_Acrobot_iLQR::f_CppAD_(Eigen::Vector<CppAD::AD<double>,4> &ax, Eigen::Vector<CppAD::AD<double>,1> &u)
{
    double lc1 = l1/2.0;
    double lc2 = l2/2.0;
    int THETA1 = 0;
    int THETA2 = 1;
    int THETA1DOT = 2;
    int THETA2DOT = 3;
    Eigen::Vector<CppAD::AD<double>,4> x_dot;
    CppAD::AD<double> d1 = m1*lc1*lc1 + m2*(l1*l1 + lc2*lc2 + 2*l1*lc2*cos(ax[THETA2])) + I1 + I2;
    CppAD::AD<double> d2 = m2*(lc2*lc2 + l1*lc2*cos(ax[THETA2])) + I2;
    CppAD::AD<double> phi2 = m2*lc2*g*cos(ax[THETA1] + ax[THETA2] - M_PI/2);
    CppAD::AD<double> phi1 = -m2*l1*lc2*ax[THETA2DOT]*ax[THETA2DOT]*sin(ax[THETA2]) -
                  2*m2*l1*lc2*ax[THETA2DOT]*ax[THETA1DOT]*sin(ax[THETA2]) +
                  (m1*lc1 + m2*l1)*g*cos(ax[THETA1]-M_PI/2) + phi2;

    // dynamics
    x_dot[THETA1]    = ax[THETA1DOT];
    x_dot[THETA2]    = ax[THETA2DOT];
    x_dot[THETA2DOT] = (u(0) + phi1*d2/d1 - m2*l1*lc2*ax[THETA1DOT]*ax[THETA1DOT]*sin(ax[THETA2]) - phi2)/(m2*lc2*lc2 + I2 - d2*d2/d1);
    x_dot[THETA1DOT] = -(d2*x_dot[THETA2DOT] + phi1)/d1;
    return x_dot;
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
    double lc1 = l1/2.0;
    double lc2 = l2/2.0;
    int THETA1 = 0;
    int THETA2 = 1;
    int THETA1DOT = 2;
    int THETA2DOT = 3;
    double d1 = m1*lc1*lc1 + m2*(l1*l1 + lc2*lc2 + 2*l1*lc2*cos(x(THETA2))) + I1 + I2;
    double d2 = m2*(lc2*lc2 + l1*lc2*cos(x(THETA2))) + I2;
    double phi2 = m2*lc2*g*cos(x(THETA1) + x(THETA2) - M_PI/2);
    double phi1 = -m2*l1*lc2*x(THETA2DOT)*x(THETA2DOT)*sin(x(THETA2)) -
                    2*m2*l1*lc2*x(THETA2DOT)*x(THETA1DOT)*sin(x(THETA2)) +
                    (m1*lc1 + m2*l1)*g*cos(x(THETA1)-M_PI/2) + phi2;

    // dynamics
    Eigen::Vector4d x_dot;
    x_dot(THETA1)    = x(THETA1DOT);
    x_dot(THETA2)    = x(THETA2DOT);
    x_dot(THETA2DOT) = (u(0) + phi1*d2/d1 - m2*l1*lc2*x(THETA1DOT)*x(THETA1DOT)*sin(x(THETA2)) - phi2)/(m2*lc2*lc2 + I2 - d2*d2/d1);
    x_dot(THETA1DOT) = -(d2*x_dot(THETA2DOT) + phi1)/d1;
    return x_dot;
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

void Mujoco_Acrobot_iLQR::Visualize()
{
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1200, 800, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_defaultScene(&scn);
    mjr_makeContext(m, &con, mjFONTSCALE_100);
    mjv_makeScene(m, &scn, 1000);
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    int i =0;
    while(i < traj_len_-1) 
    {
        // mjtNum simstart = d->time;
        std::cout << d->qpos[0] << " " << d->qpos[1] << " " << d->qvel[0] << " " << d->qvel[1]  << std::endl << std::endl;
        d->ctrl[0] = (ctrl_traj_[i]-K_iLQR_[i]*Eigen::Vector4d(d->qpos[0]-states_traj_[i](0),d->qpos[1]-states_traj_[i](1),d->qvel[0]-states_traj_[i](2),d->qvel[1]-states_traj_[i](3)))(0);
        // d->ctrl[0] = ctrl_traj_[i](0);
        mjtNum simstart = d->time;
        mj_step(m, d);
        
        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
        i++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void Mujoco_Acrobot_iLQR::writeData()
{
    std::ofstream file;
    file.open("states.csv");

    for (int i = 0; i < traj_len_; i++)
    {
        file << states_traj_[i](0) << "," << states_traj_[i](1) << "," << states_traj_[i](2) << "," << states_traj_[i](3) << std::endl;
    }
    file.close();
    file.open("ctrl.csv");
    for (int i = 0; i < traj_len_-1; i++)
    {
        file << ctrl_traj_[i](0) << std::endl;
    }
    file.close();
    file.open("K.csv");
    for(int i=0;i<traj_len_-1;i++)
    {
        file << K_iLQR_[i](0,0) << "," << K_iLQR_[i](0,1) << "," << K_iLQR_[i](0,2) << "," << K_iLQR_[i](0,3) << std::endl;
    }
    file.close();
}

int main(int argc, char *argv[])
{
    srand(static_cast<unsigned int>(time(0)));
    Mujoco_Acrobot_iLQR acrobot_iLQR;
    acrobot_iLQR.setInitialState(Eigen::VectorXd::Zero(4));
    acrobot_iLQR.setTrajLen(250);
    acrobot_iLQR.setDt(0.05);
    acrobot_iLQR.setMaxIter(1500);
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
    acrobot_iLQR.writeData();
    acrobot_iLQR.Visualize();
    return 0;
}