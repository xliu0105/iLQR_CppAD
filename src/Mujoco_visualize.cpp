#include <Mujoco_visualize.h>

// keyboard callback
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




std::vector<Eigen::MatrixXd> split_data(std::string &line_ctrl, std::string &line_states, std::string &line_K)
{
    std::vector<Eigen::MatrixXd> data;
    std::stringstream ss_ctrl(line_ctrl), ss_states(line_states), ss_K(line_K);
    std::string token;
    Eigen::MatrixXd ctrl, states, K;
    ctrl.resize(1,1); states.resize(4,1); K.resize(1,4);
    ctrl(0,0) = std::stod(line_ctrl);
    int i=0;
    while(std::getline(ss_states, token, ','))
    {
        states(i,0) = std::stod(token);
        i++;
    }
    i=0;
    while(std::getline(ss_K, token, ','))
    {
        K(0,i) = std::stod(token);
        i++;
    }
    data.push_back(ctrl); data.push_back(states); data.push_back(K);
    return data;
}

int main()
{
    // --- Load the model --- //
    char error[1000];
    m = mj_loadXML("../env/Acrobot.xml", NULL, error, 1000);
    if (m) {
        std::cout << "load model successfully." << std::endl;
    }
    d = mj_makeData(m);

    // --- init glfw to visualize --- //
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1500, 1000, "Demo", NULL, NULL);
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

    cam.type = mjCAMERA_FREE; // set camera to free mode
    cam.lookat[0] = 0.0; // 观察点的位置
    cam.lookat[1] = 0.0; // 观察点的位置
    cam.lookat[2] = 2.5; // 观察点的位置
    cam.distance = 6.0; // 距离观察点的距离
    cam.azimuth = 90; // 方位角
    cam.elevation = -10.0; // 仰角

    // --- open data files --- //
    std::ifstream ifs_ctrl, ifs_states, ifs_K;
    ifs_ctrl.open("./ctrl.csv"); ifs_states.open("./states.csv"); ifs_K.open("./K.csv");

    //
    std::string line_states, line_ctrl, line_K;
    int freq = 20;
    std::chrono::milliseconds interval(1000/freq);
    auto start = std::chrono::system_clock::now();
    auto next = std::chrono::system_clock::now() + interval;
    
    std::fill(d->ctrl,d->ctrl+m->nu,0.0); // set control to zero
    mj_step(m, d);
    while(std::chrono::system_clock::now() < start + std::chrono::seconds(3))
    {
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);
        std::this_thread::sleep_until(next);
        next += interval;
    }
    while(1)
    {
        if(!ifs_ctrl.eof() && !ifs_states.eof() && !ifs_K.eof())
        {
            std::getline(ifs_ctrl, line_ctrl) && std::getline(ifs_states, line_states) && std::getline(ifs_K, line_K);
        }
        std::vector<Eigen::MatrixXd> data = split_data(line_ctrl, line_states, line_K);
        d->ctrl[0] = (data[0]-data[2]*Eigen::Vector4d(d->qpos[0]-data[1](0),d->qpos[1]-data[1](1),d->qvel[0]-data[1](2),d->qvel[1]-data[1](3)))(0);
        
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

        std::this_thread::sleep_until(next);
        next += interval;
    }
    return 0;
}