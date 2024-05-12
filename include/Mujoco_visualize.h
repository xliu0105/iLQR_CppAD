#include <mujoco.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <Eigen/Dense>


mjModel *m = NULL;
mjData *d = NULL;
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
mjvPerturb pert;


void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);

