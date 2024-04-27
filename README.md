# README

This package was created by ***Xu Liu***.

`iLQR_CppAD.cpp` is the source code of iLQR and a shared library will be generated.

If you want to use this iLQR library, you must create a class that inherits from iLQR_CppAD, and rewrite four pure virtual functions:

- **f_dynamic_CppAD**: you need to rewrite this function. This function is used to create CppAD function of dynamic equation for automatic differentiation. This CppAD function has been declared in `iLQR_CppAD.h`: `f_dynamic_ad_`, you can use `f_dynamic_ad_.Dependent(x,y)` to construct it.
- **f_dynamic**: you need to rewrite this function. This function is used to create a common dynamic function for rollout. The return value type of this function is: `Eigen::VectorXd`. This function has two parameters: `Eigen::VectorXd &x` and `Eigen::VectorXd &u`.
- **stage_cost_CppAD**: you need to rewrite this function. This function is used to create CppAD function of stage cost equation for automatic differentiation. This CppAD function has been declared in `iLQR_CppAD.h`: `stage_cost_ad_`, you can use `stage_cost_ad_.Dependent(x,y)` to construct it.
- **terminal_cost_CppAD**: you need to rewrite this function. This function is used to create CppAD function of terminal cost equation for automatic differentiation. This CppAD function has been declared in `iLQR_CppAD.h`: `terminal_cost_ad_`, you can use `terminal_cost_ad_.Dependent(x,y)` to construct it.

You can check the structure of the `iLQR_CppAD` in `iLQR_CppAD.h`, you will get all you want to know if you are familiar with c++. I alse write a demo of **Acrobot** using `iLQR_CppAD`, you can check `Mujoco_Acrobot.h` and `Mujoco_Acrobot.cpp` to see how to use `iLQR_CppAD`.

This package has not been optimized and perfected yet, and will continue to be updated in the future. Your criticisms and suggestions are also welcome.