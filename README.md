# manipulator dynamics 🦾
A library for generating the kinematics and dynamics of floating base or fixed base robotic arm as symbolic expressions for automatic differentiations. 
<!-- ![alt text]() -->
<img src="./resources/dory.jpg" width="420"/>

## Todo/Implementation status
- [x] [forward kinematics(FK)]()
- [ ] [inverse kinematics (IK)]()
- [x] [workspace Analysis (sampling-based AABB + point cloud) ]()
- [x] [forward dynamics]()
- [x] [inverse dynamics (Lagrange–Euler)]()
- [x] [Energy and Lagrangian Terms]()
- [x] [System Identification helpers (energy-based regressors)]()
- [x] [jit support]()
- [ ] [gpu support]()

For usage examples of Diff_UVMS, see [Jupyter notebook](https://github.com/edxmorgan/underwater-manipulator-dynamics/tree/lagrange-euler/usage).


## References
- Roy Featherstone and Kluwer Academic Publishers. 1987. Robot Dynamics Algorithm. Kluwer Academic Publishers, USA.
- M. W. Spong, S. Hutchinson and M. Vidyasagar, “Robot Modeling and Control,” Wiley, New York, 2006.
- Bruno Siciliano, Lorenzo Sciavicco, Luigi Villani, and Giuseppe Oriolo. 2010. Robotics: Modelling, Planning and Control. Springer Publishing Company, Incorporated.

## Caution ⚠️  
This project is still experimental. While the core functionalities have been implemented and tested to some extent, further validation and testing are required. Use with care, especially for safety-critical applications. Contributions and feedback are welcome!  

