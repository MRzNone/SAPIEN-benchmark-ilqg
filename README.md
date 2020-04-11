# Benchmark_stable_human

A benchmark for [Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf) on SAPIEN.

# TODO
- Implement in a toy env (gym) :heavy_check_mark:
- Setup (cartpole) environment in SAPIEN
  - able to dummy simulate :heavy_check_mark:
  - set up workflow :heavy_check_mark:
  - wrap control for ease of use :heavy_check_mark:
- implement i-lqg :heavy_check_mark:
- Solving Ant task
  - Handcraft forward kinematics
  - Adjust fu, fx (final cost derivative respect to x u)
- optimize?
- repeat for
  - swimmer
  - humanoid

# LOG

HUMANOID
  - Too computational expensive, pivoting to simpler task

ANT
  - First using world coordinate as state, works really bad
    - Switched to generalised coordinate
    - Note on SAPIEN generalised coordinate
      - naming convetion: q[something].  e.g. qpos, qvel
  - Second order derivative of final cost function requires the term d2x/dq2, x is cartesian, q is general coord
    - Writing my own forward kinematics.  f: q -> x (local cartesian coord respect to robot base)
