# RandUP-RRT
Accompanying repository of _Robust-RRT: Probabilistically-Complete Motion Planning for Uncertain Nonlinear Systems_ 

## Installation
### Minimum requirement
1. Python 3.7 or above

### Setting up a virtual environment (optional)
1. Run `python3 -m venv $ENV_PATH`
2. Activate the virtual environment with `source $ENV_PATH/bin/activate`

### Setting up `randUP_RRT`
1. Clone this repository from https://github.com/StanfordASL/randUP_RRT.git
4. From `randUP_RRT`, run `pip install .`

## Running Provided Examples
If `randUP_RRT` was set up in a virtual environment, the virtual environment must be active
in the terminal used to run the examples.

For all planning examples, use the optional argument `--num_particles=1` to switch to RRT planning.
Use `--num_particles=i` for planning with `i` RandUP particles. By default, planning is done with RandUP-RRT 
with the default number of particles.
### Nonlinear Quadrotor
- Run `python3 planning/quadrotor_examples/plan_quadrotor.py` to plan and produce Figure 3.

- Run `python3 planning/quadrotor_examples/generate_quadrotor_statistics.py` to produce Table 1. 
This script does not take in `num_particles`.

### Planar Pusher
- Run `python3 planning/planar_pusher_examples/feedback_pusher.py` to plan and visualize the planar pusher with PyBullet.

### Hybrid Jumping Robot
- Run `python3 planning/hybrid_integrator_examples/plan_hybrid_integrator.py` to plan and produce Figure 5. 

## Repository Structure
- `hybrid_integrator`, `planar_pusher`, `quadrotor_planner` define the respective physical systems
- `planning` contains all planning-related code
  - `*_examples` contain the scripts for running examples
  - `randup_rrt.py` implements RandUP-RRT
