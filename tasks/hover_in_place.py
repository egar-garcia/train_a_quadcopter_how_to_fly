import math
import numpy as np
from physics_sim import PhysicsSim

class HoverInPlace():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, reward_function, init_pose=None, runtime=5., rising_meters=10.0):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        self.reward_function = reward_function
        self.init_pose = init_pose if init_pose is not None else np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

        # Simulation
        self.sim = PhysicsSim(self.init_pose, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal, same as the initial position
        self.target_pos = np.array(self.init_pose[:3])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        return self.reward_function(self)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state