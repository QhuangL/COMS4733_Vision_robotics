import sim
import pybullet as p
import numpy as np
import math

MAX_ITERS = 10000
delta_q = 0.5


def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= TODO: Problem 3 ========
    # Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)
    V = {tuple(q_init)}
    E = set()

    for i in range(MAX_ITERS):
        q_rand = SemiRandomSample(steer_goal_p, q_goal)
        q_nearest = Nearest(V, E, q_rand)
        q_new = Steer(q_nearest, q_rand, delta_q)

        if ObstacleFree(q_nearest, q_new, env):
            V = Union(V, {tuple(q_new)})
            E = Union(E, {tuple((tuple(q_nearest), tuple(q_new)))})

            visualize_path(q_nearest, q_new, env)

            if (Distance(q_new, q_goal) < delta_q):
                V = Union(V, {tuple(q_goal)})
                E = Union(E, {tuple((tuple(q_new), tuple(q_goal)))})
                # print("Finding path...")
                path = FindPath(q_init, q_goal, V, E)
                # print("Found path...")
                return path

    return None


def Union(old, new):
    return old.union(new)


def SemiRandomSample(steer_goal_q, q_goal):
    if np.random.random() < steer_goal_q:
        return q_goal
    else:
        q_rand = 2 * math.pi * np.random.random_sample((6)) - math.pi
        return q_rand


def Steer(q_nearest, q_rand, delta_q):
    q_nearest = np.array(q_nearest)
    q_rand = np.array(q_rand)
    if Distance(q_nearest, q_rand) <= delta_q:
        q_new = q_rand
    else:
        dir_vec = ((q_rand - q_nearest) / np.linalg.norm(np.array(q_rand) - np.array(q_nearest))) * delta_q
        q_new = q_nearest + dir_vec
    return tuple(q_new)


def Nearest(V, E, q_rand):
    nearest_dist = float('inf')
    nearest_vertex = None
    for i in V:
        dist = Distance(i, q_rand)
        if dist < nearest_dist:
            nearest_vertex = i
            nearest_dist = dist
    return nearest_vertex


def ObstacleFree(q_nearest, q_new, env):
    collision = env.check_collision(q_new)
    if collision:
        return False
    else:
        return True


def Distance(q1, q2):
    return np.linalg.norm(np.array(q1) - np.array(q2))


def FindPath(start, goal, V, E):
    path = []
    vertex = tuple(goal)

    while True:
        for i in E:
            if vertex == tuple(start):
                path.append(vertex)
                break
            if i[1] == vertex:
                path.append(vertex)
                vertex = i[0]
        if vertex == tuple(start):
            break

    path.reverse()
    return path


def execute_path(path_conf, env):
    # ========= TODO: Problem 3 ========
    # 1. Execute the path while visualizing the location of joint 5 
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, step the simulation, close gripper)
    # 3. Return the robot to original location by retracing the path 
    markers = []
    for joint_state in path_conf:
        env.move_joints(joint_state, speed=0.1)

        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
        markers.append(
            sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[1, 0, 0, 0.8]))

    print("Path executed. Dropping the object")

    env.open_gripper()
    env.step_simulation(3)
    env.close_gripper()

    for joint_state in list(reversed(path_conf)):
        env.move_joints(joint_state, speed=0.1)

    while markers:
        del markers[-1]

    return None
