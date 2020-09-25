import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- definition of basic setting-------------------- #
X_end = 100
V_max = 10
T_goal = 20
max_deceleration = -5
max_acceleration = 5
delta_x = delta_v = N_x = N_v = delta_t = N_t = 0

# ----------------------- declare of States, policy, list of velocity, time and location, Action, transition-----------#
Location_list = []
Velocity_list = []
Time_list = []
States = pd.DataFrame(columns=['Location', 'Velocity', 'Time'])
policy = []
PossibleAction = []
df = pd.DataFrame(columns=['current_Location', 'current_Velocity', 'current_time', 'action', 'Prob', 'next_Location',
                           'next_Velocity', 'next_time'])


def discrete_setting(x_num, v_num, t_num):
    """
    Choose the number to discrete location and velocity and time
    :param t_num: for time
    :param x_num: for location
    :param v_num: for velocity
    :return:
    """
    # rewrite N_x, N_v, delta_x, delta_v
    global N_x, N_v, delta_x, delta_v, delta_t, N_t
    N_x = x_num
    N_v = v_num
    N_t = t_num
    delta_x = X_end / N_x
    delta_v = V_max / N_v
    delta_t = T_goal / N_t
    for m in range(N_x + 1):
        Location_list.append(m * delta_x)
    for n in range((N_v + 1)):
        Velocity_list.append(n * delta_v)

    return N_x, N_v, N_t, delta_x, delta_v, delta_t


def build_state():
    global States
    for m in range(N_x + 1):
        location = m * delta_x
        for n in range(N_v + 1):
            velocity = n * delta_v
            for p in range(N_t + 1):
                time = p * delta_t
                States = States.append(
                    {'Location': location, 'Velocity': velocity, 'Time': time},
                    ignore_index=True)
    return States


'''
def chooseAction():
    lenS = len(States)
    lenV = len(Velocity_list)
    for i in range(lenS):
        curr_v = States['Velocity'][i]
        curr_t = States['Time'][i]
        tmp_actions = []
        if curr_t == T_goal:
            PossibleAction.append(tmp_actions)
        else:
            for j in range(lenV):
                if curr_v + max_deceleration * delta_t <= Velocity_list[j] <= curr_v + max_acceleration * delta_t:
                    tmpv = Velocity_list[j]
                    tmp_a = (tmpv - curr_v) / delta_t
                    tmp_x = curr_v * delta_t + 0.5 * tmp_a * delta_t * delta_t
                    if 0 <= tmp_x <= X_end + 0.1 * delta_x:
                        tmp_actions.append(tmp_a)
            PossibleAction.append(tmp_actions)
    return PossibleAction
'''
discrete_setting(20, 10, 10)
build_state()
lenS = len(States)
lenV = len(Velocity_list)
lenL = len(Location_list)
# -----------------build dataframe for transition kernel & set original policy-----------------------------------------#
for i in range(lenS):
    cur_l = States['Location'][i]
    cur_v = States['Velocity'][i]
    cur_t = States['Time'][i]

    if cur_t != T_goal:
        Next_time = cur_t + delta_t
        for j in range(lenV):
            # check if Velocity_list[j] can be reached
            if max(cur_v + max_deceleration * delta_t, 0) <= Velocity_list[j] <= cur_v + max_acceleration * max_acceleration:
                action = (Velocity_list[j] - cur_v) / delta_t
                if max_deceleration <= action <= max_acceleration:
                    x_tmp = cur_l + cur_v * delta_t + 0.5 * action * delta_t * delta_t
                    if 0 <= x_tmp <= X_end + 0.1 * delta_x:
                        tmpv = Velocity_list[j]
                        tmp_location = X_end
                        for k in range(lenL):
                            if Location_list[k] != cur_l:
                                if abs(x_tmp - Location_list[k]) < abs(x_tmp - tmp_location) :
                                    tmp_location = Location_list[k]
                            # print(tmp_location)

                        df = df.append({'current_Location': cur_l, 'current_Velocity': cur_v, 'current_time': cur_t,
                                        'action': action, 'Prob': 0, 'next_Location': tmp_location,
                                        'next_Velocity': tmpv,
                                        'next_time': Next_time}, ignore_index=True)

print(df)


def tract_or_braking(action, v):
    # mass of train : m kg
    m = 1000
    # Wind resistance : kv^2
    k = 1
    w_res = k * pow(v, 2)
    if action < 0:
        force = abs(m * action) - w_res
    else:
        force = m * action + w_res
    return force


def reward(velocity, action):
    """

    :param velocity:
    :param action:
    :return:
    """
   # if velocity == 0 and action == 0:
    #    return -
    #else:
    energy = -tract_or_braking(action, velocity) * delta_x
    return energy


def original_policy():
    for s in range(lenS):
        c_location = States['Location'][s]
        c_velocity = States['Velocity'][s]
        c_time = States['Time'][s]
        Next_action_List = df[(df.current_Location == c_location) & (df.current_Velocity == c_velocity)
                              & (df.current_time == c_time)]['action'].values
        list_num = len(Next_action_List)
        if list_num == 0:
            df.loc[(df.current_Location == c_location) & (df.current_Velocity == c_velocity) &
                   (df.current_time == c_time), 'Prob'] = 0
        else:
            df.loc[(df.current_Location == c_location) & (df.current_Velocity == c_velocity) &
                   (df.current_time == c_time), 'Prob'] = 1 / list_num


# chooseAction()
# print(States)
# print(PossibleAction)


# print(Location_list)

original_policy()
States.to_csv(('State.csv'))
df.to_csv('check.csv')
# ----------------------------policy evaluation--------------------------#
V = np.zeros(lenS)


def policy_evaluation(discount_factor, theta):
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(lenS):
            v = 0
            # Current Location and Velocity
            c_location = States['Location'][s]
            c_velocity = States['Velocity'][s]
            c_time = States['Time'][s]
            # Look at the possible next action in policy
            Next_action_List = df[(df.current_Location == c_location) & (df.current_Velocity == c_velocity)
                                  & (df.current_time == c_time)]['action'].values
            # print(Next_action_List)
            list_num = len(Next_action_List)
            if c_time == T_goal:
                if c_velocity == 0 and c_location == X_end:
                    v = 99999
                else:
                    v = -99999
            else:
                for i in range(list_num):
                    current_action = Next_action_List[i]

                    # For each action, get the possible next states
                    n_location = df[(df.current_Location == c_location) & (df.current_Velocity == c_velocity)
                                    & (df.current_time == c_time) & (df.action == current_action)][
                        'next_Location'].values
                    n_location = n_location[0]
                    n_velocity = df[(df.current_Location == c_location) & (df.current_Velocity == c_velocity)
                                    & (df.current_time == c_time) & (df.action == current_action)][
                        'next_Velocity'].values
                    n_velocity = n_velocity[0]
                    proba = df[(df.current_Location == c_location) & (df.current_Velocity == c_velocity)
                               & (df.current_time == c_time) & (df.action == current_action)]['Prob'].values
                    proba = proba[0]
                    # Locate the Next States
                    ntime = c_time + delta_t

                    idx = States[(States.Location == n_location) & (States.Velocity == n_velocity) & (
                            States.Time == ntime)].index.tolist()

                    # Calculate the expected value
                    tmpv = proba * (reward(c_velocity, current_action) + discount_factor * V[idx])
                    v = v + tmpv
                    # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return V


def policy_improvement(discount_factor):
    global V

    # Will be set to false if we make any changes to the policy
    policy_stable = True

    def find_best_next_value(location, velocity, time, V):
        # find current state index
        idx_curr = States[(States.Location == location) & (States.Velocity == velocity) & (
                States.Time == time)].index.tolist()
        # find possible action list
        Poss_act_List = df[(df.current_Location == location) & (df.current_Velocity == velocity)
                           & (df.current_time == time)]['action'].values
        len_Pa = len(Poss_act_List)
        tmpbest = -np.inf
        best_choice = -1
        for i in range(len_Pa):
            action = Poss_act_List[i]
            # For action, get the possible next states
            n_loca = df[(df.current_Location == location) & (df.current_Velocity == velocity)
                        & (df.current_time == time) & (df.action == action)][
                'next_Location'].values
            n_loca = n_loca[0]
            n_velo = df[(df.current_Location == location) & (df.current_Velocity == velocity)
                        & (df.current_time == time) & (df.action == action)][
                'next_Velocity'].values
            n_velo = n_velo[0]
            ntime = time + delta_t
            # get index of next value
            idx_next = States[(States.Location == n_loca) & (States.Velocity == n_velo) & (
                    States.Time == ntime)].index.tolist()
            # calculate value
            v = reward(velocity, action) + discount_factor * V[idx_next]
            if v > tmpbest:
                tmpbest = v
                best_choice = Poss_act_List[i]
        return best_choice

    # For each state...
    for s in range(lenS):
        # Current Location and Velocity
        c_location = States['Location'][s]

        c_velocity = States['Velocity'][s]

        c_time = States['Time'][s]

        # The best action we would take under the current policy
        idx_States = States[(States.Location == c_location) & (States.Velocity == c_velocity) & (
                States.Time == c_time)].index.tolist()
        # idx_States = States.index((c_location, c_velocity))
        chosen_a = V[idx_States]
        # print(chosen_a)
        # Find the best action by one-step lookahead
        best_a = find_best_next_value(c_location, c_velocity, c_time, V)
        # Greedily update the policy
        if chosen_a != best_a:
            policy_stable = False
            df.loc[(df.current_Location == c_location) & (df.current_Velocity == c_velocity) & (
                    df.current_time == c_time), 'Prob'] = 0
            df.loc[
                (df.current_Location == c_location) & (df.current_Velocity == c_velocity) & (df.current_time == c_time)
                & (df.action == best_a), 'Prob'] = 1

    # If the policy is stable we've found an optimal policy. Return it
    if policy_stable:
        return policy, V


policy_evaluation(1, 0.01)
policy_improvement(1)
print(V)

next_loca = 0
next_velo = 0
next_time = 0
x_plt = []
v_plt = []
a_plt = []
t_plt = []
for i in range(N_t):
    policy.append((next_loca, next_velo, next_time))
    next_loca = \
        df[(df.current_Location == next_loca) & (df.current_Velocity == next_velo) & (df.current_time == next_time)
           & (df.Prob == 1)]['next_Location'].values
    next_loca = next_loca[0]
    next_velo = \
        df[(df.current_Location == next_loca) & (df.current_Velocity == next_velo) & (df.current_time == next_time)
           & (df.Prob == 1)]['next_Velocity'].values
    next_velo = next_velo[0]
    action = df[(df.current_Location == next_loca) & (df.current_Velocity == next_velo) & (df.current_time == next_time)
                & (df.Prob == 1)]['action'].values
    action = action[0]
    next_time = next_time + delta_t

    x_plt.append(next_loca)
    v_plt.append(next_velo)
    a_plt.append(action)
    t_plt.append(next_time)

print(policy)

plt.plot(t_plt, x_plt)
plt.ylabel('location')
plt.xlabel('time')
plt.show()

plt.plot(t_plt, v_plt)
plt.ylabel('velocity')
plt.xlabel('time')
plt.show()

plt.plot(x_plt, v_plt)
plt.xlabel('location')
plt.ylabel('velocity')
plt.show()

plt.plot(t_plt, a_plt)
plt.ylabel('action')
plt.xlabel('time')
plt.show()
