import pandas as pd
import matplotlib.pyplot as plt
policy = []
df = pd.read_csv('final.csv')
print(df)

delta_t = 2
next_loca = 0
next_velo = 0
next_time = 0
x_plt = []
v_plt = []
a_plt = []
t_plt = []

for i in range(10):
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