max_avg = [2207.20391204, 3012.06550856 ,3163.75601451]
avg_true_mean = [1.53955789, 1.55619762, 1.54717906]

max_avg = [avg / 2000 for avg in max_avg]

print("max_avg_reward: ", max_avg)
print("Average True Mean of Optimal Action: ", avg_true_mean)
