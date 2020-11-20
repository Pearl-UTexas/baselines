from baselines.common import plot_util as pu
results = pu.load_results('log/breakout_1000_cs/full/')

# pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)


import matplotlib.pyplot as plt
import numpy as np
r = results[0]
# plt.plot(np.cumsum(r.monitor.l), r.monitor.r)

# raw epsiode data
# plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))

# logger data
# print(r.progress)
plt.plot(r.progress.TimestepsSoFar, r.progress.EpRewMean)