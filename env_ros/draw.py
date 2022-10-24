from cProfile import label
import json
from matplotlib import pyplot as plt


with open("../log.json") as f:
    data = json.load(f)
real_traj = data['real_traj']

plt.figure(figsize=(3, 8))

x, y, z = zip(*real_traj)
plt.plot(y, x, label='real')

# planed_trajs = data['planed_trajs']
# for traj in planed_trajs:
#     x, y, z = zip(*traj[:15])
#     plt.plot(y, x, ':', lw=1)
# plt.xlim((-2, 2))
plt.ylim((-1, 36))
# plt.legend(loc ="lower right")
plt.savefig("ours.png", dpi=200)
# plt.show()
