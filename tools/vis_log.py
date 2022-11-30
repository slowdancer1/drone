import json
import matplotlib.pyplot as plt

data = []
with open('log.txt') as f:
    for line in f:
        data.append(json.loads(line))

r, p, y, _r, _p, _y, _c = zip(*data)
plt.plot(r, label='real')
plt.plot(_r, label='ctl')
plt.show()
