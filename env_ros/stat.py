import sys
import numpy as np

secs = np.fromfile(sys.argv[1], sep='\n')

succ = np.mean(secs < 30)
secs = np.mean(secs[secs < 30])
print(35/secs, succ)
