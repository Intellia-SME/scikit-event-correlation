import pandas as pd
import numpy as np
import math

data = pd.read_csv("DATASET_CMA_CGM_NERVAL_5min.csv")
data = data.iloc[:,:-1]
columns = data.columns
data = np.array(data)

# SHEWHART

# CONSTANTS
# k = data[0]
k = 1

# INITIALIZE VECTORS
xprev = np.zeros(len(data[0]))
sprev = np.zeros(len(data[0]))
xcurr = np.zeros(len(data[0]))
scurr = np.zeros(len(data[0]))
s = np.zeros((len(data), len(data[0])))

for t in range(0,len(data)):
	for col in range(0,len(data[0])):

		xcurr[col] = xprev[col] + (data[t,col]-xprev[col])/(t+1)
		scurr[col] = math.sqrt( (1/(t+1)) * (t*(sprev[col]**2) + (data[t,col]-xcurr[col])*(data[t,col]-xprev[col]) ))

		UCL = xcurr[col] + k*scurr[col]
		LCL = xcurr[col] - k*scurr[col]

		if ((data[t,col] > UCL) or (data[t,col] < LCL)):
			s[t][col] = 1
		else:
			s[t][col] = 0
		xprev[col] = xcurr[col]
		sprev[col] = scurr[col]

# np.savetxt('../eventVectors/shewhart/shewhartEventVector.csv', s, fmt='%d', delimiter='')

# pd.DataFrame(s, columns=columns).to_csv('data/shewhart_script.csv')