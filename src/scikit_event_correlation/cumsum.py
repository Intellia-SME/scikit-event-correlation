import pandas as pd
import numpy as np

data = pd.read_csv("data/DATASET_CMA_CGM_NERVAL_5min.csv")
data = data.iloc[:,:-1] # afairei tin teleutaia stili TIMESTAMP
columns = data.columns
data = np.array(data)

# CUSUM

# CALCULATE THE AVERAGE OF EACH STREAM AS TARGET VALUE
average = np.zeros(len(data[0]))
for i in range(0,10):
	for col in range(0, len(data[0])):
		average[col] += data[i, col]

average = average/10
# CONSTANTS
# m = average
m = abs(average)
# print(m)
kpos = kneg = m/2
thpos = thneg = 2*m

# f = open('test','a')

# INITIALIZE VECTORS
P = np.zeros(len(data[0]))	#positive changes
N = np.zeros(len(data[0]))	#negative changes
s = np.zeros((len(data), len(data[0])))

for t in range(0,len(data)):
	for col in range(0,len(data[0])):

		spos = 0	#positive signal
		sneg = 0	#negative signal
		P[col] = max(0, abs(data[t,col]) - m[col] - kpos[col] + P[col])
		N[col] = min(0, abs(data[t,col]) - m[col] + kneg[col] + N[col])
		# f.write("iter "+ str(t)+" "+str(col)+" "+str(P[col])+" "+str(N[col])+" threshpos "+str(thpos[col])+" threshneg "+str(thneg[col])+'\n')
		if (P[col] > thpos[col]):
			spos = 1
			P[col] = N[col] = 0
		if (N[col] < -thneg[col]):
			sneg = 1
			P[col] = N[col] = 0
		s[t,col] = spos or sneg

pd.DataFrame(s, columns=columns).to_csv('data/cumsum_script.csv')

# f.close()
# np.savetxt('../eventVectors/cusum/cusumEventVector.csv', s, fmt='%d', delimiter='')