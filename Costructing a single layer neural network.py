import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl


# Load input data
text = np.loadtxt('Getting Mouse data to an araduino board.txt')
# Separate it into datapoints and labels
data = text[: len(text)-1:2]
labels = text[: 2:]

# Plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')
