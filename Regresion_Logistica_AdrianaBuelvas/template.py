# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#Load Dataset
dataSet=np.loadtxt("datasetRegLog.txt",delimiter=';')
x=dataSet[:,[0,1]]
y=dataSet[:,2]


#Plotting data
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y,s=25, edgecolor='k')
plt.show()