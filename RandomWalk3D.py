# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:35:22 2021

@author: arath
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def Ranwalk3D(N):
    L = 5
    
    pos = np.zeros((N,3))
    
    for i in range(0,N-1):
        for j in range(0,3):
            #step = np.random.randint(-1,2) * L
            step = -L +2*L*np.random.uniform(0,1)
            pos[i+1,j] = pos[i,j] + step
            
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
            
    fig = plt.figure(1)
    sub1 = fig.add_subplot(111,projection = "3d")
    sub1.scatter(x,y,z)
    
    plt.show
    