# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:48:47 2013

@author: dgevans
"""

from primitives import BGP_parameters
import numpy as np
from Spline import Spline
from mpi4py import MPI

def getParaBaseline():
    Para = BGP_parameters()
    #Para.P = np.array([[7.0/11.0, 4.0/11.0],[3.0/19.0,16.0/19.0]])
    Para.P = np.array([[0.636363636363636,0.363636363636364],[0.157894736842105,0.842105263157895]])
    Para.psi = 0.6994
    Para.theta_1 = np.array([3.99437196734200,4.11602796634734])
    Para.theta_2 = np.array([0.985000000000000,1.01500000000000])
    Para.g = 0.411884417673111
    Para.beta = np.array([0.98])
    Para.xmin = -3.0
    Para.xmax = 3.0
    Para.Rmin = 2.8
    Para.Rmax = 3.75
    Para.POLICY_TAG = 5
    Para.approxOrder = [2,2]
    return Para
    
    
    
Para = getParaBaseline()
xgrid =np.linspace(Para.xmin,Para.xmax,25)
Rgrid = np.linspace(Para.Rmin,Para.Rmax,25)
X = Spline.makeGrid([xgrid,Rgrid])
domain = np.vstack((X,X))
domain = zip(domain[:,0],domain[:,1],[0]*len(X)+[1]*len(X))
Para.domain = domain


comm = MPI.COMM_SELF.Spawn('python',
                           args=['mpiworker.py'],
                           maxprocs=2)
comm.bcast(Para,root=MPI.ROOT)