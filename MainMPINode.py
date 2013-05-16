# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:33:24 2013

@author: dgevans
"""
import sys
sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
from primitives import BGP_parameters
import numpy as np
import Bellman
from Spline import Spline
import initialize
from mpi4py import MPI


Para = BGP_parameters()
Para.theta_1 = np.array([3.2501,3.3499])
Para.theta_2 = np.array([0.9639 ,1.0361])
Para.g = .3
Para.beta = np.array([0.95, 0.85])
Para.xmin = -3.0
Para.xmax = 2.0
Para.Rmin = 2.4
Para.Rmax = 3.7
Para.POLICY_TAG = 5
Para.approxOrder = [2,2]
xgrid = np.sort(np.hstack((np.linspace(Para.xmin,Para.xmax,10),[-0.9943])))
Rgrid = np.sort(np.hstack((np.linspace(Para.Rmin,Para.Rmax,10),[3.0996])))
X = Spline.makeGrid([xgrid,Rgrid])
domain = np.vstack((X,X))
domain = zip(domain[:,0],domain[:,1],[0]*len(X)+[1]*len(X))
Para.domain = domain


comm = MPI.COMM_SELF.Spawn('python',
                           args=['mpiworker.py'],
                           maxprocs=2)
comm.bcast(Para,root=MPI.ROOT)
#Vf,Vf_old,Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy = Bellman.solveBellmanMPI(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy,Para)
        