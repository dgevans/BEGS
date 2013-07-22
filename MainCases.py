# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:48:47 2013

@author: dgevans
"""

from primitives import BGP_parameters
import numpy as np
from Spline import Spline
from mpi4py import MPI
import initialize
import Bellman
import cPickle
import pdb

rank = MPI.COMM_WORLD.Get_rank()

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
def getParaBaselineIID():
    Para = getParaBaseline()
    Para.P = np.array([[ 0.30275229,  0.69724771],[ 0.30275229,  0.69724771]])
    return Para
def getParaBaseline3Per():
    Para = getParaBaseline()
    Para.theta_1 = np.array([3.99437196734200,4.05519996684467
,4.11602796634734])
    Para.theta_2 = np.array([0.985000000000000,1.,1.01500000000000])
    Para.P = np.insert(Para.P,1,.1*np.ones(2)/.9,axis=1)
    Para.P = np.insert(Para.P,1,np.ones(3),axis=0)
    pdb.set_trace()
    Para.P = Para.P/np.sum(Para.P,1).reshape((3,1))
    return Para
def getParaBaseline3IID():
    Para = getParaBaseline3Per()
    Para.P = np.array([[ 0.29338692,  0.13043478,  0.5761783 ],[ 0.29338692,  0.13043478,  0.5761783 ],[ 0.29338692,  0.13043478,  0.5761783 ]])
    return Para
    
Para = getParaBaseline3IID()
xgrid =np.linspace(Para.xmin,Para.xmax,25)
Rgrid = np.linspace(Para.Rmin,Para.Rmax,25)
X = Spline.makeGrid([xgrid,Rgrid])
domain = np.vstack((X,X,X))
domain = zip(domain[:,0],domain[:,1],[0]*len(X)+[1]*len(X)+[2]*len(X))
V0 = lambda state: initialize.completeMarketsSolution(state,Para)
Para.domain = domain
(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy),_ = Bellman.approximateValueFunctionAndPoliciesMPI(V0,Para)

Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy = Bellman.solveBellmanMPI(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy,Para)

if rank == 0:
    policyFile = open('PolicyRulesBaseine3IID.data','w')
    
    cPickle.dump((Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy,Para),policyFile)
    
    policyFile.close()