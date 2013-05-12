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
from scipy.io import loadmat
from Spline import Spline
import initialize


#nx = 400
#S  = 2
#matVf = loadmat('Vf.mat')
#Vs = matVf['Vf'].reshape(400)
#domain = matVf['domain'][:,0:2]
#PR = matVf['PolicyRulesStore']
#Vf = []
#c1_policy = {}
#c2_policy = {}
#xprime_policy = {}
#Rprime_policy = {}
#for s_ in range(0,S):
#    Vf.append(Spline(domain[s_*nx:(s_+1)*nx,:],Vs,[3,3]))
#    for s in range(0,S):
#        c1_policy[(s_,s)] = Spline(domain[s_*nx:(s_+1)*nx,:],PR[s_*nx:(s_+1)*nx,s],[3,3])
#        c2_policy[(s_,s)] = Spline(domain[s_*nx:(s_+1)*nx,:],PR[s_*nx:(s_+1)*nx,S+s],[3,3])
#        Rprime_policy[(s_,s)] = Spline(domain[s_*nx:(s_+1)*nx,:],PR[s_*nx:(s_+1)*nx,5*S+s],[3,3])
#        xprime_policy[(s_,s)] = Spline(domain[s_*nx:(s_+1)*nx,:],PR[s_*nx:(s_+1)*nx,6*S+s],[3,3])
Para = BGP_parameters()
Para.theta_1 = np.array([3.2501,3.3499])
Para.theta_2 = np.array([0.9639 ,1.0361])
Para.g = .3
Para.beta = np.array([0.95, 0.85])
Para.xmin = -3.0
Para.xmax = 2.0
Para.Rmin = 2.8
Para.Rmax = 3.5
Para.approxOrder = [2,2]
xgrid = np.sort(np.hstack((np.linspace(Para.xmin,Para.xmax,20),[-2.234664454712415])))
Rgrid = np.sort(np.hstack((np.linspace(Para.Rmin,Para.Rmax,15),[3.261051140535215])))
X = Spline.makeGrid([xgrid,Rgrid])
domain = np.vstack((X,X))
domain = zip(domain[:,0],domain[:,1],[0]*len(X)+[1]*len(X))

V0 = lambda state: initialize.completeMarketsSolution(state,Para)
Para.domain = domain
(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy),_ = Bellman.approximateValueFunctionAndPolicies(V0,Para)

Vf,Vf_old,Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy = Bellman.solveBellman(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy,Para)
        


    