# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 07:40:19 2013

@author: dgevans
"""
from cpp_interpolator import interpolate
from cpp_interpolator import interpolate_INFO

from inner_opt import BGP
import primitives
import NAgents as na
from numpy import *
from utilities import makeGrid
import bellman

Para = primitives.BGP_parameters()
SSz = na.findSteadyState(Para,0.,3.)
CMz,x,R = na.SSz_to_CMz(SSz,Para)

X =makeGrid([linspace(0.5*x[0],1.2*x[0],10),linspace(0.5*x[1],1.5*x[1],10),linspace(0.8*R[0],1.2*R[0],5),linspace(0.8*R[1],1.2*R[1],5)])

INFO = interpolate_INFO(['spline','spline','hermite','hermite'],
                        [10,10,4,4],
                        [3,3,3,3])

VCM =hstack([na.VCM(hstack((x.flatten(),R.flatten())),CMz,Para)[0][0]]*len(X))

V0 = interpolate(X,VCM,INFO)

Slist = []
for s in range(0,len(Para.P)):
    Slist += [s]*len(X)
    
Para.domain = zip(X,Slist)
Para.xmin = amin(X[:,:2],0).reshape(-1,1)
Para.xmax = amin(X[:,2:],0).reshape(-1,1)
z0 = hstack((SSz[:2],SSz[2],SSz[4]))
Policies = [z0]*len(Para.domain)

T = bellman.BellmanMap(Para)
T.getInitialGuess = lambda state: bellman.getInitialGuess(state,Para,Policies)

Vnew = T([V0,V0])





