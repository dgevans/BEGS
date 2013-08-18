# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:20:32 2013

@author: dgevans
"""
import numpy as np
from inner_opt import BGP
import pdb

def simulateState(state0,T,xprime_policy,Rprime_policy,c1_policy,c2_policy,Para):
    '''
    Simulates the economy starting at state 0
    '''
    xHist = np.zeros(T)
    RHist = np.zeros(T)
    sHist = np.zeros(T,dtype=np.int)
    tauHist = np.zeros(T)
    
    xHist[0],RHist[0],sHist[0] = state0 #unpack state
    S = Para.P.shape[0]
    Pdist = Para.P.cumsum(1)
    for t in range(1,T):
        r = np.random.rand()
        s_ = sHist[t-1]
        for s in range(0,S):
            if r < Pdist[s_,s]:
                break
        sHist[t] = s
        xHist[t] = xprime_policy[(s_,s)]([xHist[t-1],RHist[t-1]])
        RHist[t] = Rprime_policy[(s_,s)]([xHist[t-1],RHist[t-1]])
        tauHist[t] = getTau(c1_policy,c2_policy,xHist[t-1],RHist[t-1],s_,Para)[s]
        
    return xHist,RHist,sHist,tauHist
    
    
def getTau(c1_policy,c2_policy,x,R,s_,Para):
    S = Para.P.shape[0]
    z0 = np.zeros(2*S-1)
    for s in range(0,S):
        z0[s] = c1_policy[(s_,s)]([x,R])
        if s < S-1:
            z0[S+s] = c2_policy[(s_,s)]([x,R])
    c1 = z0[0:S]
    c2_ = z0[S:2*S-1]        
    c1,c2,gradc1,gradc2 = BGP.ComputeC2(c1,c2_,R,s_,Para)
    Rprime,gradRprime = BGP.ComputeR(c1,c2,gradc1,gradc2,Para)
    l1,gradl1,l2,gradl2 = BGP.Computel(c1,gradc1,c2,gradc2,Rprime,gradRprime,Para)
    return 1+Para.Ul(l1)/(Para.theta_1*Para.Uc(c1))
        