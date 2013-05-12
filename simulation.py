# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:20:32 2013

@author: dgevans
"""
import numpy as np

def simulateState(state0,T,xprime_policy,Rprime_policy,Para):
    '''
    Simulates the economy starting at state 0
    '''
    xHist = np.zeros(T)
    RHist = np.zeros(T)
    sHist = np.zeros(T,dtype=np.int)
    
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
        
    return xHist,RHist,sHist