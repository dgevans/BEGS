# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:07:40 2013

@author: dgevans
"""
from scipy.optimize import root
import numpy as np

def C(R,Para):
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    return (1.0+R)/(alpha_1*(1.0-R)+2*R**2*alpha_2)

def l(R,Para):
    g = Para.g
    theta_1 = Para.theta_1
    return (g+np.sqrt(g**2 + 4*C(R,Para)*theta_1**2))/(2*theta_1)
    
def uc2(R,Para):
    return l(R,Para)*(1+R)/(Para.theta_1*C(R,Para))
    
def x(R,Para):
    uc = uc2(R,Para)
    Euc = Para.P.dot(uc)
    return (1.0 + R*(l(R,Para)**2-1))/(uc/Euc- Para.beta)

def findSS(Para,R0 = 2.0):
    xdiff = lambda R: x(R,Para)[0]-x(R,Para)[1]
    RSS = root(xdiff,R0).x
    xSS = x(RSS,Para)
    return xSS,RSS
    
def findSS_alt(Para,xSS=1.0,RSS = 3.0):
    S = Para.P.shape[0]
    c2 = 1/uc2(RSS,Para)
    c1 = RSS*c2
    l1 = l(RSS,Para)
    z0 = np.zeros(5*S+4)
    z0[0:S] = c1
    z0[S:2*S] = c2
    z0[2*S:3*S] = l1
    z0[5*S] = xSS
    z0[5*S+1] = RSS
    
    zSS = root(lambda z: SSResiduals(z,Para),z0).x
    return zSS[5*S],zSS[5*S+1],zSS
    
def SSResiduals(z,Para):
    S = Para.P.shape[0]
    beta = Para.beta
    theta_1 = Para.theta_1    
    g = Para.g
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    
    c1 = z[0:S]
    c2 = z[S:2*S]
    l1 = z[2*S:3*S]
    xi = z[3*S:4*S]
    rho = z[4*S:5*S]
    x = z[5*S]
    R = z[5*S+1]
    mu = z[5*S+2]
    lamb = z[5*S+3]
    
    uc = 1.0/c2
    Euc = Para.P.dot(uc) 
    
    res = np.zeros(7*S)
    res[0:S] = 1.0+R*(l1**2-1)+beta*x -x*uc/Euc
    res[S:2*S] = theta_1*l1 - c1-c2-g
    res[2*S:3*S] = R*c2-c1
    res[3*S:4*S] = alpha_1/c1 - xi - rho
    res[4*S:5*S] = alpha_2/c2 - xi + R*rho
    res[5*S:6*S] = (2*mu*R-alpha_1)*l1 + theta_1*xi
    res[6*S:7*S] = (lamb/R)*( uc-beta*Euc ) + mu*( l1**2-1) + rho*c2
    
    return res