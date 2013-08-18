# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:31:57 2013

@author: dgevans
"""

import numpy as np
from scipy.optimize import root

def completeMarketsResiduals(z,state,Para):
    '''
    Compute the residuals for the complete markets solution for a given state
    '''
    x,R,s_ = state #unpack state
    #get some things from Para
    S = Para.P.shape[0]
    P = Para.P
    g = Para.g
    Uc = Para.Uc
    Ul = Para.Ul
    Ucc = Para.Ucc
    Ull = Para.Ull
    n1 = Para.n1
    n2 = Para.n2
    alpha_1 =Para.alpha_1
    alpha_2 = Para.alpha_2
    theta_1 = Para.theta_1
    theta_2 = Para.theta_2
    beta = Para.beta
    
    c1 = z[0:S]
    c2 = z[S:2*S]
    l1 = z[2*S:3*S]
    l2 = z[3*S:4*S]
    phi = z[4*S:5*S]
    xi = z[5*S:6*S]
    rho = z[6*S:7*S]
    mu = z[7*S]
    uc1,ucc1,ul1,ull1 = Uc(c1),Ucc(c1),Ul(l1),Ull(l2)
    uc2,ucc2,ul2,ull2 = Uc(c2),Ucc(c2),Ul(l2),Ull(l2)
    
    res = np.zeros(7*S+1)
    
    res[0:S] = theta_2*R*ul1/theta_1-ul2
    res[S:2*S] = n1*l1*theta_1+n2*l2*theta_2 - n1*c1-n2*c2-g
    res[2*S:3*S] = R*uc1-uc2
    res[3*S:4*S] = alpha_1*uc1+R*mu*( uc1+ucc1*c1 ) -n1*xi + ucc1*R*rho
    res[4*S:5*S] = alpha_2*uc2-mu*( uc2+ucc2*c2 ) - n2*xi - ucc2*rho
    res[5*S:6*S] = alpha_1*ul1 + R*mu*( ul1 + ull1*l1 ) +theta_2*R*ull1*phi/theta_1 + n1*theta_1*xi
    res[6*S:7*S] = alpha_2*ul2 - mu*( ul2 + ull2*l2 ) -ull2*phi + n2*theta_2*xi
    I  = uc2*c2+ul2*l2-R*( uc1*c1 + ul1*l1)
    res[7*S] = P[s_,:].dot(np.linalg.solve(np.eye(S)-(beta*P.T).T,I))-x
    return res
    
def completeMarketsSolution(state,Para):
    '''
    Solve for the complete markets solution
    '''
    x,R,s_ = state #unpack state
    S = Para.P.shape[0]
    U = Para.U
    z0 = np.hstack((0.5*np.ones(4*S),np.zeros(3*S+1)))
    zLS = root(lambda z:completeMarketsResiduals(z,state,Para),z0).x
    c1 = zLS[0:S]
    c2 = zLS[S:2*S]
    l1 = zLS[2*S:3*S]
    l2 = zLS[3*S:4*S]
    xprime = x * np.ones(S)
    u = Para.alpha_1*U(c1,l1)+Para.alpha_2*U(c2,l2)
    V = Para.P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,u))
    return c1,c2,xprime,V
    
    
def firstBestSolution(state,Para):
    '''
    Solves for first best given the R constraint
    '''
    x,R,s_ = state
    S = Para.P.shape[0]
    U = Para.U
    Uc = Para.Uc
    Ul = Para.Ul
    def FBResiduals(z):
        c1 = z[0:S]
        c2 = z[S:2*S]
        l1 = z[2*S:3*S]
        l2 = z[3*S:4*S]
        xi = z[4*S:5*S]
        rho = z[5*S:6*S]
        g = Para.g
        Uc = Para.Uc
        Ul = Para.Ul
        Ucc = Para.Uc
        n1 = Para.n1
        n2 = Para.n2
        alpha_1 =Para.alpha_1
        alpha_2 = Para.alpha_2
        theta_1 = Para.theta_1
        theta_2 = Para.theta_2
        uc1,ucc1,ul1 = Uc(c1),Ucc(c1),Ul(l1)
        uc2,ucc2,ul2 = Uc(c2),Ucc(c2),Ul(l2)
        res = np.zeros(6*S)
        res[:S] = n1*l1*theta_1+n2*l2*theta_2 - n1*c1-n2*c2-g
        res[S:2*S] = R*uc1-uc2
        res[2*S:3*S] = alpha_1*uc1 -n1*xi + ucc1*R*rho
        res[3*S:4*S] = alpha_2*uc2 - n2*xi - ucc2*rho
        res[4*S:5*S] = alpha_1*ul1  + n1*theta_1*xi
        res[5*S:6*S] = alpha_2*ul2 + n2*theta_2*xi
        return res
    z0 = np.hstack((0.5*np.ones(4*S),np.zeros(2*S)))
    zFB = root(FBResiduals,z0).x
    c1 = zFB[0:S]
    c2 = zFB[S:2*S]
    l1 = zFB[2*S:3*S]
    l2 = zFB[3*S:4*S]
    uc1,ul1 = Uc(c1),Ul(l1)
    uc2,ul2 = Uc(c2),Ul(l2)
    Euc2 = Para.P.dot(uc2)[s_]
    xprime = (x*uc2/Euc2+R*(uc1*c1+ul1*l1)-uc2*c2-ul2*l2)/Para.beta
    u = Para.alpha_1*U(c1,l1)+Para.alpha_2*U(c2,l2)
    V = Para.P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,u))
    return c1,c2,xprime,V
    
    
    
    
    