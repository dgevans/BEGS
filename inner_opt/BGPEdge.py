# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:23:13 2013

@author: dgevans
"""

import numpy as np
def ComputeC2(c1,R,s_,Para):
    '''
    COMPUTEC2_@ Computes c2(S-1).  The (2*S-1)xS matrix format is useful for computing future
    gradients.  gradc1 will a matrix containing the derivative of
    c1 with respect to the various z's.  For instance the ith row and jth
    column will be the derivative of c_1(j) with respect to x(i).  Thus
    gradc1 will be
           1   0
           0   1      
           0   0
    Similarly for gradc2
    return c1,c2,gradc1,gradc2
    '''
    S = Para.P.shape[0]
    sigma_1= Para.sigma_1
    
    c2 = (R**(-1.0*sigma_1))*c1
    gradc1 = np.eye(S)
    gradc2 = (R**(-1.0*sigma_1))*gradc1
    
    return c1,c2,gradc1,gradc2
    
    
def Computel(c1,gradc1,c2,gradc2,R,Para):
    '''
    COMPUTEL computes l_1 and l_2, the labor supply  of agent 1 and 2 in the
    standard 3x2 format, along with their gradients with respect to z.  Uses
    c1, c2, Rprime computed using computeC2 and computeRprime as well as their
    gradients.  Also passed are the primitives theta_1,theta_2, n_1, n_2 and
    the vector of government expenditures g.
    return l1,gradl1,l2,gradl2
    '''
    sigma_2 = Para.sigma_2
    theta_1 = np.atleast_1d(Para.theta_1)    
    theta_2 = np.atleast_1d(Para.theta_2)
    g = Para.g
    n1 = Para.n1
    n2 = Para.n2
    S = Para.P.shape[0]    
    
    Rtilde = R**(1.0/sigma_2)
    
    if np.all(theta_1 != 0.0) and np.all(theta_2 != 0.0):
        #Compute l2 first
        num = n1*c1+n2*c2+g+n1*theta_2*Rtilde-n1*theta_1
        gradnum = n1*gradc1+n2*gradc2
        den = theta_2*(n2+Rtilde*n1)
        
        #l2 = (n1*c1+n2*c2+g+n1*theta_2*Rtilde-n1*theta_1  )/(theta_2*(n2+Rtilde*n1))
        l2 = num/den
        #now gradl2
        #gradl2 = n1*gradRtilde/(n2+n1*Rtilde) - n1*gradRtilde*l2/(n2+n1*Rtilde)
        #+n1*gradc1/(theta_2*(n2+n1*Rtilde)) +n2*gradc2/(theta_2*(n2+n1*Rtilde))
        gradl2 = gradnum/den
        
        #now l1
        l1 = 1.0 - (1.0-l2)*Rtilde*theta_2/theta_1
        gradl1 = gradl2*Rtilde*theta_2/theta_1
    elif theta_1 == 0.0:
        l2 = (n1*c1+n2*c2+g)/(n2*theta_2)
        gradl2 = n1*gradc1/(n2*theta_2) + gradc2/theta_2
        l1 = np.zeros(S)
        gradl1 = np.zeros(gradc1.shape)
    elif theta_2 == 0.0:
        l2 = np.zeros(S)
        gradl2 = np.zeros(gradc1.shape)
        l1 = (n1*c1+n2*c2+g)/(n1*theta_1)
        gradl1 = gradc1/theta_1+n2*gradc2/(n1*theta_1)
            
    
    return l1,gradl1,l2,gradl2

def ComputeXprime(c1,gradc1,c2,gradc2,l1,gradl1,l2,gradl2,x,R,s_,Para):
    '''
    COMPUTEXPRIME %Computes the choice of the state variable xprime tomorrow in the
    standard 3x2 format as well as gradient with respect to z
      return xprime,gradxprime
    '''
    P = Para.P
    beta = Para.beta
    uc2 = Para.Uc(c2)
    ucc2 = Para.Ucc(c2)
    uc1 = Para.Uc(c1)
    ucc1 = Para.Ucc(c1)
    ul1 = Para.Ul(l1)
    ull1 = Para.Ull(l1)
    ul2 = Para.Ul(l2)
    ull2 = Para.Ull(l2)
    #Now the expected marginal utility of agent 2.  Again want it in 3x2
    #format
    Euc2 = P[s_,:].dot(uc2)
    #gives a 2S-1 x S matrix with each column representing the partial derivative of
    #Euc2 with eash S
    gradEuc2 = (ucc2*gradc2).dot(P[s_,:]).reshape((-1,1))

    
    #Now compute xprime from formula in notes
    xprime = (x*uc2/(Euc2) - (uc2*c2+ul2*l2) + R*(uc1*c1+ul1*l1))/beta

    #Now compute the gradient
    gradxprime = ((x*ucc2/(Euc2)-uc2-ucc2*c2)*gradc2-x*uc2/(Euc2**2)*gradEuc2+R*(uc1+ucc1*c1)*gradc1-(ul2+ull2*l2)*gradl2+R*(ul1+ull1*l1)*gradl1)/beta
    
    return xprime,gradxprime
    
    
def GradObjectiveUncon(z,x,R,s_,Vf,Para):
    '''
    Computes the gradient of the unconstrained objective function with respect to z
    '''

    P = Para.P
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    Uc = Para.Uc
    Ul = Para.Ul
    beta = Para.beta
    
    
    S = P.shape[0]
    #get c1 and c2_ from z
    c1 = z[0:S]
    if min(z) > 0:
        #first compute c2
        c1,c2,gradc1,gradc2 = ComputeC2(c1,R,s_,Para)
        l1,gradl1,l2,gradl2 = Computel(c1,gradc1,c2,gradc2,R,Para)
        xprime,gradxprime = ComputeXprime(c1,gradc1,c2,gradc2,l1,gradl1,l2,gradl2,x,R,s_,Para)
        #Now compute gradient of value function tomorrow
        V_x = np.zeros(S)
        for s in range(0,S):
            V_x[s] = Vf[s]([xprime[s],R],[1,0])
        #compute gradient of objective in each state
        gradobj = alpha_1*(Uc(c1)*gradc1+Ul(l1)*gradl1)\
        +alpha_2*(Uc(c2)*gradc2+Ul(l2)*gradl2) + beta*V_x*gradxprime
        if max(l1)>1 or max(l2) >1:
            return np.abs(z)+100
        
        
        return gradobj.dot(Para.P[s_,:].T)
    else:
        return np.abs(z)+100

def ConstrainedObjective(z,x,R,s_,Vf,Para):
    '''
    Computes the constrained objective function.  Here we pass c1,c2,and xprime.
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    U = Para.U
    #now compute l
    c1,c2,_,_ = ComputeC2(c1,R,s_,Para)
    l1,_,l2,_ = Computel(c1,0,c2,0,R,Para)
    xprime,_ = ComputeXprime(c1,0,c2,0,l1,0,l2,0,x,R,s_,Para)
    #Compute Vprime
    Vprime = np.zeros(S)
    for s in range(0,S):
        Vprime[s] = Vf[s]([xprime[s],R])
    
    #return value
    obj = alpha_1*U(c1,l1)+alpha_2*U(c2,l2)+Para.beta*Vprime
    return -Para.P[s_,:].dot(obj)
    
def ConstrainedObjectiveJac(z,x,R,s_,Vf,Para):
    '''
    Computes the Jacobian of the constrained objective function
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    Uc = Para.Uc
    Ul = Para.Ul
    beta = Para.beta
    
    
    c1,c2,gradc1,gradc2 = ComputeC2(c1,R,s_,Para)
    l1,gradl1,l2,gradl2 = Computel(c1,gradc1,c2,gradc2,R,Para)
    xprime,gradxprime = ComputeXprime(c1,gradc1,c2,gradc2,l1,gradl1,l2,gradl2,x,R,s_,Para)
    #compute derivatives of V
    V_x = np.zeros(S)
    for s in range(0,S):
        V_x[s] = Vf[s]([xprime[s],R],[1,0])
    #compute gradient of objective in each state
    gradobj = alpha_1*(Uc(c1)*gradc1+Ul(l1)*gradl1)\
    +alpha_2*(Uc(c2)*gradc2+Ul(l2)*gradl2) + beta*V_x*gradxprime
    
    return -Para.P[s_,:].dot(gradobj)
    
    
def ieq_cons(z,x,R,s_,Vf,Para):
    '''
    Computes the inequality constraints associated with the constrained optization.  Namely
    those associated with R and xprime
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    
    c1,c2,_,_ = ComputeC2(c1,R,s_,Para)    
    l1,_,l2,_ = Computel(c1,0,c2,0,R,Para)
    xprime,_ = ComputeXprime(c1,0,c2,0,l1,0,l2,0,x,R,s_,Para)
    
    return np.hstack((xprime-Para.xmin,Para.xmax-xprime))
    
    
    
def ieq_consJacobian(z,x,R,s_,Vf,Para):
    '''
    Computes the Jacobian of the inequality constraints associated with the constrained optimization.
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    
    c1,c2,gradc1,gradc2 = ComputeC2(c1,R,s_,Para)
    l1,gradl1,l2,gradl2 = Computel(c1,gradc1,c2,gradc2,R,Para)
    xprime,gradxprime = ComputeXprime(c1,gradc1,c2,gradc2,l1,gradl1,l2,gradl2,x,R,s_,Para)
    
    return np.vstack((gradxprime.T,-gradxprime.T))
    
        
    