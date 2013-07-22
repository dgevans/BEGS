# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:23:13 2013

@author: dgevans
"""

import numpy as np  
from scipy import weave

def ComputeC2(c1,c2_,R,s_,Para):
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
    P = Para.P
    sigma_1= Para.sigma_1
    
    
    S = P.shape[0]
    P_ = P[s_,0:S-1]
    #Compute c2_S from formula
    frac = (R*P[s_,:].dot(c1**(-sigma_1)) - P_.dot(c2_**(-sigma_1)))/P[s_,S-1]


    c2_S = frac**(-1/sigma_1)
    
    
    # take gradients
    gradc2_S=np.zeros(2*S-1)
    gradc2_S[0:S] = c1**(-sigma_1-1.0)*frac**(-1.0/sigma_1-1.0)*R*P[s_,:]/P[s_,S-1]
    gradc2_S[S:2*S-1] = -c2_**(-sigma_1-1.0)*frac**(-1.0/sigma_1-1.0)*P_/P[s_,S-1]
    
    gradc1 = np.vstack((np.eye(S),np.zeros((S-1,S))))
    
    gradc2 = np.zeros((2*S-1,S))
    gradc2[S:2*S-1,0:S-1] = np.eye(S-1)
    gradc2[:,S-1] = gradc2_S
    #vectorize c1, c2 to be of the form described above
    c1=c1
    c2=np.hstack((c2_,c2_S))
    
    return c1,c2,gradc1,gradc2
    
def ComputeR(c1,c2,gradc1,gradc2,Para):
    '''
    Computes Rprime and the gradient with respecto to z.  Note gradient will be
    a 2*S-1xS array
    return Rprime,gradRprime
    '''
    sigma_1 = Para.sigma_1
    Rprime = (c2**(-sigma_1) )/(c1**(-sigma_1));
    
    gradRprime = sigma_1*c2**(-sigma_1)*c1**(sigma_1-1)*gradc1\
    -sigma_1*c2**(-sigma_1-1.0)*c1**(sigma_1)*gradc2
    
    return Rprime,gradRprime
    
def Computel(c1,gradc1,c2,gradc2,Rprime,gradRprime,Para):
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
    
    Rtilde = Rprime**(1.0/sigma_2)    
    gradRtilde = gradRprime*Rprime**(1.0/sigma_2-1.0)/sigma_2    
    
    if np.all(theta_1 != 0.0) and np.all(theta_2 != 0.0):
        
        Rtilde = Rprime**(1.0/sigma_2)    
        gradRtilde = gradRprime*Rprime**(1.0/sigma_2-1.0)/sigma_2
        #Compute l2 first
        num = n1*c1+n2*c2+g+n1*theta_2*Rtilde-n1*theta_1
        gradnum = n1*gradc1+n2*gradc2+n1*theta_2*gradRtilde
        den = theta_2*(n2+Rtilde*n1)
        gradden = n1*gradRtilde*theta_2
        
        #l2 = (n1*c1+n2*c2+g+n1*theta_2*Rtilde-n1*theta_1  )/(theta_2*(n2+Rtilde*n1))
        l2 = num/den
        #now gradl2
        #gradl2 = n1*gradRtilde/(n2+n1*Rtilde) - n1*gradRtilde*l2/(n2+n1*Rtilde)
        #+n1*gradc1/(theta_2*(n2+n1*Rtilde)) +n2*gradc2/(theta_2*(n2+n1*Rtilde))
        gradl2 = gradnum/den-num*gradden/(den**2)
        
        #now l1
        l1 = 1.0 - (1.0-l2)*Rtilde*theta_2/theta_1
        gradl1 = gradl2*Rtilde*theta_2/theta_1 - (1.0-l2)*gradRtilde*theta_2/theta_1
    elif theta_1 == 0.0:
        l2 = (n1*c1+n2*c2+g)/(n2*theta_2)
        gradl2 = n1*gradc1/(n2*theta_2) + gradc2/theta_2
        l1 = np.zeros(S)
        gradl1 = np.zeros(gradRprime.shape)
    elif theta_2 == 0.0:
        l2 = np.zeros(S)
        gradl2 = np.zeros(gradRprime.shape)
        l1 = (n1*c1+n2*c2+g)/(n1*theta_1)
        gradl1 = gradc1/theta_1+n2*gradc2/(n1*theta_1)
            
    
    return l1,gradl1,l2,gradl2

def ComputeXprime(c1,gradc1,c2,gradc2,Rprime,gradRprime,l1,gradl1,l2,gradl2,x,s_,Para):
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
    xprime = (x*uc2/(Euc2) - (uc2*c2+ul2*l2) + Rprime*(uc1*c1+ul1*l1))/beta

    #Now compute the gradient
    gradxprime = ((x*ucc2/(Euc2)-uc2-ucc2*c2)*gradc2-x*uc2/(Euc2**2)*gradEuc2+Rprime*(uc1+ucc1*c1)*gradc1-(ul2+ull2*l2)*gradl2+Rprime*(ul1+ull1*l1)*gradl1+(uc1*c1+ul1*l1)*gradRprime)/beta
    
    return xprime,gradxprime

def GradObjectiveUncon(z,x,R,s_,Vf,Para):
    '''
    Computes the gradient of the unconstrained objective function with respect to z
    '''

    P = Para.P
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    sigma_1= Para.sigma_1
    Uc = Para.Uc
    Ul = Para.Ul
    beta = Para.beta
    
    
    S = P.shape[0]
    P_ = P[s_,0:S-1]
    #get c1 and c2_ from z
    c1 = z[0:S]
    c2_ = z[S:2*S-1]
    frac = (R*P[s_,:].dot(c1**(-sigma_1)) - P_.dot(c2_**(-sigma_1)))/P[s_,S-1]
    if min(z) > 0 and frac > 0:
        #first compute c2
        c1,c2,gradc1,gradc2 = ComputeC2(c1,c2_,R,s_,Para)
        Rprime,gradRprime = ComputeR(c1,c2,gradc1,gradc2,Para)
        l1,gradl1,l2,gradl2 = Computel(c1,gradc1,c2,gradc2,Rprime,gradRprime,Para)
        xprime,gradxprime = ComputeXprime(c1,gradc1,c2,gradc2,Rprime,gradRprime,l1,gradl1,l2,gradl2,x,s_,Para)
        #Now compute gradient of value function tomorrow
        V_x = np.zeros(S)
        V_R = np.zeros(S)
        for s in range(0,S):
            V_x[s] = Vf[s]([xprime[s],Rprime[s]],[1,0])
            V_R[s] = Vf[s]([xprime[s],Rprime[s]],[0,1])
        #compute gradient of objective in each state
        gradobj = alpha_1*(Uc(c1)*gradc1+Ul(l1)*gradl1)\
        +alpha_2*(Uc(c2)*gradc2+Ul(l2)*gradl2) + beta*(V_x*gradxprime+V_R*gradRprime)
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
    c2 = z[S:2*S]
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    U = Para.U
    
    #first compute Rprime
    Rprime = Para.Uc(c2)/Para.Uc(c1)
    #now compute l
    l1,_,l2,_ = Computel(c1,0,c2,0,Rprime,0,Para)
    xprime,_ = ComputeXprime(c1,0,c2,0,Rprime,0,l1,0,l2,0,x,s_,Para)
    #Compute Vprime
    Vprime = np.zeros(S)
    for s in range(0,S):
        Vprime[s] = Vf[s]([xprime[s],Rprime[s]])
    
    #return value
    obj = alpha_1*U(c1,l1)+alpha_2*U(c2,l2)+Para.beta*Vprime
    return -Para.P[s_,:].dot(obj)
    
def ConstrainedObjectiveJac(z,x,R,s_,Vf,Para):
    '''
    Computes the Jacobian of the constrained objective function
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    gradc1 = np.zeros((3*S,S))
    gradc1[0:S,:] = np.eye(S)
    c2 = z[S:2*S]
    gradc2 = np.zeros((3*S,S))
    gradc2[S:2*S,:] = np.eye(S)
    alpha_1 = Para.alpha_1
    alpha_2 = Para.alpha_2
    Uc = Para.Uc
    Ul = Para.Ul
    beta = Para.beta
    
    
    Rprime,gradRprime = ComputeR(c1,c2,gradc1,gradc2,Para)
    l1,gradl1,l2,gradl2 = Computel(c1,gradc1,c2,gradc2,Rprime,gradRprime,Para)
    xprime,gradxprime = ComputeXprime(c1,gradc1,c2,gradc2,Rprime,gradRprime,l1,gradl1,l2,gradl2,x,s_,Para)
    #compute derivatives of V
    V_x = np.zeros(S)
    V_R = np.zeros(S)
    for s in range(0,S):
        V_x[s] = Vf[s]([xprime[s],Rprime[s]],[1,0])
        V_R[s] = Vf[s]([xprime[s],Rprime[s]],[0,1])
    #compute gradient of objective in each state
    gradobj = alpha_1*(Uc(c1)*gradc1+Ul(l1)*gradl1)\
    +alpha_2*(Uc(c2)*gradc2+Ul(l2)*gradl2) + beta*(V_x*gradxprime+V_R*gradRprime)
    
    return -Para.P[s_,:].dot(gradobj)
    
def eq_con(z,x,R,s_,Vf,Para):
    '''
    Computes the constraints associated with the constrained optization.  Namely
    those associated with R and xprime
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    c2 = z[S:2*S]
    P = Para.P    
    Uc = Para.Uc
    
    Rprime,_ = ComputeR(c1,c2,0,0,Para)
    l1,_,l2,_ = Computel(c1,0,c2,0,Rprime,0,Para)
    
    return R*P[s_,:].dot(Uc(c1))-P[s_,:].dot(Uc(c2))
    
def ieq_cons(z,x,R,s_,Vf,Para):
    '''
    Computes the inequality constraints associated with the constrained optization.  Namely
    those associated with R and xprime
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    c2 = z[S:2*S]
    
    Rprime,_ = ComputeR(c1,c2,0,0,Para)
    l1,_,l2,_ = Computel(c1,0,c2,0,Rprime,0,Para)
    xprime,_ = ComputeXprime(c1,0,c2,0,Rprime,0,l1,0,l2,0,x,s_,Para)
    
    return np.hstack((xprime-Para.xmin,Para.xmax-xprime,Rprime-Para.Rmin,Para.Rmax-Rprime))
    
    
def eq_conJacobian(z,x,R,s_,Vf,Para):
    '''
    Computes the Jacobian of the constraints associated with the constrained optimization.
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    gradc1 = np.zeros((2*S,S))
    gradc1[0:S,:] = np.eye(S)
    c2 = z[S:2*S]
    gradc2 = np.zeros((2*S,S))
    gradc2[S:2*S,:] = np.eye(S)
    Ucc = Para.Ucc
    P = Para.P
    
    
    Rprime,gradRprime = ComputeR(c1,c2,gradc1,gradc2,Para)
    
    return R*(Ucc(c1)*gradc1).dot(P[s_,:])-(Ucc(c2)*gradc2).dot(P[s_,:])
    
def ieq_consJacobian(z,x,R,s_,Vf,Para):
    '''
    Computes the Jacobian of the inequality constraints associated with the constrained optimization.
    '''
    S = Para.P.shape[0]
    c1 = z[0:S]
    gradc1 = np.zeros((2*S,S))
    gradc1[0:S,:] = np.eye(S)
    c2 = z[S:2*S]
    gradc2 = np.zeros((2*S,S))
    gradc2[S:2*S,:] = np.eye(S)
    
    Rprime,gradRprime = ComputeR(c1,c2,gradc1,gradc2,Para)
    l1,gradl1,l2,gradl2 = Computel(c1,gradc1,c2,gradc2,Rprime,gradRprime,Para)
    xprime,gradxprime = ComputeXprime(c1,gradc1,c2,gradc2,Rprime,gradRprime,l1,gradl1,l2,gradl2,x,s_,Para)
    
    return np.vstack((gradxprime.T,-gradxprime.T,gradRprime.T,-gradRprime.T))
    
        
    