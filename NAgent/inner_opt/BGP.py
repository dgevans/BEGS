# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:57:21 2013

@author: dgevans
"""
from numpy import *
import pycppad as ad

def ComputeCi(c1,ci_,R,s_,Para):
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
    frac = (R*P[s_,:].dot(c1**(-sigma_1)) - P_.dot(ci_.T**(-sigma_1)).reshape((-1,1)))/P[s_,S-1]


    ci_S = frac**(-1/sigma_1)
    
    #vectorize c1, c2 to be of the form described above
    ci= hstack((ci_,ci_S))
    
    return ci
    
def ComputeR(c1,ci,Para):
    '''
    Computes Rprime and the gradient with respecto to z.  Note gradient will be
    a 2*S-1xS array
    return Rprime,gradRprime
    '''
    sigma_1 = Para.sigma_1
    Rprime = (ci**(-sigma_1) )/(c1**(-sigma_1));
    
    return Rprime
    
def Computel(c1,ci,Rprime,Para):
    '''
    COMPUTEL computes l_1 and l_2, the labor supply  of agent 1 and 2 in the
    standard 3x2 format, along with their gradients with respect to z.  Uses
    c1, c2, Rprime computed using computeC2 and computeRprime as well as their
    gradients.  Also passed are the primitives theta_1,theta_2, n_1, n_2 and
    the vector of government expenditures g.
    return l1,gradl1,l2,gradl2
    '''
    sigma_2 = Para.sigma_2
    g = Para.g
    n1 = Para.n[0]
    ni = Para.n[1:].reshape((-1,1))
    theta_1 = Para.theta[0,:]
    theta_i = Para.theta[1:,:]
    
    Rtild = (theta_i*Rprime/theta_1)**(-1.0/sigma_2)
    
    num = n1*c1+sum(ni*ci,0)+g+sum(ni*(Rtild-1)*theta_i,0)
    
    den = n1*theta_1 + sum(ni*theta_i*Rtild,0)
    
    l1 = num/den
    
    li = 1.-(1.-l1)*Rtild
    
    return l1,li
    

def ComputeXprime(c1,ci,Rprime,l1,li,x,s_,Para):
    '''
    COMPUTEXPRIME %Computes the choice of the state variable xprime tomorrow in the
    standard 3x2 format as well as gradient with respect to z
      return xprime,gradxprime
    '''
    P = Para.P
    beta = Para.beta
    uci = Para.Uc(ci)
    uc1 = Para.Uc(c1)
    ul1 = Para.Ul(l1)
    uli = Para.Ul(li)
    #Now the expected marginal utility of agent 2.  Again want it in 3x2
    #format
    Euci = P[s_,:].dot(uci.T).reshape((-1,1))

    
    #Now compute xprime from formula in notes
    xprime = (x*uci/(Euci) - (uci*ci+uli*li) + Rprime*(uc1*c1+ul1*l1))/beta

    return xprime
    
def computeGradients(z,x,R,s_,Para):
    '''
    Uses auto differentiation to compute the gradients
    '''
    S = len(Para.P)
    N = len(Para.theta)
    #defne the function for auto diff
    def F(z):
        c1 = z[:S]
        ci_ = z[S:].reshape((N-1,-1))
        ci = ComputeCi(c1,ci_,R,0,Para)
        Rprime = ComputeR(c1,ci,Para)
        l1,li = Computel(c1,ci,Rprime,Para)
        xprime = ComputeXprime(c1,ci,Rprime,l1,li,x,0,Para)
        return hstack((c1.flatten(),ci.flatten(),l1.flatten(),
                       li.flatten(),Rprime.flatten(),xprime.flatten()))
    #find the gradient
    a_z = ad.independent(z)
    a_F = F(a_z)
    f = ad.adfun(a_z,a_F)
    J = f.jacobian(z).T
    #unpack
    nz = 0
    gradc1 = J[:,0:S].reshape(-1,S)
    nz += S
    gradci = J[:,nz:nz+(N-1)*S].reshape(-1,N-1,S)
    nz +=(N-1)*S
    gradl1 = J[:,nz:nz+S].reshape(-1,S)
    nz += S
    gradli = J[:,nz:nz+(N-1)*S].reshape(-1,N-1,S)
    nz += (N-1)*S
    gradRprime = J[:,nz:nz+(N-1)*S].reshape(-1,N-1,S)
    nz += (N-1)*S
    gradxprime = J[:,nz:nz+(N-1)*S].reshape(-1,N-1,S)
    nz += (N-1)*S
    
    return gradc1,gradci,gradl1,gradli,gradRprime,gradxprime
    
    
    
def GradObjectiveUncon(z,x,R,s_,Vf,Para):
    '''
    Computes the gradient of the unconstrained objective function with respect to z
    '''

    P = Para.P
    N = len(Para.theta)
    alpha_1 = Para.alpha[0]
    alpha_i = Para.alpha[1:].reshape(N-1,1)
    sigma_1= Para.sigma_1
    Uc = Para.Uc
    Ul = Para.Ul
    beta = Para.beta
    
    
    S = P.shape[0]
    P_ = P[s_,0:S-1]
    #get c1 and c2_ from z
    c1 = z[0:S]
    ci_ = z[S:].reshape((N-1,-1))
    frac = (R*P[s_,:].dot(c1**(-sigma_1)) - P_.dot(ci_.T**(-sigma_1)).reshape((-1,1)))/P[s_,S-1]
    if min(z) > 0 and all(frac > 0):
        #first compute c2
        ci = ComputeCi(c1,ci_,R,0,Para)
        Rprime = ComputeR(c1,ci,Para)
        l1,li = Computel(c1,ci,Rprime,Para)
        xprime = ComputeXprime(c1,ci,Rprime,l1,li,x,0,Para)
        #get the gradients
        gradc1,gradci,gradl1,gradli,gradRprime,gradxprime = computeGradients(z,x,R,s_,Para)
        #Now compute gradient of value function tomorrow
        V_x = zeros((N-1,S))
        V_R = zeros((N-1,S))
        D = kron(eye(N-1,dtype=int),eye(N-1,dtype=int))
        for s in range(0,S):
            state = hstack((xprime[:,s],Rprime[:,s]))
            for i in range(0,N-1):
                V_x[i,s] = Vf[s](state,D[i,:])
                V_R[i,s] = Vf[s](state,D[i+N-1,:])
        #compute gradient of objective in each state
        gradobj = alpha_1*(Uc(c1)*gradc1+Ul(l1)*gradl1)\
        +sum(alpha_i*(Uc(ci)*gradci+Ul(li)*gradli),1) + beta*sum(V_x*gradxprime+V_R*gradRprime,1)
        if max(l1)>1 or max(li.flatten()) >1:
            return abs(z)+100
        
        return -gradobj.dot(Para.P[s_,:].T)
    else:
        return abs(z)+100
        
        
def GradObjectiveCon(z,x,R,s_,Vf,Para):
    '''
    Computes the gradient of the constrained objective function with respect to z
    '''

    P = Para.P
    N = len(Para.theta)
    alpha_1 = Para.alpha[0]
    alpha_i = Para.alpha[1:].reshape(N-1,1)
    xmax = Para.xmax.reshape(-1,1)
    xmin = Para.xmin.reshape(-1,1)
    sigma_1= Para.sigma_1
    Uc = Para.Uc
    Ul = Para.Ul
    beta = Para.beta
    
    
    S = P.shape[0]
    P_ = P[s_,0:S-1]
    #get c1 and c2_ from z
    c1 = z[0:S]
    ci_ = z[S:S+(N-1)*(S-1)].reshape((N-1,S-1))
    iz = S+(N-1)*(S-1)
    lambda_u = z[iz:iz+(N-1)*S].reshape((N-1,S))
    iz += (N-1)*S 
    lambda_l = z[iz:iz+(N-1)*S].reshape((N-1,S))
    
    frac = (R*P[s_,:].dot(c1**(-sigma_1)) - P_.dot(ci_.T**(-sigma_1)).reshape((-1,1)))/P[s_,S-1]
    if min(z[:S+(N-1)*(S-1)]) > 0 and all(frac > 0):
        #first compute c2
        ci = ComputeCi(c1,ci_,R,0,Para)
        Rprime = ComputeR(c1,ci,Para)
        l1,li = Computel(c1,ci,Rprime,Para)
        xprime = ComputeXprime(c1,ci,Rprime,l1,li,x,0,Para)
        #get the gradients
        gradc1,gradci,gradl1,gradli,gradRprime,gradxprime = computeGradients(z[:S+(N-1)*(S-1)],x,R,s_,Para)
        #Now compute gradient of value function tomorrow
        V_x = zeros((N-1,S))
        V_R = zeros((N-1,S))
        D = kron(eye(N-1,dtype=int),eye(N-1,dtype=int))
        for s in range(0,S):
            state = hstack((xprime[:,s],Rprime[:,s]))
            for i in range(0,N-1):
                V_x[i,s] = Vf[s](state,D[i,:])
                V_R[i,s] = Vf[s](state,D[i+N-1,:])
        #compute gradient of objective in each state
        gradobj = alpha_1*(Uc(c1)*gradc1+Ul(l1)*gradl1)\
        +sum(alpha_i*(Uc(ci)*gradci+Ul(li)*gradli),1) + beta*sum(V_x*gradxprime+V_R*gradRprime,1)
        -sum(lambda_u*gradxprime,1)+sum(lambda_l*gradxprime,1)
        if max(l1)>1 or max(li.flatten()) >1:
            return abs(z)+100
        
        gradobj =  gradobj.dot(Para.P[s_,:].T)
        
        con_u = amax(vstack(((abs(lambda_u*(xmax-xprime))).flatten(),-(xmax-xprime).flatten())),0)  
        con_l = amax(vstack(((abs(lambda_l*(xprime-xmin))).flatten(),-(xprime-xmin).flatten())),0) 
        return hstack((gradobj,con_u,con_l))
    else:
        return abs(z)+100
        
def Objective(z,x,R,s_,Vf,Para):
    '''
    Computes the gradient of the unconstrained objective function with respect to z
    '''

    P = Para.P
    N = len(Para.theta)
    alpha_1 = Para.alpha[0]
    alpha_i = Para.alpha[1:].reshape(N-1,1)
    U = Para.U

    beta = Para.beta
    
    
    S = P.shape[0]
    #get c1 and c2_ from z
    c1 = z[0:S]
    ci_ = z[S:].reshape((N-1,-1))
    #first compute c2
    ci = ComputeCi(c1,ci_,R,0,Para)
    Rprime = ComputeR(c1,ci,Para)
    l1,li = Computel(c1,ci,Rprime,Para)
    xprime = ComputeXprime(c1,ci,Rprime,l1,li,x,0,Para)
    Vprime = zeros(S)
    for s in range(0,S):
        state = hstack((xprime[:,s],Rprime[:,s]))
        Vprime[s] = Vf[s](state)

    #compute gradient of objective in each state
    obj = alpha_1*U(c1,l1)+sum(alpha_i*U(ci,li),0) +beta*Vprime
    return -obj.dot(Para.P[s_,:].T)
    
def Constraint(z,x,R,s_,Vf,Para):
    '''
    Computest the constraint
    '''
    P = Para.P
    N = len(Para.theta)
    
    S = P.shape[0]
    #get c1 and c2_ from z
    c1 = z[0:S]
    ci_ = z[S:].reshape((N-1,-1))
    #first compute c2
    ci = ComputeCi(c1,ci_,R,0,Para)
    Rprime = ComputeR(c1,ci,Para)
    l1,li = Computel(c1,ci,Rprime,Para)
    xprime = ComputeXprime(c1,ci,Rprime,l1,li,x,0,Para)
    
    return hstack(((Para.xmax-xprime).flatten(),(xprime-Para.xmin).flatten()))
    
def ConstraintJac(z,x,R,s_,Vf,Para):
    '''
    Computest the constraint
    '''
    _,_,_,_,_,gradxprime = computeGradients(z,x,R,s_,Para)
    nz = len(z)
    gradxprime = gradxprime.reshape(nz,-1)
    return hstack((-gradxprime,gradxprime)).T
