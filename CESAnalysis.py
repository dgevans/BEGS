# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:33:25 2013

@author: dgevans
"""
from numpy import *
from scipy.optimize import root

class parameters(object):
    
    alpha_1 = 0.69

    alpha_2 = 0.31    
    
    sigma = 2.
    
    gamma = 3.
    
    theta_1 = array([3.3,3.])
    
    theta_2 = array([1.1,1.])
    
    g = .35
    
    P = array([0.5,0.5])
    
    beta = 0.9
    
    
Para= parameters()



def get_c1(tau,R,Para):
    '''
    computes c1
    '''
    gamma = Para.gamma
    sigma = Para.sigma
    g = Para.g
    theta_1 = Para.theta_1
    theta_2 = Para.theta_2
    def c1res(c1):
        return (theta_1*( (1-tau)*theta_1 )**(1./gamma) + theta_2*( (1-tau)*R*theta_2)**(1./gamma))*c1**(-sigma/gamma) -g - (1+R**(-1./sigma))*c1
    
    sol = root(c1res,5*ones(2))
    if not sol.success:
        raise Exception(sol.message)
    else:
        return sol.x
        
def dc1_dtau(tau,R,Para,c1=None):
    '''
    Computes derivative of c1 with respect to tau
    '''
    if c1==None:
        c1 = get_c1(tau,R,Para)
    gamma = Para.gamma
    sigma = Para.sigma
    theta_1 = Para.theta_1
    theta_2 = Para.theta_2
    num = (theta_1**2*((1-tau)*theta_1)**(1./gamma-1)+theta_2**2*((1-tau)*theta_2)**(1./gamma-1)*R**(1./gamma))*c1**(-sigma/gamma)/gamma
    den = 1+R**(-1./sigma)+sigma*(theta_1*((1-tau)*theta_1)**(1./gamma)+theta_2*((1-tau)*theta_2)**(1./gamma)*R**(1./gamma))*c1**(-sigma/gamma-1)/gamma
    return -num/den
    
    
def dc1_dR(tau,R,Para,c1=None):
    '''
    Computes derivative of c1 with respect to R
    '''
    if c1==None:
        c1 = get_c1(tau,R,Para)
    gamma = Para.gamma
    sigma = Para.sigma
    theta_1 = Para.theta_1
    theta_2 = Para.theta_2
    num = (theta_2*((1-tau)*theta_2)**(1./gamma)*R**(1./gamma-1))*c1**(-sigma/gamma)/gamma + R**(-1./sigma-1)*c1/sigma
    den = 1+R**(-1./sigma)+sigma*(theta_1*((1-tau)*theta_1)**(1./gamma)+theta_2*((1-tau)*theta_2)**(1./gamma)*R**(1./gamma))*c1**(-sigma/gamma-1)/gamma
    return num/den
    
def Dtau(tau,R,Para):
    '''
    Computes the derivatives of quantities with respect to tau
    '''
    gamma = Para.gamma
    sigma = Para.sigma
    theta_1 = Para.theta_1
    theta_2 = Para.theta_2
    c1 = get_c1(tau,R,Para)
    dc1 = dc1_dtau(tau,R,Para,c1)
    dc2 = R**(-1./sigma)*dc1
    dl1 = -(theta_1/gamma)*((1-tau)*theta_1)**(1./gamma-1)*c1**(-sigma/gamma) - (sigma/gamma)*((1-tau)*theta_1)**(1./gamma)*c1**(-sigma/gamma-1)*dc1
    dl2 = (theta_2*R/theta_1)**(1/gamma)*dl1
    return dc1,dc2,dl1,dl2
    
def DR(tau,R,Para):
    '''
    Computes the derivatives of quantities with respect to tau
    '''
    gamma = Para.gamma
    sigma = Para.sigma
    theta_1 = Para.theta_1
    theta_2 = Para.theta_2
    c1 = get_c1(tau,R,Para)
    c2 = R**(-1./sigma)*c1
    dc1 = dc1_dR(tau,R,Para,c1)
    dc2 = R**(-1./sigma)*dc1 - R**(-1./sigma-1)*c1/sigma
    dl1 = - (sigma/gamma)*((1-tau)*theta_1)**(1./gamma)*c1**(-sigma/gamma-1)*dc1
    dl2 = - (sigma/gamma)*((1-tau)*theta_2)**(1./gamma)*c2**(-sigma/gamma-1)*dc2
    return dc1,dc2,dl1,dl2

def getQuantities(tau,R,Para):
    '''
    Computes c1,c2,l1,l2 from tau and R
    '''
    gamma = Para.gamma
    sigma = Para.sigma
    theta_1 = Para.theta_1
    theta_2 = Para.theta_2
    c1 = get_c1(tau,R,Para)
    c2 = R**(-1./sigma)*c1
    l1 = ((1-tau)*theta_1)**(1./gamma)*c1**(-sigma/gamma)
    l2 = ((1-tau)*theta_2)**(1./gamma)*c2**(-sigma/gamma)
    return c1,c2,l1,l2
    
    
def getI(tau,R,Para):
    '''
    Computes I as a function f tau and R
    '''
    gamma = Para.gamma
    sigma = Para.sigma
    c1,c2,l1,l2 = getQuantities(tau,R,Para)
    return c2**(1-sigma)-l2**(1+gamma) - R*(c1**(1-sigma)-l1**(1+gamma))
    
def dIdtau(tau,R,Para):
    '''
    Computes I as a function f tau and R
    '''
    gamma = Para.gamma
    sigma = Para.sigma
    c1,c2,l1,l2 = getQuantities(tau,R,Para)
    dc1,dc2,dl1,dl2 = Dtau(tau,R,Para)
    return (1-sigma)*c2**(-sigma)*dc2-(1+gamma)*(l2**gamma)*dl2 - R*((1-sigma)*c1**(-sigma)*dc1-(1+gamma)*(l1**gamma)*dl1)

def dIdR(tau,R,Para):
    '''
    Computes I as a function f tau and R
    '''
    gamma = Para.gamma
    sigma = Para.sigma
    c1,c2,l1,l2 = getQuantities(tau,R,Para)
    dc1,dc2,dl1,dl2 = DR(tau,R,Para)
    return (1-sigma)*c2**(-sigma)*dc2-(1+gamma)*(l2**gamma)*dl2 - R*((1-sigma)*c1**(-sigma)*dc1-(1+gamma)*(l1**gamma)*dl1)-(c1**(1-sigma)-l1**(1+gamma))
    
def getUc1(tau,R,Para):
    '''
    Computes I as a function f tau and R
    '''
    sigma = Para.sigma
    c1,_,_,_ = getQuantities(tau,R,Para)
    return c1**(-sigma)
    
    
def getU(tau,R,Para):
    '''
    Computes U as a function of tau and R
    '''
    
    gamma = Para.gamma
    sigma = Para.sigma
    c1,c2,l1,l2 = getQuantities(tau,R,Para)
    return Para.alpha_2*(c2**(1-sigma)/(1-sigma)-l2**(1+gamma)/(1+gamma)) + Para.alpha_1*(c1**(1-sigma)/(1-sigma)-l1**(1+gamma)/(1+gamma))
    
    
def dUdtau(tau,R,Para):
    '''
    Computes U as a function of tau and R
    '''
    
    gamma = Para.gamma
    sigma = Para.sigma
    c1,c2,l1,l2 = getQuantities(tau,R,Para)
    dc1,dc2,dl1,dl2 = Dtau(tau,R,Para)
    return Para.alpha_2*(c2**(-sigma)*dc2-l2**gamma*dl2)+Para.alpha_1*(c1**(-sigma)*dc1-l1**gamma*dl1)
    
def dUdR(tau,R,Para):
    '''
    Computes U as a function of tau and R
    '''
    
    gamma = Para.gamma
    sigma = Para.sigma
    c1,c2,l1,l2 = getQuantities(tau,R,Para)
    dc1,dc2,dl1,dl2 = DR(tau,R,Para)
    return Para.alpha_2*(c2**(-sigma)*dc2-l2**gamma*dl2)+Para.alpha_1*(c1**(-sigma)*dc1-l1**gamma*dl1)
    
def mu(tau,R,Para):
    return dUdtau(tau,R,Para)/dIdtau(tau,R,Para)
    
def Rcon(tau,R,Para):
    '''
    Computes Rcon constraint
    '''
    mu_ = mu(tau,R,Para)
    dUdR_ = dUdR(tau,R,Para)
    dIdR_ = dIdR(tau,R,Para)
    Uc1 = getUc1(tau,R,Para)
    
    P = Para.P
    beta = Para.beta
    
    num = mu_*(dIdR_+beta*P.dot(dIdR_)/(1-beta)) - (dUdR_+beta*P.dot(dUdR_)/(1-beta))
    return num/Uc1
    
def getTau(R,Para):
    '''
    Compute tau to solve the bond constraint
    '''
    f = lambda tau: Rcon(tau,R,Para)
    return root(f,0.2).x
    
def SSfun(R,Para):
    '''
    Function the residual of which is the steady state
    '''
    tau = getTau(R,Para)
    I = getI(tau,R,Para)
    uc1 = getUc1(tau,R,Para)
    Euc1 = Para.P.dot(uc1)
    x = I/(uc1/Euc1-Para.beta)
    return x