# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:59:07 2013

@author: dgevans
"""

import numpy as np

class parameters(object):
    
    beta = 0.9
    
    alpha_1 = 0.69
    
    alpha_2 = 0.15
    alpha_3=0.15
    alpha=np.array([alpha_1,2*alpha_2]);
    alpha=np.array([alpha_1,alpha_2,alpha_3]);
    
    n1 = 1.0    
    n2 = 0.5
    n3=0.5
    n=np.array([n1,2*n2])
    n=np.array([n1,n2,n3])
    
    theta_1 = 4
    theta_2 = .5
    theta_3=  .5
   
    theta=np.array([[theta_1, theta_1],[2*theta_2,2*theta_2]])
    theta=np.array([[theta_1, theta_1*1.03],[theta_2,theta_2*1.03],[theta_3,theta_3*1.03]])   
    g = np.array([0.3, 0.3])
    
    P = np.ones((2,2))/2.0
    
    
class CES_parameters(parameters):
    
    sigma = 1.0
    
    gamma = 1.0
    
    def U(self,c,l):
        if self.sigma == 1:
            return np.log(c)-l**(1.0+self.gamma)/(1.0+self.gamma)
        else:
            return c**(1.0-self.sigma)/(1.0 -self.sigma)-l**(1.0+self.gamma)/(1.0+self.gamma)
            
    def Uc(self,c):
        return c**(-self.sigma)
        
    def Ul(self,l):
        return -l**(self.gamma)
        
    def Ucc(self,c):
        return -self.sigma*c**(-self.sigma-1.0)
        
    def Ull(self,l):
        return -self.gamma*l**(self.gamma-1)
        

class BGP_parameters(parameters):
    
    sigma_1 = 1.0
    
    sigma_2 = 1.0
    
    psi = 0.6958
    
    def U(self,c,l):
        if self.sigma_1 == 1:
            U = self.psi*np.log(c)
        else:
            U = self.psi*c**(1.0-self.sigma_1)/(1.0-self.sigma_1)
        if self.sigma_2 == 1:
            U += (1.0-self.psi)*np.log(1.0-l)
        else:
            U += (1.0-self.psi)*(1.0-l)**(1.0-self.sigma_2)/(1.0-self.sigma_2)
        return U
        
    def Uc(self,c):
        return self.psi*c**(-self.sigma_1)
    
    def Ul(self,l):
        return -(1.0-self.psi)*(1.0-l)**(-self.sigma_2)
    
    def Ucc(self,c):
        return -self.psi*self.sigma_1*c**(-self.sigma_1-1.0)
        
    def Ull(self,l):
        return -(1.0-self.psi)*self.sigma_2*(1.0-l)**(-self.sigma_2-1.0)
    
    
    