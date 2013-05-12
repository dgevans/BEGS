# -*- coding: utf-8 -*-
"""
Created on Sat May  4 13:59:07 2013

@author: dgevans
"""

import numpy as np

class parameters(object):
    
    beta = 0.9

    alpha_1 = 0.69
    
    alpha_2 = 0.31
    
    n1 = 1.0
    
    n2 = 1.0

    theta_1 = 3.3
    
    theta_2 = 1.0
    
    g = np.array([0.3194,0.3775])
    
    P = np.ones((2,2))/2.0
    
    
class CES_parameters(parameters):
    
    sigma = 2.0
    
    gamma = 2.0
    
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
    
    
    