# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:38:16 2013

@author: root
"""
'''
TEST 1
] Take the old code to compute the hessian on the value function [V_{x,x},V_{x,rho},V_{rho,x}V_{rho,rho}]

2] Compute the policy rule for x from the new code 

1/V_{x,x} ties up with the coefficients on mu for the policy rule for x 

-V_{x,rho} * 1/V_{x,x} ties up with the  coefficients on rho for the policy rule for x 

TEST 3
] Take the old code to compute the hessian on the value function [V_{x,x},V_{x,rho},V_{rho,x}V_{rho,rho}]

2] Compute the policy rule for x from the new code 

1/V_{x,x} ties up with the coefficients on mu for the policy rule for x 

-V_{x,rho} * 1/V_{x,x} ties up with the  coefficients on rho for the policy rule for x 


'''


import NAgents_mu as Na_mu
import NAgents as Na
import primitives
from numpy import *

Para2 = primitives.CES_parameters()
Para3 = primitives.CES_parameters()
Para2.theta = array([[3.5,3.],[1.1,1.]])
Para3.theta = array([[3.3,3.],[1.2,1.],[.9,1.]])
Para2.alpha = array([1.,2.])/3.
Para3.alpha = ones(3)
Para2.n = array([1.,2.])
Para3.n = ones(3)

Dyz,H,DzF,DyF,DvF,Dv,HV,MatrixEquation=Na.linearization(Para2,0,3)
 

V_xx=HV[0][0]
V_xrho=HV[0][1]


Dyz,H,DzF,DyF,DvF,Dv,HV,MatrixEquation=Na_mu.linearization(Para2,0,3)

x_pol_mu= Dyz[8][0]

x_pol_rho= Dyz[8][1]


print '--------------------- test 1 --------------------'


print '---------------------'
print V_xx**(-1)
print -V_xx**(-1)*V_xrho
print '---------------------'


print '---------------------'
print x_pol_mu
print x_pol_rho
print '---------------------'



Dyz,H,DzF,DyF,DvF,Dv,HV,MatrixEquation=Na_mu.linearization(Para3,0,3)

tempB=Dyz[3*2*2+2:3*2*2+2+8]
B_1=tempB[0::2]
B_2=tempB[1::2]
Bbar = Para3.P[0,0]*B_1+Para3.P[0,1]*B_2

print '--------------------- test 2 stability of eigen values for the mean transition--------------------'

print real(linalg.eigvals(Bbar))

print '--------------------- Locating the degeneracy --------------------'


Para3.theta = array([[3.3,3.],[.5,.5],[.9,.7]])
Dyz,H,DzF,DyF,DvF,Dv,HV,MatrixEquation=Na_mu.linearization(Para3,0,3)
print Dyz[12:14]
        
        