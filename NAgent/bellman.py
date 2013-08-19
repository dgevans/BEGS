# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:20:49 2013

@author: dgevans
"""

import primitives
import inner_opt
from copy import deepcopy
from scipy.optimize import root
from scipy.optimize import fmin_slsqp
from numpy import *
from utilities import DictWrap
import itertools
from mpi4py import MPI
import sys

class BellmanMap:
    
    def __init__(self,Para):
        assert isinstance(Para,primitives.parameters)
        if isinstance(Para,primitives.BGP_parameters):
            self.Para = deepcopy(Para)
            self.io = inner_opt.BGP
        else:
            raise Exception('Para must be an instance of a type of primitive')
        self.U = self.Para.U
        self.Uc = self.Para.Uc
        self.Ul = self.Para.Ul
        self.Ucc = self.Para.Ucc
        self.Ull = self.Para.Ull
        self.getInitialGuess = None
            
    def __call__(self,Vf):
        self.Vf = Vf
        
        return self.maximizeObjective
        
    def getPolicies(self,z,x,R,s_):
        '''
        From a given z backout the policies
        '''
        Para = self.Para
        U = self.U
        P = Para.P
        beta = Para.beta
        alpha_1 = Para.alpha[0]
        alpha_i = Para.alpha[1:].reshape((-1,1))
        S = len(self.Para.P)
        N = len(self.Para.theta)
        c1 = z[:S]
        ci_ = z[S:].reshape((N-1,-1))
        ci = self.io.ComputeCi(c1,ci_,R,s_,Para)
        Rprime = self.io.ComputeR(c1,ci,Para)
        l1,li = self.io.Computel(c1,ci,Rprime,Para)
        xprime = self.io.ComputeXprime(c1,ci,Rprime,l1,li,x,s_,Para)
        
        Vprime = zeros(S)
        for s in range(0,S):
            state = hstack((xprime[:,s],Rprime[:,s]))
            Vprime[s] = self.Vf[s](state)
        
        V = P[s_,:].dot(alpha_1*U(c1,l1)+sum(alpha_i*U(ci,li),0)+beta*Vprime)
        return c1,ci,xprime,V
        
    def maximizeObjective(self,state):
        '''
        Maximizes the objective function
        '''
        if self.getInitialGuess ==None:
            raise Exception('BellmanMap.getInitialGuess must be set')
        z0 = self.getInitialGuess(state)
        policy = self.maximizeObjectiveFromGuess(state,z0)
        i = 0
        while policy == None and i < 100:
            policy = self.maximizeObjectiveFromGuess(state,z0*(0.4*random.rand(len(z0))+0.8))
            i += 1
        if i >= 100:
            return None
        else:
            return policy
            
        
        
    def maximizeObjectiveFromGuess(self,state,z_0):
        '''
            Maximizes the objective function trying the unconstrained method first
            and then moving on the unconstrained method if that fails
        '''

        #first try unconstrained maximization
        sol = self.maximizeObjectiveUnconstrained(state,z_0)
        if sol.success:
            return sol.policies
        else:
            #now solve the constrained maximization
            #fist solve at guess returned by unconstrained maximization
            if sol.policies != None:
                z,V = sol.policies
                ConSol = self.maximizeObjectiveConstrained(state,z)
                if ConSol.success and not isnan(ConSol.policies[1]):
                    return ConSol.policies
            #if maximizing with the result of unconstrained failed try initial guess
            ConSol = self.maximizeObjectiveConstrained(state,z_0)
            if ConSol.success and not isnan(ConSol.policies[1]):
                return ConSol.policies
        # if everything failed return None to indicate failer
        return None
        
    def maximizeObjectiveUnconstrained(self,state,z_0=None):
        '''
        Maximize the objective function.  First try FOC conditions then do constrained
        maximization
        '''
        N = len(self.Para.theta)
        x = state[0][:N-1].reshape((-1,1))
        R = state[0][N-1:].reshape((-1,1))
        s_ = state[1]
        #fist let's try to solve it by solving the unconstrained FOC
        #get initial guess
        z0 = z_0
        sol = root(self.io.GradObjectiveUncon,z0,(x,R,s_,self.Vf,self.Para),tol=1e-12)
        PR = sol.x
        if sol.success:
            c1,ci,xprime,V = self.getPolicies(PR,x,R,s_)
            #now check if 
            if all(xprime<=self.Para.xmax) and all(xprime>=self.Para.xmin): 
                #compute objective from c1,c2,xprime
                return DictWrap({'policies':(PR,V),'success':True})
            else:
                return DictWrap({'policies':(PR,V),'success':False})
        return DictWrap({'policies':None,'success':False})
        
    def maximizeObjectiveConstrained(self,state,z_0=None):
        '''
        Maximize the objective function.  First try FOC conditions then do constrained
        maximization
        '''
        io = self.io
        N = len(self.Para.theta)
        x = state[0][:N-1].reshape((-1,1))
        R = state[0][N-1:].reshape((-1,1))
        s_ = state[1]
        #fist let's try to solve it by solving the unconstrained FOC
        #get initial guess
        S = self.Para.P.shape[0]
        PR,_,_,imode,_ = fmin_slsqp(io.Objective,z_0,f_ieqcons=io.Constraint,fprime=io.GradObjectiveUncon,fprime_ieqcons=io.ConstraintJac,args=(x,R,s_,self.Vf,self.Para),
                        full_output=True,iprint=0,acc=1e-10,iter=1000)
        if imode == 0:
            c1,ci,xprime,V = self.getPolicies(PR[:S+(S-1)*(N-1)],x,R,s_)
            #now check if 
            return DictWrap({'policies':(PR,V),'success':True})
        return DictWrap({'policies':None,'success':False})
        
        
def getInitialGuess(state,Para,Policies):
    '''
    From the Policies choose the policy closest to to the state
    '''
    diff = inf
    policy = None
    for i,x in enumerate(Para.domain):
        if state[1] == x[1]:
            if linalg.norm(state[0]-x[0]) <diff:
                diff = linalg.norm(state[0]-x[0])
                policy = Policies[i]
    return policy