# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:08:54 2013

@author: dgevans
"""
import primitives
import inner_opt
from copy import deepcopy
from scipy.optimize import root
from scipy.optimize import fmin_slsqp
import numpy as np
from utilities import DictWrap
from Spline import Spline
from utilities import ValueFunctionSpline
import itertools
from mpi4py import MPI
import sys

class BellmanMap:
    
    def __init__(self,Para):
        assert isinstance(Para,primitives.parameters)
        if isinstance(Para,primitives.BGP_parameters):
            self.Para = deepcopy(Para)
            self.io = inner_opt.BGP
            self.ioEdge = inner_opt.BGPEdge
        else:
            raise Exception('Para must be an instance of a type of primitive')
        self.U = self.Para.U
        self.Uc = self.Para.Uc
        self.Ul = self.Para.Ul
        self.Ucc = self.Para.Ucc
        self.Ull = self.Para.Ull
            
    def __call__(self,Vf,c1_policy,c2_policy,xprime_policy,Rprime_policy):
        self.Vf = Vf
        self.c1_policy = c1_policy
        self.c2_policy = c2_policy
        self.xprime_policy = xprime_policy
        self.Rprime_policy = Rprime_policy
        self.policies = {}
        
        return self.maximizeObjective
        
        
    def getPolicies(self,c1,c2_,x,R,s_):
        '''
        Computes xprime given c1 and c_2 using internal opitmization methods
        '''
        U = self.Para.U
        P = self.Para.P
        beta= self.Para.beta
        S = P.shape[0]        
        
        alpha_1 = self.Para.alpha_1
        alpha_2 = self.Para.alpha_2
        c1,c2,_,_ = self.io.ComputeC2(c1,c2_,R,s_,self.Para)
        Rprime,_ = self.io.ComputeR(c1,c2,0,0,self.Para)
        l1,_,l2,_ = self.io.Computel(c1,0,c2,0,Rprime,0,self.Para)
        xprime,_ = self.io.ComputeXprime(c1,0,c2,0,Rprime,0,l1,0,l2,0,x,s_,self.Para)
        
        Vprime = np.zeros(S)
        for s in range(0,S):
            Vprime[s] = self.Vf[s]([xprime[s],Rprime[s]])
        
        V = P[s_,:].dot(alpha_1*U(c1,l1)+alpha_2*U(c2,l2)+beta*Vprime)
        return c1,c2,xprime,V
    def getPoliciesEdge(self,c1,x,R,s_):
        '''
        Computes xprime given c1 and c_2 using internal opitmization methods
        '''
        U = self.Para.U
        P = self.Para.P
        beta= self.Para.beta
        S = P.shape[0]        
        
        alpha_1 = self.Para.alpha_1
        alpha_2 = self.Para.alpha_2
        c1,c2,_,_ = self.ioEdge.ComputeC2(c1,R,s_,self.Para)
        l1,_,l2,_ = self.ioEdge.Computel(c1,0,c2,0,R,self.Para)
        xprime,_ = self.ioEdge.ComputeXprime(c1,0,c2,0,l1,0,l2,0,x,R,s_,self.Para)
        
        Vprime = np.zeros(S)
        for s in range(0,S):
            Vprime[s] = self.Vf[s]([xprime[s],R])
        
        V = P[s_,:].dot(alpha_1*U(c1,l1)+alpha_2*U(c2,l2)+beta*Vprime)
        return c1,c2,xprime,V    
        
    def maximizeObjective(self,state,z_0=None):
        '''
            Maximizes the objective function trying the unconstrained method first
            and then moving on the unconstrained method if that fails
        '''
        if self.policies.has_key(state):
            return self.policies[state]
        if state[1] != self.Para.Rmin and state[1] != self.Para.Rmax:
            #first try unconstrained maximization
            sol = self.maximizeObjectiveUnconstrained(state,z_0)
            if sol.success:
                self.policies[state] = sol.policies
                return sol.policies
            else:
                #now solve the constrained maximization
                #fist solve at guess returned by unconstrained maximization
                if sol.policies != None:
                    c1,c2,xprime,V = sol.policies
                    ConSol = self.maximizeObjectiveConstrained(state,np.hstack((c1,c2,xprime)))
                    if ConSol.success:
                        self.policies[state] = ConSol.policies
                        return ConSol.policies
                #if maximizing with the result of unconstrained failed try initial guess
                ConSol = self.maximizeObjectiveConstrained(state,z_0)
                if ConSol.success:
                    self.policies[state] = ConSol.policies
                    return ConSol.policies
            # if everything failed return None to indicate failer
            return None
        else:
            #first try unconstrained maximization
            sol = self.maximizeObjectiveUnconstrainedEdge(state,z_0)
            if sol.success:
                self.policies[state] = sol.policies
                return sol.policies
            else:
                #now solve the constrained maximization
                #fist solve at guess returned by unconstrained maximization
                if sol.policies != None:
                    c1,c2,xprime,V = sol.policies
                    ConSol = self.maximizeObjectiveConstrainedEdge(state,c1)
                    if ConSol.success:
                        self.policies[state] = ConSol.policies
                        return ConSol.policies
                #if maximizing with the result of unconstrained failed try initial guess
                ConSol = self.maximizeObjectiveConstrainedEdge(state,z_0)
                if ConSol.success:
                    self.policies[state] = ConSol.policies
                    return ConSol.policies
            # if everything failed return None to indicate failer
            return None
        
        
    def maximizeObjectiveUnconstrained(self,state,z_0=None):
        '''
        Maximize the objective function.  First try FOC conditions then do constrained
        maximization
        '''
        x = state[0]
        R = state[1]
        s_ = state[2]
        #fist let's try to solve it by solving the unconstrained FOC
        #get initial guess
        S = self.Para.P.shape[0]
        z0 = np.zeros(2*S-1)        
        if z_0 == None:
            for s in range(0,S):
                z0[s] = self.c1_policy[(s_,s)]([x,R])
                if s < S-1:
                    z0[S+s] = self.c2_policy[(s_,s)]([x,R])
        else:
            z0 = z_0
        sol = root(self.io.GradObjectiveUncon,z0,(x,R,s_,self.Vf,self.Para),tol=1e-12)
        PR = sol.x
        if sol.success:
            c1 = PR[0:S]
            c2_ = PR[S:2*S-1]
            c1,c2,xprime,V = self.getPolicies(c1,c2_,x,R,s_)
            Rprime = self.Para.Uc(c2)/self.Para.Uc(c1)
            #now check if 
            if np.all(xprime<=self.Para.xmax) and np.all(xprime>=self.Para.xmin) and \
            np.all(Rprime <= self.Para.Rmax) and np.all(Rprime >= self.Para.Rmin): 
                #compute objective from c1,c2,xprime
                return DictWrap({'policies':(c1,c2,xprime,V),'success':True})
            else:
                return DictWrap({'policies':(c1,c2,xprime,V),'success':False})
        return DictWrap({'policies':None,'success':False})
        
    def maximizeObjectiveConstrained(self,state,z_0=None):
        '''
        Maximize the objective function using constrained optimization
        '''
        x = state[0]
        R = state[1]
        s_ = state[2]
        Para = self.Para
        S  = Para.P.shape[0]
        z0 = np.zeros(2*S)
        #if we don't have an initial guess, take initial guess from policies
        if z_0 == None:
            for s in range(0,S):
                z0[s] = self.c1_policy[(s_,s)]([x,R])
                z0[S+s] = self.c2_policy[(s_,s)]([x,R])
        else:
            if len(z_0) == 2*S:
                z0 = z_0
            else:
                c1,c2,_,_ = self.getPolicies(z_0[0:S],z_0[S:2*S-1],x,R,s_)
                z0 = np.vstack((c1,c2))
        
        #create bounds
        if isinstance(Para,primitives.BGP_parameters):
            cbounds = zip(np.zeros(S),Para.theta_1+Para.theta_2-Para.g)*2
        else:
            cbounds = [(0,100)]*2*S
        bounds = cbounds
        #perfom minimization
        policy,minusV,_,imode,smode = fmin_slsqp(self.io.ConstrainedObjective,z0,f_ieqcons=self.io.ieq_cons,eqcons=[self.io.eq_con],bounds=bounds,
                   fprime_eqcons=self.io.eq_conJacobian,fprime_ieqcons=self.io.ieq_consJacobian,args=(x,R,s_,self.Vf,Para),iter=1000,
                    acc=1e-12,disp=0,full_output=True)
        
        policies = self.getPolicies(policy[0:S],policy[S:2*S-1],x,R,s_)
        if imode == 0:
            return DictWrap({'policies':policies,'success':True})
        print smode
        return DictWrap({'policies':policies,'success':False})
        
    def maximizeObjectiveUnconstrainedEdge(self,state,z_0=None):
        '''
        Maximize the objective function.  First try FOC conditions then do constrained
        maximization
        '''
        x = state[0]
        R = state[1]
        s_ = state[2]
        #fist let's try to solve it by solving the unconstrained FOC
        #get initial guess
        S = self.Para.P.shape[0]
        z0 = np.zeros(S)        
        if z_0 == None:
            for s in range(0,S):
                z0[s] = self.c1_policy[(s_,s)]([x,R])
        else:
            z0 = z_0[0:S]
        sol = root(self.ioEdge.GradObjectiveUncon,z0,(x,R,s_,self.Vf,self.Para),tol=1e-12)
        PR = sol.x
        if sol.success:
            c1 = PR[0:S]
            c1,c2,xprime,V = self.getPoliciesEdge(c1,x,R,s_)
            #now check if 
            if np.all(xprime<=self.Para.xmax) and np.all(xprime>=self.Para.xmin):
                #compute objective from c1,c2,xprime
                return DictWrap({'policies':(c1,c2,xprime,V),'success':True})
            else:
                return DictWrap({'policies':(c1,c2,xprime,V),'success':False})
        return DictWrap({'policies':None,'success':False})
        
    def maximizeObjectiveConstrainedEdge(self,state,z_0=None):
        '''
        Maximize the objective function using constrained optimization
        '''
        x = state[0]
        R = state[1]
        s_ = state[2]
        Para = self.Para
        S  = Para.P.shape[0]
        z0 = np.zeros(S)
        #if we don't have an initial guess, take initial guess from policies
        if z_0 == None:
            for s in range(0,S):
                z0[s] = self.c1_policy[(s_,s)]([x,R])
        else:
            z0 = z_0[0:S]
        
        #create bounds
        if isinstance(Para,primitives.BGP_parameters):
            cbounds = zip(np.zeros(S),Para.theta_1+Para.theta_2-Para.g)
        else:
            cbounds = [(0,100)]*S
        bounds = cbounds
        #perfom minimization
        policy,minusV,_,imode,smode = fmin_slsqp(self.ioEdge.ConstrainedObjective,z0,f_ieqcons=self.ioEdge.ieq_cons,bounds=bounds,
                   fprime_ieqcons=self.ioEdge.ieq_consJacobian,args=(x,R,s_,self.Vf,Para),iter=1000,
                    acc=1e-12,disp=0,full_output=True)
        
        policies = self.getPoliciesEdge(policy[0:S],x,R,s_)
        if imode == 0:
            return DictWrap({'policies':policies,'success':True})
        print smode
        return DictWrap({'policies':policies,'success':False})
                

def approximateValueFunctionAndPolicies(Vfnew,Para):
    '''
    Approximates the new value function along with associated policie
    '''
    domain = Para.domain
    policies = map(Vfnew,domain)
    
    allSolved = all([policy != None for policy in policies])
    
    return fitNewPolicies(domain,policies,Para),allSolved
    
def approximateValueFunctionAndPoliciesMPI(Vfnew,Para):
    '''
    Approximates the new value function along with associated policie
    '''    
    comm = MPI.COMM_WORLD
    #first split up domain for each 
    s = comm.Get_size()
    rank = comm.Get_rank()
    n = len(Para.domain)
    m = n/s
    r = n%s
    mydomain = Para.domain[rank*m+min(rank,r):(rank+1)*m+min(rank+1,r)]

    mypolicies = map(Vfnew,mydomain)
    chunked_policies = comm.allgather(mypolicies) #gather all the policies at master 
    policies = list(itertools.chain.from_iterable(chunked_policies))
    allSolved = all([policy != None for policy in policies])
    policyFunctions = fitNewPolicies(Para.domain,policies,Para)

    
    return comm.bcast(policyFunctions,root=0),allSolved
    
        
def solveBellman(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy,Para):
    
    T = BellmanMap(Para)    
    S = Para.P.shape[0]
    Vs_old = []
    domain = np.vstack(filter(lambda state: state[2] == 0,Para.domain))[:,0:2]
    for s_ in range(0,S):
        Vs_old.append(Vf[s_](domain))
        
    niter = 100
    Vf_old = Vf
    for t in range(0,niter):
        
        Vfnew = T(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy)
        (Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy), allSolved = approximateValueFunctionAndPolicies(Vfnew,Para)
        
        while not allSolved:
            print "iterating"
            Vfnew = T(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy)
            (Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy), allSolved = approximateValueFunctionAndPolicies(Vfnew,Para)

        
        diff = 0
        for s_ in range(0,S):
            diff = max(diff,np.linalg.norm(Vf[s_](domain)-Vfprime[s_](domain))/len(domain))
            #print domain[np.abs(Vs_old[s_]-Vfprime[s_](domain)).argmax(),:]
            Vs_old[s_] = Vf[s_](domain)
        print diff
        #if diff > 2*diff_old and t >50:
        #    return Vf,Vf_old,Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy
        Vf_old = Vf
        Vf= Vfprime
    return Vf,Vf_old,Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy
    
def solveBellmanMPI(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy,Para):
    '''
    Solves the bellman equation using MPI
    '''
    comm = MPI.COMM_WORLD
    T = BellmanMap(Para)    
    S = Para.P.shape[0]
    Vs_old = []
    domain = np.vstack(filter(lambda state: state[2] == 0,Para.domain))[:,0:2]
    for s_ in range(0,S):
        Vs_old.append(Vf[s_](domain))
        
    niter = 1000
    for t in range(0,niter):
        
        Vfnew = T(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy)
        (Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy), allSolved = approximateValueFunctionAndPoliciesMPI(Vfnew,Para)
        
        while not allSolved:
            print "iterating"
            Vfnew = T(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy)
            (Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy), allSolved = approximateValueFunctionAndPoliciesMPI(Vfnew,Para)

        
        diff = 0
        for s_ in range(0,S):
            diff = max(diff,np.linalg.norm(Vf[s_](domain)-Vfprime[s_](domain))/len(domain))
            #print domain[np.abs(Vs_old[s_]-Vfprime[s_](domain)).argmax(),:]
            Vs_old[s_] = Vf[s_](domain)
        if comm.Get_rank() == 0:
            print diff
            sys.stdout.flush()
        #if diff > 2*diff_old and t >50:
        #    return Vf,Vf_old,Vfprime,c1_policy,c2_policy,Rprime_policy,xprime_policy
        Vf= Vfprime
        if diff < 1e-8:
            break
    return Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy


def fitNewPolicies(domain,policies,Para):
    S = Para.P.shape[0]
    merged = zip(domain,policies) #merge policies with domain so we can pull out policie that did not work
    c1_policy,c2_policy,Rprime_policy,xprime_policy = {},{},{},{} #initialize new policies as empty dicts
    Vf = []
    
    for s_ in range(0,S):
        #filter out the states where the maximization does not work
        s_domain,s_policies = zip(* filter(lambda x: x[1] != None and x[0][2]==s_, merged) )
        s_domain = np.vstack(s_domain)[:,0:2]
        c1New,c2New,xprimeNew,Vnew = map(np.vstack,zip(*s_policies)) #unpack s_policis and then stacks them into a matrix
        RprimeNew = Para.Uc(c2New)/Para.Uc(c1New)
        Vnew= Vnew.reshape(-1) #make 1d array        
        
        #fit policies
        beta = (Para.P[s_,:]*Para.beta).sum()
        if isinstance(Para,primitives.BGP_parameters):
            sigma = Para.sigma_1
        else:
            sigma = Para.sigma
        Vf.append(ValueFunctionSpline(s_domain,np.hstack(Vnew),Para.approxOrder,sigma,beta))
        for s in range(0,S):
            c1_policy[(s_,s)] = Spline(s_domain,c1New[:,s],[1,1])
            c2_policy[(s_,s)] = Spline(s_domain,c2New[:,s],[1,1])
            Rprime_policy[(s_,s)] = Spline(s_domain,RprimeNew[:,s],[1,1])
            xprime_policy[(s_,s)] = Spline(s_domain,xprimeNew[:,s],[1,1])
            
            
    return Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy
        