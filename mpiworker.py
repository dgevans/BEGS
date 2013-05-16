# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:31:16 2013

@author: dgevans
"""
import sys
sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
from mpi4py import MPI
import Bellman
import initialize
print "solving bellman"
comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
print rank
print size
Para = []
Para = comm.bcast(Para,root=0)
V0 = lambda state: initialize.completeMarketsSolution(state,Para)
(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy),_ = Bellman.approximateValueFunctionAndPoliciesMPI(V0,Para)

policies = Bellman.solveBellmanMPI(Vf,c1_policy,c2_policy,Rprime_policy,xprime_policy,Para)
if rank == 0:
    print "done"
    comm.send(policies,tag=Para.POLICY_TAG)

