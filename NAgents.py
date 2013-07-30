# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:43:46 2013

@author: dgevans
"""
from numpy import *
from copy import deepcopy
from scipy.optimize import root
import pycppad as ad
import pdb
import numdifftools as nd

def SSresiduals(z,Para):
    '''
    Steady State residuals
    '''
    theta_1 = Para.theta[0,:]
    theta_i = Para.theta[1:,:]
    alpha_1 = Para.alpha[0]
    alpha_i = Para.alpha[1:].reshape((-1,1))
    n_1 = Para.n[0]
    n_i = Para.n[1:].reshape((-1,1))
    beta = Para.beta
    g = Para.g
    
    c1,ci,l1,li,x_i,rho_i,mu_i,lambda_i,xi,phi_i,eta_i = getSSQuantities(z,Para)
    
    uc1 = Para.Uc(c1)
    ucc1 = Para.Ucc(c1)
    uci = Para.Uc(ci)
    Euci = (Para.P.dot(uci.T)).T
    ucci = Para.Ucc(ci)
    ul1 = Para.Ul(l1)
    ull1 = Para.Ull(l1)
    uli = Para.Ul(li)
    ulli = Para.Ull(li)
     
    res = array([])
    con = x_i*(uci/Euci-beta) - (uci*ci+uli*li)+rho_i*(uc1*c1+ul1*l1)
    res = hstack((res,con.flatten()))
    
    con = rho_i*ul1/theta_1-uli/theta_i
    res = hstack((res,con.flatten()))
    
    con = n_1*theta_1*l1 + sum(n_i*theta_i*li,axis=0)-g-n_1*c1-sum(n_i*ci,0)
    res = hstack((res,con.flatten()))
    
    con = rho_i*uc1-uci
    res = hstack((res,con.flatten()))
    
    foc = alpha_i*uci-mu_i*( ucci*ci+uci )-n_i*xi-eta_i*ucci
    res = hstack((res,foc.flatten()))
    
    foc = alpha_1*uc1+sum(mu_i*rho_i,0)*( ucc1*c1+uc1 )-n_1*xi + sum(eta_i*rho_i,0)*ucc1
    res = hstack((res,foc.flatten()))
    
    foc = alpha_i*uli - mu_i*( ulli*li+uli ) - phi_i*ulli/theta_i + n_i*theta_i*xi
    res = hstack((res,foc.flatten()))
    
    foc = alpha_1*ul1 + sum(mu_i*rho_i,0)*( ull1*l1 + ul1 ) + sum(phi_i*rho_i,0)*ull1/theta_1 + n_1*theta_1*xi
    res = hstack((res,foc.flatten()))
    
    foc = lambda_i*Euci*( uci/Euci-beta )+mu_i*( uc1*c1+ul1*l1 ) + phi_i*ul1/theta_1 + eta_i*uc1
    res = hstack((res,foc.flatten()))
    
    return res
    
def getSSQuantities(z,Para):
    '''
    Gets quantities from z
    '''
    N = len(Para.theta)
    S = len(Para.P)
    c1 = z[0:S]
    
    ci = z[S:N*S].reshape((N-1,S))
    
    l1 = z[N*S:(N+1)*S]
    
    li = z[(N+1)*S:2*N*S].reshape((N-1,S))
    
    zi = 2*N*S
    x_i = z[zi:zi+N-1].reshape((N-1,1))
    zi+= N-1
    
    rho_i = z[zi:zi+N-1].reshape((N-1,1))
    zi += N-1
    
    mu_i = z[zi:zi+N-1].reshape((N-1,1))
    zi += N-1
    
    lambda_i = z[zi:zi+N-1].reshape((N-1,1))
    zi += N-1

    xi = z[zi:zi+S].reshape(S)
    zi += S    
    
    phi_i = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    eta_i = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    return c1,ci,l1,li,x_i,rho_i,mu_i,lambda_i,xi,phi_i,eta_i
    
def getSSNz(Para):
    '''
    Finds the number of elements in the steady state
    '''
    N = len(Para.theta)
    S = len(Para.P)
    zi = 2*N*S
    zi+= N-1
    zi += N-1
    zi += N-1
    zi += N-1
    zi += S    
    zi += S*(N-1)
    zi += S*(N-1)
    return zi
    
def findSteadyState(Para,x0,rho0):
    '''
    Finds the steady state for given Para
    '''
    Para2 = deepcopy(Para)
    Para2.theta = zeros(Para.theta.shape)
    Para2.theta[0,:] = Para.theta[0,:]
    Para2.theta[1:,:] = Para.theta[1,:]
    z0 = getInitialGuess(Para2,x0,rho0)
    res = root(lambda z: SSresiduals(z,Para2),z0)
    if not res.success:
        raise Exception(res.message)
    z1 = res.x
    
    res = root(lambda z: SSresiduals(z,Para),z1,tol=1e-12)
    if not res.success:
        raise Exception(res.message)
    return res.x
    
def getInitialGuess(Para,x,rho):
    '''
    Find the steady State
    '''
    N = len(Para.theta)
    S = len(Para.P)
    Para2 = deepcopy(Para)
    Para2.theta = Para.theta[0:2,:]
    Para2.n = array([Para.n[0],sum(Para.n[1:])])
    Para2.alpha = array([Para.alpha[0],sum(Para.alpha[1:])])
    def f(q):
        z = zeros(getSSNz(Para2))
        z[0:4*S+2] = hstack((q,[x,rho]))
        return SSresiduals(z,Para2)[0:4*S]
    res = root(f,0.5*ones(4*S))
    if not res.success:
        raise Exception(res.message)
    q=res.x
    z = zeros(getSSNz(Para))
    z[0:S] = q[0:S]
    z[S:N*S] = tile(q[S:2*S],N-1)
    z[N*S:N*S+S] = q[2*S:3*S]
    z[N*S+S:2*N*S] = tile(q[3*S:4*S],N-1)
    z[2*N*S:2*N*S+N-1] = x
    z[2*N*S+N-1:2*N*S+2*(N-1)] = rho
    return z
        
        
def FOCResiduals(z,x_i,rho_i,V_x,V_rho,Para):
    '''
    Compute the residuals of the first order 
    '''
    N = len(Para.theta)
    theta_1 = Para.theta[0,:]
    theta_i = Para.theta[1:,:]
    alpha_1 = Para.alpha[0]
    alpha_i = Para.alpha[1:].reshape((-1,1))
    n_1 = Para.n[0]
    n_i = Para.n[1:].reshape((-1,1))
    beta = Para.beta
    g = Para.g
    P = Para.P[0,:]
    
    c1,ci,l1,li,xprime,rhoprime,mu_i,lambda_i,xi,phi_i,eta_i = getFOCQuantities(z,Para)
    
    uc1 = Para.Uc(c1)
    ucc1 = Para.Ucc(c1)
    uci = Para.Uc(ci)
    Euci = uci.dot(P).reshape((N-1,-1))
    Emu_uci = (mu_i*uci).dot(P).reshape((N-1,-1))
    ucci = Para.Ucc(ci)
    ul1 = Para.Ul(l1)
    ull1 = Para.Ull(l1)
    uli = Para.Ul(li)
    ulli = Para.Ull(li)
    
    res = alpha_1*uc1 + sum(rhoprime*mu_i,0)*(ucc1*c1+uc1) - n_1*xi + sum(eta_i*rhoprime,0)*ucc1
    res = res.flatten()
    
    foc = alpha_i*uci -mu_i*( ucci*ci+uci ) + x_i*ucci/Euci*( mu_i-Emu_uci/Euci )\
    +lambda_i*ucci*(rhoprime-rho_i)-n_i*xi-eta_i*ucci
    res = hstack((res,foc.flatten()))

    foc = alpha_1*ul1 + sum(mu_i*rhoprime,0)*(ull1*l1+ul1) + sum(phi_i*rhoprime,0)*ull1/theta_1 + n_1*theta_1*xi
    res = hstack((res,foc.flatten()))
    
    foc = alpha_i*uli -mu_i*( ulli*li+uli ) - phi_i*ulli/theta_i + n_i*theta_i*xi
    res = hstack((res,foc.flatten()))

    foc = beta*V_x-beta*mu_i
    res = hstack((res,foc.flatten()))    
    
    foc = beta*V_rho + mu_i*(uc1*c1+ul1*l1) + phi_i*ul1/theta_1 + lambda_i*uci + eta_i*uc1
    res = hstack((res,foc.flatten()))
    
    con = x_i*uci/Euci - beta*xprime - (uci*ci+uli*li) + rhoprime*( uc1*c1+ul1*l1 )
    res = hstack((res,con.flatten()))
    
    con = (uci*(rhoprime-rho_i)).dot(P)
    res = hstack((res,con.flatten()))
    
    con = rhoprime*ul1/theta_1-uli/theta_i
    res = hstack((res,con.flatten()))
    
    con = n_1*theta_1*l1+sum(n_i*theta_i*li,0)-g-c1*n_1 - sum(n_i*ci,0)
    res = hstack((res,con.flatten()))
    
    con = rhoprime*uc1-uci
    res = hstack((res,con.flatten()))
    
    
    return res
    
def envelopeCondition(z,Para):
    '''
    Computes V_x and V_rho from the envelope condition
    '''
    N = len(Para.theta)
    P = Para.P[0,:]
    c1,ci,l1,li,xprime,rhoprime,mu_i,lambda_i,xi,phi_i,eta_i = getFOCQuantities(z,Para)
    
    uci = Para.Uc(ci)
    Euci = uci.dot(P).reshape((N-1,-1))
    Emu_uci = (mu_i*uci).dot(P).reshape((N-1,-1))
    
    V_x = Emu_uci/Euci
    V_rho = -lambda_i*Euci
    return V_x,V_rho
        
def getFOCQuantities(z,Para):
    '''
    Gets quantities from z
    '''
    N = len(Para.theta)
    S = len(Para.P)
    c1 = z[0:S]
    
    ci = z[S:N*S].reshape((N-1,S))
    
    l1 = z[N*S:(N+1)*S]
    
    li = z[(N+1)*S:2*N*S].reshape((N-1,S))
    
    zi = 2*N*S
    x_prime = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi+= S*(N-1)
    
    rho_prime = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    mu_i = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    lambda_i = z[zi:zi+N-1].reshape((N-1,1))
    zi += N-1

    xi = z[zi:zi+S].reshape(S)
    zi += S    
    
    phi_i = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    eta_i = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    return c1,ci,l1,li,x_prime,rho_prime,mu_i,lambda_i,xi,phi_i,eta_i
    
def getFOCNz(Para):
    '''
    Computes number of elements in z for FOC equations
    '''
    N = len(Para.theta)
    S = len(Para.P)
    zi = 2*N*S
    zi += S*(N-1)
    zi += S*(N-1)
    zi += S*(N-1)
    zi += N-1
    zi += S
    zi += S*(N-1)
    zi += S*(N-1)
    return zi
    
def SSz_to_FOCz(SSz,Para):
    '''
    Transforms the quantities and multipliers of the steady state into the quantities and
    multipliers  
    '''
    N = len(Para.theta)
    S = len(Para.P)
    q = SSz[0:2*S*N]
    zi = 2*N*S    
    x_i = SSz[zi:zi+N-1].reshape((N-1,1))
    xprime = tile(x_i,S)
    zi+= N-1
    
    rho_i = SSz[zi:zi+N-1].reshape((N-1,1))
    rhoprime = tile(rho_i,S)
    zi += N-1
    
    mu_i = SSz[zi:zi+N-1].reshape((N-1,1))
    mu_i = tile(mu_i,S)
    zi += N-1

    mult = SSz[zi:]    
    
    return hstack((q,xprime.flatten(),rhoprime.flatten(),mu_i.flatten(),mult))
    
    
def linearization(Para,x0,rho0):
    '''
    Finds and computes the linearization around the steady state
    '''
    
    N = len(Para.theta)
    S = len(Para.P)
    
    assert S==2    
    
    SSz = findSteadyState(Para,x0,rho0)
    _,_,_,_,x_i,rho_i,_,_,_,_,_ = getSSQuantities(SSz,Para)
    zbar = SSz_to_FOCz(SSz,Para)
    ybar = hstack((x_i.flatten(),rho_i.flatten()))
    V_x,V_rho = envelopeCondition(zbar,Para)
    vbar = tile(hstack((V_x.flatten(),V_rho.flatten())),2)
    Phi = getPhi(N,len(zbar))
    
    #define functions to use when taking derivatives
    def v_fun(z):
        V_x,V_rho = envelopeCondition(z,Para)
        return hstack((V_x.flatten(),V_rho.flatten()))
    def F(z,y,v):
        x_i = y[0:N-1].reshape((N-1,1))
        rho_i = y[N-1:2*(N-1)].T.reshape((N-1,1))
        V_x = vstack((v[0:N-1],v[2*(N-1):3*(N-1)])).T
        V_rho = vstack((v[N-1:2*(N-1)],v[3*(N-1):4*(N-1)])).T
        return FOCResiduals(z,x_i,rho_i,V_x,V_rho,Para)
        
    #take derivatives using auto-differentiation
    '''
    a_z = ad.independent(zbar)
    a_F = F(a_z,ybar,vbar)
    DzF = ad.adfun(a_z,a_F).jacobian(zbar)
    
    a_y = ad.independent(ybar)
    a_F = F(zbar,a_y,vbar)
    DyF = ad.adfun(a_y,a_F).jacobian(ybar)
    
    a_v = ad.independent(vbar)
    a_F = F(zbar,ybar,a_v)
    DvF = ad.adfun(a_v,a_F).jacobian(vbar)
    
    a_z = ad.independent(zbar)
    a_F = v(zbar)
    Dv = ad.adfun(a_z,a_F).jacobian(zbar)'''
    DzF = nd.Jacobian(lambda z: F(z,ybar,vbar))(zbar)
    DyF = nd.Jacobian(lambda y: F(zbar,y,vbar))(ybar)
    DvF = nd.Jacobian(lambda v: F(zbar,ybar,v))(vbar)
    Dv = nd.Jacobian(v_fun)(zbar)
    '''
    def MatrixEquation(Dyz_flat):
        Dyz = Dyz_flat.reshape((len(zbar),len(ybar)))
        return (DzF.dot(Dyz)+DyF+DvF.dot(kron(eye(2),Dv).dot(kron(eye(2),Dyz).dot(Phi.dot(Dyz))))).flatten()
        
    def MatrixEquation2(Bflat,test=None):
        B = Bflat.reshape((2*len(ybar),len(ybar)))
        def f(Dyz_flat):
            Dyz = Dyz_flat.reshape((len(zbar),len(ybar)))
            return (DzF.dot(Dyz) + DvF.dot(kron(eye(2),Dv).dot(kron(eye(2),Dyz).dot(B)))).flatten()
        nM = len(zbar)*len(ybar)
        M = zeros((nM,nM))
        for i in range(0,nM):
            M[:,i] = f(eye(nM)[i,:])
        Dyz_flat = linalg.solve(M,-DyF.flatten())
        pdb.set_trace()
        return Phi.dot(Dyz_flat.reshape((len(zbar),len(ybar)))).flatten()-Bflat
    '''    
    def MatrixEquation(Aflat):
        A = Aflat.reshape((len(ybar),len(ybar)))
        M = DzF+DvF.dot(kron(eye(2),A)).dot(Phi)
        Dyz = linalg.solve(M,-DyF)
        return (Dv.dot(Dyz)).flatten()-Aflat
    diff = 1
    A = 0
    for i in range(0,5000):
        A0 = random.randn(len(ybar)**2)
        res = root(MatrixEquation,A0)
        if res.success:
            if max(abs(MatrixEquation(res.x))) < diff:
                A = res.x
                diff = max(abs(MatrixEquation(res.x)))
                print diff    
    M = DzF+DvF.dot(kron(eye(2),A.reshape((len(ybar),len(ybar))))).dot(Phi)
    Dyz = linalg.solve(M,-DyF)

    H = (DzF +(DvF.dot(kron(eye(2),Dv)).dot(kron(eye(2),Dyz)).dot(Phi)))#Constructs the bordered Hessian
    return Dyz,H,DzF,DyF,DvF,Dv,MatrixEquation
    
def Check2ndOrder(Para,x0,rho0):
    '''
    Check the Bordered Hessian 2nd order conition
    '''
    N = len(Para.theta)
    S = len(Para.P)
    nQuant = 2*S*N+2*(N-1)*S
    k = getFOCNz(Para)-nQuant
    Nz = getFOCNz(Para)
    test = []
    Dyz,H = linearization(Para,x0,rho0)[0:2]
    H = fliplr(flipud(H)) #need to flip it to get it to match statement f the proof
    #(as it currently set up took derivatives of quantities first then Lagrange Multipliers )
    for j in range(2*k+1,Nz+1):
        test.append((-1)**(j-k)*linalg.det(H[:j,:j]))
    return all(test>0)
        
    
def getPhi(N,Nz):
    '''
    Computes the Phi matrix
    '''
    S = 2
    Phi_0 = zeros((2*(N-1),Nz))
    Phi_1 = zeros((2*(N-1),Nz))
    base = 2*2*N
    for i in range(0,N-1):
        Phi_0[i,base+i*S] = 1
        Phi_0[(N-1)+i,base+(N-1)*S+i*S] = 1
        Phi_1[i,base+i*S+1] = 1
        Phi_1[(N-1)+i,base+(N-1)*S+i*S+1] = 1
    return vstack((Phi_0,Phi_1))
    
def CMResiduals(z,x_i,rho_i,Para):
    '''
    Computest the complete market solution for the iid case
    '''
    theta_1 = Para.theta[0,:]
    theta_i = Para.theta[1:,:]
    alpha_1 = Para.alpha[0]
    alpha_i = Para.alpha[1:].reshape((-1,1))
    n_1 = Para.n[0]
    n_i = Para.n[1:].reshape((-1,1))
    beta = Para.beta
    g = Para.g
    P = Para.P[0,:]
    
    c1,ci,l1,li,mu_i,xi,phi_i,eta_i = getCMQuantities(z,Para)
    
    uc1 = Para.Uc(c1)
    ucc1 = Para.Ucc(c1)
    uci = Para.Uc(ci)
    ucci = Para.Ucc(ci)
    ul1 = Para.Ul(l1)
    ull1 = Para.Ull(l1)
    uli = Para.Ul(li)
    ulli = Para.Ull(li)
     
    res = array([])
    con = x_i - ((uci*ci+uli*li)-rho_i*(uc1*c1+ul1*l1)).dot(P).reshape((-1,1))/(1.-beta)
    res = hstack((res,con.flatten()))
    
    con = rho_i*ul1/theta_1-uli/theta_i
    res = hstack((res,con.flatten()))
    
    con = n_1*theta_1*l1 + sum(n_i*theta_i*li,axis=0)-g-n_1*c1-sum(n_i*ci,0)
    res = hstack((res,con.flatten()))
    
    con = rho_i*uc1-uci
    res = hstack((res,con.flatten()))
    
    foc = alpha_i*uci-mu_i*( ucci*ci+uci )-n_i*xi-eta_i*ucci
    res = hstack((res,foc.flatten()))
    
    foc = alpha_1*uc1+sum(mu_i*rho_i,0)*( ucc1*c1+uc1 )-n_1*xi + sum(eta_i*rho_i,0)*ucc1
    res = hstack((res,foc.flatten()))
    
    foc = alpha_i*uli - mu_i*( ulli*li+uli ) - phi_i*ulli/theta_i + n_i*theta_i*xi
    res = hstack((res,foc.flatten()))
    
    foc = alpha_1*ul1 + sum(mu_i*rho_i,0)*( ull1*l1 + ul1 ) + sum(phi_i*rho_i,0)*ull1/theta_1 + n_1*theta_1*xi
    res = hstack((res,foc.flatten()))
    
    return res

def getCMQuantities(z,Para):
    '''
    Gets quantities from z
    '''
    N = len(Para.theta)
    S = len(Para.P)
    c1 = z[0:S]
    
    ci = z[S:N*S].reshape((N-1,S))
    
    l1 = z[N*S:(N+1)*S]
    
    li = z[(N+1)*S:2*N*S].reshape((N-1,S))
    
    zi = 2*N*S
    
    mu_i = z[zi:zi+(N-1)].reshape((N-1,1))
    zi += (N-1)

    xi = z[zi:zi+S].reshape(S)
    zi += S    
    
    phi_i = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    eta_i = z[zi:zi+S*(N-1)].reshape((N-1,S))
    zi += S*(N-1)
    
    return c1,ci,l1,li,mu_i,xi,phi_i,eta_i
    
def SSz_to_CMz(z,Para):
    '''
    Transforms SS z to CM z
    '''
    N = len(Para.theta)
    S = len(Para.P)
    c1 = z[0:S]
    
    ci = z[S:N*S].reshape((N-1,S))
    
    l1 = z[N*S:(N+1)*S]
    
    li = z[(N+1)*S:2*N*S].reshape((N-1,S))
    
    zi = 2*N*S
    x_i = z[zi:zi+N-1].reshape((N-1,1))
    zi+= N-1
    
    rho_i = z[zi:zi+N-1].reshape((N-1,1))
    zi += N-1
    
    mu_i = z[zi:zi+N-1].reshape((N-1,1))
    zi += N-1
    
    lambda_i = z[zi:zi+N-1].reshape((N-1,1))
    zi += N-1
    
    mult = z[zi:]
    
    return hstack((c1,ci.flatten(),l1,li.flatten(),mu_i.flatten(),mult)),x_i,rho_i
    
def VCM(y,CMzbar):
    N = len(Para.theta)
    x = y[0:N-1].reshape((-1,1))
    rho = y[N-1:].reshape((-1,1))
    CMz = root(lambda z: CMResiduals(z,x,rho,Para),CMzbar).x
    c1,ci,l1,li,mu_i,xi,phi_i,eta_i = getCMQuantities(CMz,Para)
    P = Para.P[0,:]
    alpha_1 = Para.alpha[0]
    alpha_i = Para.alpha[1:]
    U = alpha_1*Para.U(c1,l1)+alpha_i.dot(Para.U(ci,li))
    return P.dot(U)/(1-Para.beta)

    