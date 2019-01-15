# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:21:01 2018

@author: Chris
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import itertools as it
import matplotlib.pyplot as plt

def four_junction_computation_sparse(nmax):
    """ 
    This function will create an object which stores the base matrices which
    make up the flux qubit Hamiltonian. The object can then be called with
    arbitrary qubit parameters to generate the full sparse Hamiltonian
    
    Example:
        place_holder = four_junction_computation_sparse(N)
        Ham = place_holder(args)
    
    Ham can then be diagonalized, transformed, etc...
    """
    N = 2*nmax+1
    
    fm1_s = np.ones(N**3-N**2)
    fm1 = sparse.csc_matrix((fm1_s,
                               (range(N**3-N**2),range(N**2,N**3))),
                               shape=(N**3, N**3),
                               dtype=int)
    fm1 = fm1 + fm1.transpose()
    
    temp = np.ones((1,N-1),dtype=int)
    temp = np.append(temp,0)
    fm3_s = np.kron(np.ones((1,N)),
                    np.kron(temp,np.ones((1,N))))
    fm3 = sparse.csc_matrix((fm3_s[0,:N**3-N],
                               (range(N**3-N),range(N,N**3))),
                               shape=(N**3,N**3),
                               dtype=int)
    fm3 = fm3 + fm3.transpose()
    # Create the Off-diagonal entries sparsely
    fm4_s = np.kron(np.ones((1,N)),
                    np.kron(np.ones((1,N)),
                            temp))
    fm4 = sparse.csc_matrix((fm4_s[0,:N**3-1],
                               (range(N**3-1),range(1,N**3))),
                               shape=(N**3,N**3),
                               dtype=int)
    fm4 = fm4+fm4.transpose()
    # Create the block diagonals sparsely
    fm2m_s = np.kron(np.kron(temp,temp),temp)
    fm2m = sparse.csc_matrix((fm2m_s[:N**3-N**2-N-1],
                                 (range(N**3-N**2-N-1),range(N**2+N+1,N**3))),
                                 shape=(N**3,N**3),
                                 dtype=int)
    fm2p = fm2m.transpose()

    C = np.ones((1,N))
    n_vec=np.arange(-nmax,nmax+1,dtype=int)
    diag1=np.kron(np.kron(n_vec,C),C)
    diag3=np.kron(np.kron(C,n_vec),C)
    diag4=np.kron(C,np.kron(C,n_vec))
    
    sparse_dict = {'fm1':fm1,
                   'fm2p':fm2p,
                   'fm2m':fm2m,
                   'fm3':fm3,
                   'fm4':fm4,
                   'diag1':diag1,
                   'diag3':diag3,
                   'diag4':diag4}
    
    def INT_FUNCTION(Ej_v,Ec_v,r1_v,r2_v,r3_v,r4_v,f_v,ng12_v,ng23_v,ng34_v):
        d = {'Ej':Ej_v,
             'Ec':Ec_v,
             'r1':r1_v,
             'r2':r2_v,
             'r3':r3_v,
             'r4':r4_v,
             'f':f_v,
             'ng12':ng12_v,
             'ng23':ng23_v,
             'ng34':ng34_v}
        for key in d:
            try:
                inputlist = list(d[key])
            except TypeError:
                d[key] = list((d[key],))
        #don't do nested for loops like a chump
        var_iter = it.product(*d.values())
        
#        energy_container = []
        for var_list in var_iter:
            H = compiled_ham(sparse_dict,var_list,N)
#            H = H.todense()
#            if not np.array_equal(H.H,H):
#                print('The Hamiltonian is not Hermitian')
#                break
#            else:
#                eigs = np.linalg.eigvalsh(H)
#                energies = eigs[:nelevels]
#                energy_container.append(energies)
#        return np.array(energy_container)
        return H
    return INT_FUNCTION

def matrix_comparator(a,b):
    return np.equal(a,b).astype(int)

def compiled_ham(dict_sp,var_list,N):
    # Unpack the variables for ease of readability
    Ej = var_list[0]
    Ec = var_list[1]
    r1 = var_list[2]
    r2 = var_list[3]
    r3 = var_list[4]
    r4 = var_list[5]
    f = var_list[6]
    ng12 = var_list[7]
    ng23 = var_list[8]
    ng34 = var_list[9]
      
    H1 = (-Ej/2)*(r1*dict_sp['fm1']+
                  r3*dict_sp['fm3']+
                  r4*dict_sp['fm4'])
    H2 = (-Ej*r2/2)*(np.exp(-1j*2*np.pi*f)*dict_sp['fm2m']+
                     np.exp(1j*2*np.pi*f)*dict_sp['fm2p'])
    nq1 = ng12
    nq3 = -ng23
    nq4 = -ng23-ng34
    
    dn1 = dict_sp['diag1']-nq1
    dn3 = dict_sp['diag3']-nq3
    dn4 = dict_sp['diag4']-nq4
    
    # Per junction Ec
    dg = 4*Ec/(r1*r2*r3 + r1*r2*r4 + r1*r3*r4 + r2*r3*r4)
    dg = dg*((r2*r3 + r2*r4 + r3*r4)*dn1**2+
             (r1*r2 + r1*r4 + r2*r4)*dn3**2+
             (r1*r2 + r1*r3 + r2*r3)*dn4**2+
             (-2.0*r2*r4)*dn1*dn3+
             (-2.0*r2*r3)*dn1*dn4+
             (-2.0*r1*r2)*dn3*dn4)
    H3 = sparse.csc_matrix((dg[0],(range(N**3),range(N**3))),shape=(N**3,N**3))
    H = H1+H2+H3

    return H

def transition_matrix(nmax):
    N = 2*nmax+1
    temp = np.ones((1,N-1),dtype=int)
    temp = np.append(temp,0)
    
    fm2m_s = np.kron(np.kron(temp,temp),temp)
    fm2m = sparse.csc_matrix((fm2m_s[:N**3-N**2-N-1],
                                 (range(N**3-N**2-N-1),range(N**2+N+1,N**3))),
                                 shape=(N**3,N**3),
                                 dtype=int)
    fm2p = fm2m.transpose()
    def INT_FUNCTION(f):
    
        T = -(1.0j/2.0)*(np.exp(-1j*2*np.pi*f)*fm2m -
                    np.exp(1j*2*np.pi*f)*fm2p)
        T.todense()
        return T
    return INT_FUNCTION
    
def tidyup(a,tol=1e-12):
    small_flags = np.abs(a) < tol
    a[small_flags] = 0
    return a

def wrapper(func,*args,**kwargs):
    def wrapped():
        return func(*args,**kwargs)
    return wrapped


if __name__=="__main__":
    #testing = [2**n for n in range(1,8)]
    #testing.reverse()
    #t = []
    #for val in testing:
    #    wrapped = wrapper(four_junction_computation_sparse,val)
    #    t.append( timeit.timeit(wrapped,number=1) )
    #    print('done '+ str(np.log2(val)))
    #plt.plot(np.log2(testing),t)
    #plt.show()
    
    #f = np.linspace(0.47,0.53,101)
    #for i,val in enumerate(f):
    #    test_container = test(350,5,alpha,beta,1,1,val,0,0,0,5)
    #    x = test_container[0]
    #    x1.append(x[0])
    #    x2.append(x[1])
    #    x3.append(x[2])
    #    if i%10==0:
    #        print('We are %d percent completed'%(i/(len(f)-1)*100))
    #plt.plot(f,x1)
    #plt.plot(f,x2)
    #plt.plot(f,x3)
    #plt.show()
    #print(x2[50]-x1[50])
    #test_container = test(350,5,alpha,beta,1,1,0.5,0,0,0,5)
    #beta = np.linspace(0.5,50,101)
    #for i, val in enumerate(beta):
    #    start = time.time()
    #    test_container = test(350,20,alpha,val,1,1,0.5,0,0,0,2)
    #    x = test_container[0]
    #    x1.append(x[0])
    #    x2.append(x[1])
    #    if (i)%int((len(beta)-1)*0.1)==0:
    #        print('We are %d percent completed'%((i)/(len(beta)-1)*100))
    #    stop = time.time()
    #    print(stop-start)
    #plt.plot(beta,np.array(x2)-np.array(x1))
    #plt.show() 
       
    #test_container = test(350,20,alpha,beta,1,1,0.5,0,0,0,5)
    #x = test_container[0]
    #print(x[1]-x[0])
    
    #beta = np.linspace(1,101,11)
    #Delta = []
    #test = four_junction_computation_sparse(9)
    #for val in beta:  
    #    test_container = test(350,20,0.7,val,1.0,1.0,0.5,0,0,0)
    #    eigs = np.linalg.eigvalsh(test_container)
    #    delta = eigs[1]-eigs[0]   
    #    Delta.append(delta)
    #plt.plot(beta,Delta)
    #beta = np.linspace(1,51,51)
    ##beta = np.linspace(1,1.6,7)
    f = [0.5]
#    f = np.linspace(0.49,0.51,201)
    t01 = []
    t12 = []
    t02 = []
    nelevels = 3
    test = four_junction_computation_sparse(10)
    T_matrix = transition_matrix(10)
    #SUPERfast sweep using sparse linear algebra
    for val in f: 
        test_container = test(350,5,0.7,1,1,1000,val,0,0,0)
        transition = T_matrix(val)
    #    transition = transition.todense()
    #    transition = np.asarray(transition)
        eigs,evecs = linalg.eigsh(test_container,
                                  k=nelevels,
                                  which='SA',
                                  return_eigenvectors=True)
        temp = []
        for i in range(nelevels):
            temp.append(evecs[:,i])
        #Sort both eigenvalues and eigenvectors on an ascending list of
        #eigenvalues
        zipped = zip(eigs,temp)
        #Unpack the zipped list and convert into numpy arrays
        eigs, evecs = zip(*sorted(zipped, key=lambda x: x[0]))
        print(type(evecs))
        #Sorted list of eigenvalues
        eigs = np.array(list(eigs))
#        evecs = np.array(list(evecs))
        
        #Sorted 2D array of normalized eigenvectors the eigenvector associated to
        #eigenvalue eigs[i] is given by evec[:,0]
#        print(evecs.shape)
        evecs = np.transpose(np.array(evecs))
        print(evecs.shape)
        evecs = sparse.csc_matrix(evecs)
        print(evecs.shape)
        
        t01_temp = (evecs[:,1].H).dot(transition.dot(evecs[:,0]))
        t12_temp = (evecs[:,2].H).dot(transition.dot(evecs[:,1]))
        t02_temp = (evecs[:,2].H).dot(transition.dot(evecs[:,0]))
        
        t01.append(np.abs(t01_temp.data[0]))
        t12.append(np.abs(t12_temp.data[0]))
        t02.append(np.abs(t02_temp.data[0]))
    
    #    t01.append(np.abs(np.vdot(evecs[:,1],np.dot(transition,evecs[:,0]))))
    #    t12.append(np.abs(np.vdot(evecs[:,2],np.dot(transition,evecs[:,1]))))
    #    t02.append(np.abs(np.vdot(evecs[:,2],np.dot(transition,evecs[:,0]))))
        
#    plt.plot(f,t01)
#    plt.plot(f,t12)
#    plt.plot(f,t02)
#    plt.legend(['t01','t12','t02'])
#    plt.show()
