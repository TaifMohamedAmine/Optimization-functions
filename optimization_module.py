import numpy as np
from math import * 
from numdifftools import Gradient, Hessian
import scipy.optimize as spo
from numpy import linalg as alg
from scipy.misc import derivative as deriv




# 1 - One dimentional optimization methods: 

# 1.1- fixed step size search: 


def fixed_step(f, A, S):
    
    # f: the real function to minimize
    # A: intial starting point
    # S: the fixed step size given as argument
    
    if f(A) > f(A + S):
        B = A + S
        while( f(B - S) > f(B) ):
            B += S
    else:
        B = A + S
        while( f(B) > f (B - S) ):
            B += S
    return B


# 1.2 - accelerated step size search:

def accelerated_left(f , Xs , S ): # S : initial step size given
    C = Xs + S
    while( f(C-S) > f(C)):
        S *= 2
        C += S
    return C

def accelerated_right(f , Xs , S ): # S : initial step size given
    C = Xs - S
    while( f(C-S) < f(C)):
        S *= 2
        C -= S
    return C

def accelerated_step(f, Xs, S, p): 
    
    # Xs : starting point
    # S : initial step size given
    # p : precision/tolerance
    
    C = Xs - p - 1000
    while (abs(Xs - C) > p ):
        temp = accelerated_left(f,Xs,S)
        C = Xs
        Xs = temp
        temp = accelerated_right(f,Xs,S)
        C = Xs
        Xs = temp
    return C

# 1.3 - Halving method :

def halving(f, Xs , Xf , p ):
    
    #Xs, Xf : start and finish of the search interval
    #p : precision of the search
    
    M = (Xf - Xs)/2
    X0 = Xs + M
    X1 = Xs + M/2
    X2 = Xf - M/2
    while (abs(X1 - X2) > p) :
        if f(X1) < f(X0) and f(X0) < f(X2) :
            Xf = X0
        if f(X1) > f(X0) and f(X0) > f(X2) :
            Xs = X0
        if f(X1) > f(X0) and f(X0) < f(X2) :
            Xs = X1
            Xf = X2
        M = (Xf - Xs)/2
        X0 = Xs + M
        X1 = Xs + M/2
        X2 = Xf - M/2
    return (X1+ X2)/2

# 1.4 - exhaustive search method :

def exhaustive_left(f,Xs, Xf, n):
    S = (Xf - Xs)/n
    B = Xs + S
    while ( f(B - S) > f(B)):
        B += S
    return  B - S

def exhaustive_right(f,Xs, Xf, n):
    S = (Xf - Xs)/n
    B = Xf - S
    while ( f(B + S) > f(B)):
        B -= S
    return B + S


def exhaustive(f , Xs , Xf, n ):
    
    #Xs, Xf : start and finish of the search interval
    #n : number of samples in the search interval
    
    I = exhaustive_left(f, Xs, Xf, n)
    F = exhaustive_right(f, Xs, Xf, n)
    return (I + F)/2

# 1.5 - dichotomous search

def dichotomous(f, Xs, Xf, p):
    
    #Xs, Xf : start and finish of the search interval
    #p : precision of the search
    
    M = (Xf - Xs )/2

    A = Xs + M - (p/2)
    B = Xf - M + (p/2)
    while M > p :
        if f(A) < f(B):
            Xf = B           
        else : 
            Xs = A 
        M = (Xf - Xs) / 2
        A = Xs + M - (p/2)
        B = Xf - M + (p/2)
    return (A + B)/2

# 1.6 - Fibonacci search method :

def fibonacci_term(n): # function to calculate the n term of the fibonacci sequence
    if n <= 1 :
        return 1 ;
    else :
        a = 1
        b = 1
        for i in range(1, n+1):
            c = a + b
            a = c
            b = a
    return c

def fibonacci_search(f, Xs, Xf, p, n):
    
    #Xs, Xf : start and finish of the search interval
    #p : precision of the search
    #n : the number of fibonacci terms
    
    while ( abs(Xf - Xs) > p  ) :
        X1 = Xs + (fibonacci_term(n-2)/fibonacci_term(n))*(Xf - Xs)
        X2 = Xs - (fibonacci_term(n-1)/fibonacci_term(n))*(Xf - Xs)
        if f(X2) > f(X1) :
            Xs = X1
        else :
            Xf = X2
        n -= 1 
    return (Xs+ Xf)/2


# 1.7 - the golden ratio method :

golden = (1 + 5 ** 0.5) / 2   # an approximation of the golden ratio constante

def golden_section(f, Xs, Xf, p):
    
    #Xs, Xf : start and finish of the search interval
    #p : precision of the search
    
    golden_power =(1 / (golden * golden))
    
    X1 = Xs + (Xf - Xs) * (golden_power)
    
    X2 = Xf - (Xf - Xs) * (golden_power)
    
    while( golden_power > p ):
        
        if f(X1) < f(X2):
            Xf = X2 
            X2 = X1
            X1 = Xs + (Xf - Xs) * (golden_power)
        else :
            Xs = X1
            X1 = X2
            X2 = Xf - (Xf - Xs) * (golden_power)
        golden_power *= (1 / golden)
    
    return (Xs + Xf)/2

# 1.8 - Newton rapson method :

def Newton_Rapson(fp , fpp, Xs, p ):
    
    # fp : first derivative of the function
    #fpp : second derivative of the functio
    #p : search precision
    
    B = Xs
    while abs(fp(B)) > p:
        B -= fp(B)/fpp(B)
    return B

# 1.9 - Quasi Newton method :

def quasi_Newton_1D(f, Xf, p, epsilon):
    
    #Xf : the initial search point
    #epsilon : search precision
    #p : derivative step size
    
    B = Xf
    while abs((f(B + p) - f(B - p))/(2*p)) > epsilon : 
        B -= p*(f(B + p) - f(B - p))/(2*(f(B + p) - 2*f(B) + f(B - p)))
    return B

# 1.10 - secant search method :

def secant_Method(df, A, t0):
    
    epsilon = 0.0000001 
    
    while abs(df(A)) > epsilon:
        
        if df(t0) < 0:
            A = t0
            t0 *= 2
        
        else:
            B = t0
            A -= df(A)/((df(B)-df(A))/(B-A))
    return A

# 2 - Solving a linear system :

# 2.1 - Choleski decomposition :

# on cherche à decomposer une matrice A symetrique definie postive 
# en produit d'une matrice triangulaire inferieure L et sa transposée A = L @ L.T

def choleski(A):
    test = np.all(np.linalg.eigvals(A)>0)
    if test == False:
        return None
    n = len(A)
    L = np.zeros_like(A)  
    for i in range(0,n):
        for j in range(i,n):
            if i == j :
                sum = 0
                for k in range(0,i):
                    sum += L[j,k] ** 2   
                L[j,i] = np.sqrt(A[j,i] - sum)
            else :
                sum = 0
                for k in range(0,i):
                    sum += L[j,k]*L[i,k] 
                L[j,i] = (A[j,i] - sum)/ L[i,i]
    return L    

# 2.2 - LU decomposition :

#on cherche à decomposer une matrice donnée A en produit de deux matrices triangulaires inferieure L
# et superieure U tel que : A = L@U

def LU(A):
    n = len(A)
    L = np.identity(n)
    U = A.copy()
    for i in range(0,n):
        for j in range(i+1, n):
            L[j,i] = U[j,i]/U[i,i]
            U[j] = U[j] - L[j,i]*U[i]
    return L, U

# 2.3 - Inverse of a matrix :

def echange_ligne(A, i, j):  # function to switch ligns of a matrix in order to find the right pivot
    temp = np.copy(A[i])
    A[i] = A[j]
    A[j] = temp
    return A

def Inverse_Matrice(A):
    n = len(A)
    I = np.identity(n)
    A_1 = np.float32(np.copy(A))
    I_1 = np.float32(np.copy(I))
    if np.linalg.det(A) == 0:
        print("la matrice n'est pas inversible")
        return None
    else:
        for i in range(n):
            if A_1[i,i] == 0:
                for k in range(i+1, n):
                    if A_1[i,k] != 0:
                        index = k
                        break
                echange_ligne(A_1,i,index)
                echange_ligne(I_1,i,index)
            for j in range(n):
                if j != i :
                    ratio = A_1[j,i] / A_1[i,i]
                    for k in range(n):
                        A_1[j,k] -= ratio * A_1[i,k]
                        I_1[j,k] -= ratio * I_1[i,k]
        for i in range(n):
            divisor = A_1[i,i]
            for j in range (n):
                A_1[i,j] = A_1[i,j] / divisor
                I_1[i,j] = I_1[i,j] / divisor
        
        return I_1, A_1 

# 2.4 - solve a system using Gauss method :

def echange_ligne(A, i, j): # function to switch ligns of a matrix in order to find the right pivot
    temp = np.copy(A[i])
    A[i] = A[j]
    A[j] = temp
    return A

def system_Gauss(A,b):     # solving an equation of the format : Ax = b
    A_1 = np.copy(A)
    n=len(A)
    for i in range(n):
        for j in range(i+1, n):
            index = i
            if A_1[i,i] == 0:
                for k in range(i+1, n):
                    if A_1[k,i] != 0:
                        index = k
                        break
            echange_ligne(A,i,index)
            echange_ligne(b, i, index)
            ratio = A_1[j,i] / A_1[i,i]
            A_1[j] -= A_1[i] * ratio
            b[j] -= b[i] * ratio  
    X = b
    for i in range(n-1,-1,-1):
        for k in range(n-1,i, -1):
            if i == n-1:
                continue
            else:
                X[i] -= A_1[i,k]*X[k]
        X[i] /= A_1[i,i]
    return X 

# 2.5 - solve a system using Choleski decomposition :

def system_Choleski(A,b):
    
    # solving an equation of the format : Ax = b
    
    A,B = choleski(A),choleski(A).T
    w = system_Gauss(A,b)
    X = system_Gauss(B,w)
    
    return X

# 2.6 - solve a system using LU decomposition :

def system_LU(A,b):
    
    # solving an equation of the format : Ax = b
    
    L,U = LU(A)
    w = system_Gauss(L,b)
    X = system_Gauss(U,w)

    return X

# 3 - multivdimentional optimization methods :

# 3.1 - Gradient descent :

def gradient_descent(f,x0, delta=0.01):
    
    #X0 : starting search point
    # delta : search precision
    
    X = x0
    nrm = np.linalg.norm
    grad = Gradient(f)
    while (nrm(grad(X)) > delta):
        phi = lambda alpha : f(X - alpha * grad(X))
        alpha_k = accelerated_step(phi, 0, 0.0001, 0.001)
        X = X - alpha_k*grad(X)   
    return X

# 3.2 - conjugate gradient :

def Gradient_conjugate(f, X0, Q ):
    n = len(X0)
    grd = Gradient(f)
    d = -grd(X0)
    X = X0
    for k in range(0, n):
        alpha_k = (np.transpose(d)@d)/(np.transpose(d)@Q@d)
        X = X + alpha_k*d
        beta_k = (np.transpose(grd(X))@Q@d)/(np.transpose(d)@Q@d)
        d = -grd(X) + beta_k*d
    return X

# 3.3 - Newton method :

def Newton_Method(f, X0, delta = 0.001):
    grd = Gradient(f)
    norm = alg.norm
    inv = alg.inv
    H = Hessian(f)
    n = len(X0)
    X = X0
    I = np.identity(n)
    d = -np.dot(inv(H(X)), grd(X))
    while (norm(d) > delta):
        phi = lambda alpha : f(X - alpha * d)
        alpha_k = accelerated_step(phi, 0, 0.0001, 0.001)
        X = X - alpha_k*d
        if (np.all(alg.eigvals(H(X))) > 0) :
            d = -np.dot(inv(H(X)), grd(X))
        else:
            d = -np.dot(inv(delta*I + H(X)), grd(X))
    return X

#3.4 - Quasi Newton method :

def armijo(f,eta = 2,epsilon = 0.01):
    
    alpha = np.random.uniform(0, 1, 1)[0]
    if f(alpha) <= f(0) + epsilon * alpha * deriv(f, 0, dx = 1e-6):    
        while f(alpha) <= f(0) + epsilon * alpha * deriv(f, 0, dx = 1e-6):
            alpha *= eta
    else:   
        while f(alpha) > f(0) + epsilon * alpha * deriv(f, 0, dx = 1e-6):
            alpha /= eta
    return alpha



def DFP(H, a_k,d, y):    #Davidon–Fletcher–Powell function in order to calculate the hessian
    A = (a_k*d@(np.transpose(d)))/(((np.transpose(d))@y))
    B = -((H@y) @ (np.transpose(H@y))) / (np.transpose(y) @ H @ y)
    H_nouveau = H + A + B
    return H_nouveau



def Quasi_newton(f, X0, delta = 0.01):
    n = len(X0)
    norm = alg.norm
    grd = Gradient(f)
    g = np.transpose(grd(X0))
    H = np.identity(n)
    X_0 = X0
    while (norm(g) > delta):
        d = - H @ g 
        phi = lambda alpha : f(X_0 - alpha*d)
        alpha_k = accelerated_step(phi, 0, 0.0001, 0.001)
        #alpha_k = armijo(phi)                   # armijo retourne alpha_k trop petit
        X_1 = X_0 - alpha_k * d
        y = grd(X_1) - grd(X_0)
        H = DFP(H, alpha_k,d, y)
        X_0 = X_1
        g = np.transpose(grd(X_1))
    return X_1

    



