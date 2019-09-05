# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

#Domain (periodic)
L=1000.*10**3 #m
dx=10.*10**3 #m
nr=int(L/dx)
grid=np.linspace(0,L-dx,num=nr)

#Advection Velocity and Chemical Decay
u0=10. #m/s
k0=10**(-4)

#Time settings
dt=1*60. #s
tmax=1*24*60*60. #s


#Compute the Concentration Profile using the Euler method
def Euler(C0,E,dt,tmax):
    stepmax=int(tmax/dt+1)
    C=np.zeros((stepmax,nr))
    for j in range(0,nr):
        C[0][j]=C0[j]
    for t in range(1,stepmax):
        for j in range(0,nr):
            C[t][j] = E[j]*dt - k0*C[t-1][j]*dt + C[t-1][j] - dt*u0*((C[t-1][j]-C[t-1][j-1])/dx)
    return C


#Compute forward matrix and its transpose as a function of the initial concentration and time
def ForwardK(C0,dt,tmax,S0):
    stepmax=int(tmax/dt+1)
    stations=len(S0)
    K=np.zeros([nr,stepmax*stations])
    for i in range(0,nr):
        E=np.zeros(nr)
        E[i]=1.
        Cres=Euler(C0,E,dt,tmax)
        y0=np.zeros([stations,stepmax])
        for j in range(0,stations):
            y0[j]=Cres[:,S0[j]]
        K[i]=np.array(y0).flatten()
    KT=K
    K=np.transpose(K)
    return [K,KT]

# Invert matrix K
#This computes a matrix such that Kinv*K=Identity(100x100)
#Also see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html
def InvertK(K):
    Kinv=la.pinv(np.matrix(K))
    return Kinv

#Compute Gain Matrix
def GainM(K,KT,Sa,Se):
    leny=len(Se[0])
    M1=np.dot(np.dot(K,Sa),KT)
    M2=np.zeros((leny,leny))
    for i in range(0,leny):
        for j in range(0,leny):
            M2[i][j]=M1[i][j]+Se[i][j]
    M3=la.inv(M2)
    M4=np.dot(Sa,KT)
    G=np.dot(M4,M3)
    return G

#Gaussian Distribution
def Gauss(x,mu,Sigma):
    Ex=1/np.sqrt(2*Sigma**2*np.pi)*np.exp(-(x-mu)**2/(2*Sigma**2))
    return Ex

#Compute A Priori Emissions
def APrioriE(Ea0,Eastrength,Sigma):
    Ea=np.zeros(nr)
    if Ea0==0.:
        Ea=np.ones(nr)*Eastrength
    elif Sigma==0.:
        emissionsa=len(Ea0)
        for i in range(0,emissionsa):
            Ea[int(Ea0[i]/dx)]=Eastrength
    else:
        emissionsa=len(Ea0)
        for i in range(0,emissionsa):
            for j in range(0,nr):
                Ea[j]=Ea[j]+Gauss(j*dx,Ea0[i],Sigma[i])*dx
        Ea=Ea*Eastrength
    return Ea

def Post(K,KT,Sa,Se):
    leny=len(Sa[0])
    SaI=la.inv(Sa)
    SeI=la.inv(Se)
    N1=np.dot(np.dot(KT,SeI),K)
    N2=np.zeros((leny,leny))
    for i in range(0,leny):
        for j in range(0,leny):
            N2[i][j]=N1[i][j]+SaI[i][j]
    Shat=la.inv(N2)
    return Shat



# Scheme determines whether Euler (0) or Runge-Kutta (1) is used
# Input Vectors for Emissions and Stations in Real Distance (m)
# For a Uniform A Priori Profile set Ea0 = 0
# Sigma gives the Spatial Spread, while Erra the Uncertainty in the Strength of Emissions
# Cov gives the Correlation Distance, setting it Zero means no Covariance

def TestModel(C0,E0,Estrength,S0I,Munc,Ea0,Eastrength,Sigma,Erra,Cov,Errm):
    
    stepmax = int(tmax/dt+1)  
    stations = len(S0I)
    S0 = np.zeros(stations)
    for i in range(0,stations):
        S0[i] = int(S0I[i]/dx)    
    
    #Create Emission Vector
    emissions = len(E0)
    E = np.zeros(nr)
    for i in range(0,emissions):
        E[int(E0[i]/dx)] = Estrength[i]
    
    #Create Matrices for the Forward Model
    [K,KT] = ForwardK(C0,dt,tmax,S0)  
    
    #Create Measurement Vector
    y0 = np.dot(K,E)
    errorstrength = Munc
    errorC0 = np.zeros([stations,stepmax])
    for i in range(0,stations):
        for j in range(0,stepmax):
            errorC0[i][j] = np.random.normal(0,errorstrength)
    error0 = np.array(errorC0).flatten()    
    ym = y0 + error0
    
    #Create A Priori Emission Vector and Corresponding Error Matrix
    Ea = APrioriE(Ea0,Eastrength,Sigma)
    ErrorEa = Ea*Erra
    Sa0 = np.diag(ErrorEa**2)
    if Cov == 0.:
        Sa = Sa0
    else:
        SaN = np.zeros((nr,nr))
        CovN = Cov/dx
        for j in range(0,nr):
            for i in range(0,nr):
                if i>j:
                    SaN[i][j] = ErrorEa[i]*ErrorEa[j]*np.exp(-(i-j)**2/CovN**2)
        Sa = SaN + Sa0 + np.transpose(SaN)
    
    #Create Measurement Error Matrix
    ErrorM = np.ones(len(y0))*Errm
    Se = np.diag(ErrorM)
    
    #Compute the Gain Matrix and Corresponding Solution
    G = GainM(K,KT,Sa,Se)
    SolErr = np.array(Ea + np.dot(G,ym-np.dot(K,Ea)))
    surface1=np.zeros(nr)
    surface2=np.zeros(nr)
    surface3=np.zeros(nr)
    for k in range(0,nr):
        if k>25and k<35:
            if SolErr[k]<10**(-4)or SolErr[k]<0:
                SolErr[k]=0.
            surface1[k]=((SolErr[k]+SolErr[k-1])/2.)*dx
        elif k>40 and k<60:
            if SolErr[k]<10**(-4) or SolErr[k]<0:
                SolErr[k]=0.
            surface2[k]=((SolErr[k]+SolErr[k-1])/2.)*dx
        elif k>80 and k<100:
            if SolErr[k]<10**(-4)or SolErr[k]<0:
                SolErr[k]=0.0
            surface3[k]=((SolErr[k]+SolErr[k-1])/2.)*dx
    surface11=np.sum(surface1)/1000.
    surface21=np.sum(surface2)/1000.
    surface31=np.sum(surface3)/1000.
    print surface11,surface21,surface31    
    #Compute Posterior Error Covariance Matrix and the difference with the A Priori
    Shat = Post(K,KT,Sa,Se)
    Reduction = np.zeros((nr,nr))
    for i in range(0,nr):
        for j in range(0,nr):
            Reduction[i][j] = Sa[i][j] - Shat[i][j]
    
    #Plot the Concentration Data
    Data = plt.figure()
    plt.plot(ym)
    plt.plot(y0)
    
    #Plot the Estimated Emissions
    Prof = plt.figure()
    plt.plot(grid/1000.,E,'r',lw=2.0,label='Real emission point')
    plt.plot(grid/1000.,SolErr,'b',lw=2.0,label='Model of emission')
    plt.plot(np.array(S0I)/1000.,np.zeros(stations),'go',ms=8.0)
    if Ea0 != 0:
        plt.plot(grid/1000.,Ea,'k',lw=2.0,label='A priori emission')
    plt.xlabel('x (km)')
    plt.ylabel('Emission Strength')
    plt.legend(loc=2, fontsize = 'x-small')
    
    #Plot the Difference of the A Priori and Posterior Error Matrix
    PostErr = plt.figure()
    X, Y = np.meshgrid(grid/1000., grid/1000.)
    Cont = plt.contourf(X, Y, Reduction)
    plt.colorbar(Cont)
    
    PostErrDiag = plt.figure()
    plt.plot(np.diag(Sa))
    plt.plot(np.diag(Shat))

    return Data, Prof, PostErr, PostErrDiag



#Choose Stations
S1=100.*10**3 #m
S2=400.*10**3 #m
S3=800.*10**3 #n
S0I=[S1,S2,S3]

#Emissions with Strength
E1=300.*10**3 #m
E2=500.*10**3 #m
E3=900.*10**3 #m
Estrength1=1.*10**(-1) #ton/s
Estrength2=1.*10**(-1) #ton/s
Estrength3=1.*10**(-1) #ton/s
E0=[E1,E2,E3]
Estrength =[Estrength1,Estrength2,Estrength3]

#Set Initial Concentration
C0I=np.zeros(nr)
emissions = len(E0)
#E = np.zeros(nr)
#for i in range(0,emissions):
#    E[int(E0[i]/dx)] = Estrength[i]
t0=2*60*60. #s
C0 = np.ones(nr) #Euler(C0I,E,dt,t0)[-1]

#Measurement Error
Munc=1. #ton
Errm=Munc

#A Priori Estimate
Ea1=330.*10**3 #m
Ea2=490.*10**3 #m
Ea3=910*10**3 #m
Eastrength=1.*10**(-2) #ton/s

Sigma1=60.*10**3 #m
Sigma2=20.*10**3 #m
Sigma3=20.*10**3 #m
Cov=0. #m (Covariance Measure)

Ea0=[Ea1,Ea2,Ea3]
Sigma=[Sigma1,Sigma2,Sigma3]
Erra=0.6 #fraction
Data, Prof, PostErr, PostErrDiag = TestModel(C0,E0,Estrength,S0I,Munc,Ea0,Eastrength,Sigma,Erra,Cov,Errm)

# Scheme _ Emissions _ Stations, Error _ A Priori, Strength, Error (%) _ MEasurement Error
#Data.savefig('Concentrations_Euler_3_220_31025_20.pdf')
#Prof.savefig('Emissions_Euler_3_220_31025_20.pdf')

plt.show(Data)

