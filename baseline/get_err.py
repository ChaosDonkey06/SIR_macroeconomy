import numpy as np

def get_err(guess,A,theta,i_ini,pop_ini,pis1,pis2,pis3,pir,pid,betta,Uiss,HH,crss,nrss,Urss,muc,phii,deltav,deltac,kappa):

    #back out guesses for ns,ni,nr
    ns = guess[ 0:HH ]
    ni = guess[ HH+1:2*HH ]
    nr = guess[ 2*HH+1:3*HH ] 


    # equilibrium equations

    # Recovered people
    lambr    = (theta*nr)/A
    cr       = ((1+muc)*lambr)^(-1)
    ur       = np.log(cr)-theta/2*nr**2
    Ur       = (-1)*np.ones(HH+1,1)
    Ur[HH+1] = Urss
    
    for tt in np.flip( np.linspace(0,HH,HH+1) ):    
        Ur[tt,1] = ur(tt)+betta*Ur(tt+1,1)

    Gamma=(1+muc)*cr-A*nr

    #Infected People
    lambi=(theta*ni)/(phii*A)
    ci=((1+muc)*lambi)**(-1)
    ui=log(ci)-theta/2*ni**2

    #Susceptible People
    cs = 1/(1+muc)*(A*ns+Gamma)
    us = log(cs)-theta/2*ns**2

    #pre-allocate
    I   = -1*np.ones((HH+1,1))
    S   = -1*np.ones((HH+1,1))
    D   = -1*np.ones((HH+1,1))
    R   = -1*np.ones((HH+1,1))
    Pop = -1*np.ones((HH+1,1))
    T   = -1*np.ones((HH,1))

    #initial conditions
    Pop[0] = pop_ini
    I[0]   = i_ini
    S[0]   = Pop[0]-I[0]
    D[0]   = 0
    R[0]   = 0

    # Endogenous death probability
    pid_endo=-1*np.ones((HH,1))
    # iterate on SIR equations
    for j in np.arange(HH+1):

        T[j,1]        = pis1*S[j]*cs[j]*I[j]*ci[j]+pis2*S[j]*ns[j]*I[j]*ni[j] + pis3*S[j]*I[j]
        pid_endo[j,1] = pid+kappa*I[j]**2
        S[j+1,1]      = S[j] - T[j]
        I[j+1,1]      = I[j] + T[j]-(pir+pid_endo[j,1] )*I[j]
        R[j+1,1]      = R[j] + pir*I[j]
        D[j+1,1]      = D[j] + pid_endo[j,1]*I[j]
        Pop[j+1,1]    = Pop[j,1] - pid_endo[j,1]*I[j]
    

    #Infected People (continued)
    Ui        = -1*np.ones((HH+1,1)) 
    Ui[HH+1]  = (1-deltac)**HH*Uiss+(1-(1-deltac)**HH)*Urss #terminal condition 
    #Ui[HH+1] = Uiss
    #Ui[HH+1] = Urss
    for tt in np.flip( np.arange(HH) ):
        Ui[tt,1] = ui[tt]+(1-deltac)*betta*((1-pir-pid_endo[tt])*Ui[tt+1,1]+pir*Ur[tt+1,1] )+deltac*betta*Ur[tt+1,1] 

    #Susceptible People (continued)
    Us       = -1*np.ones((HH+1,1))
    Usss     = Urss #PV utility of susceptibles same as recovered in steady state
    Us[HH+1] = (1-deltav)**HH*Usss+(1-(1-deltav)**HH)*Urss #terminal condition
    
    for tt in np.flip( np.arange(HH) ):
        Us[tt,1] = us[tt]+(1-deltav)*betta*(1-T[tt]/S[tt])*Us[tt+1,1]+(1-deltav)*betta*T[tt]/S[tt]*Ui[tt+1,1]+deltav*betta*Ur[tt+1,1]

    #Lagrange multipliers susceptibles
    lamtau = (1-deltav)*betta*(Ui[0:HH+1]-Us[0:HH+1])
    lambs  = (cs^(-1)+lamtau*pis1*I[0:HH]*ci)/(1+muc)

    #equation residuals
    err              = -1*np.ones((3*HH,1))
    err[0:HH+1]      = (1+muc)*ci - phii*A*ni-Gamma
    err[HH+1:2*HH]   = muc*(S[0:HH+1]*cs + I[1:HH]*ci + R[1:HH]*cr)-Gamma*(S[0:HH+1] + R[0:HH+1]+I[0:HH+1])
    err[2*HH+1:3*HH] = -theta*ns + A*lambs + lamtau*pis2*I[0:HH]*ni

    # Aggregate consumption and hours
    aggC = S[0:HH]*cs + I[1:HH]*ci + R[0:HH+1]*cr
    aggH = S[0:HH]*ns + I[1:HH]*ni*phii + R[0:HH+1]*nr

    # Present value of society utility 
    U = S[0:HH+1]*Us[0:HH+1] + I[0:HH+1]*Ui[0:HH+1] + R[0:HH+1]*Ur[0:HH+1]

    RnotSIRmacro=T[0]/I[0]/(pir+pid)

    return [err,I,S,R,D,T,Pop,cs,ns,Us,RnotSIRmacro,aggC,aggH,ci,cr,ni,nr,Ui,Ur,U,pid_endo]
