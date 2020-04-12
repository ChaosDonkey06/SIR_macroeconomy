import numpy as np

def calibrate_pis(pis_guess,HH,i_ini,pop_ini,pir,pid,pis1_shr_target,pis2_shr_target,RplusD_target,phii,C,N,scale1,scale2):


    #back out initial guesses
    pis1 = pis_guess[1]/scale1
    pis2 = pis_guess[2]/scale2
    pis3 = pis_guess[3]

    #pre-allocate
    I = -1*np.ones((HH+1,1))
    S = -1*np.ones((HH+1,1))
    D = -1*np.ones((HH+1,1))
    R = -1*np.ones((HH+1,1))
    T = -1*np.ones((HH,1))

    #initial conditions
    I[0] = i_ini
    S[0] = pop_ini-I[0]
    D[0] = 0
    R[0] = 0

    #iterate on SIR model equations
    for j in range(HH+1):
        T[j,0]   = pis1*S[j]*C^2*I[j]+pis2*S[j]*N^2*I[j]+pis3*S[j]*I[j]
        S[j+1,0] = S[j]-T[j]
        I[j+1,0] = I[j]+T[j]-(pir+pid)*I[j]
        R[j+1,0] = R[j]+pir*I[j]
        D[j+1,0] = D[j]+pid*I[j]
    
    err = np.zeros((3))
    #calculate equation residuals for target equations
    err[0] = pis1_shr_target-(pis1*C^2)/(pis1*C^2+pis2*N^2+pis3)
    err[1] = pis2_shr_target-(pis2*N^2)/(pis1*C^2+pis2*N^2+pis3)
    err[2] = RplusD_target-(R[-1]+D[-1])


    RnotSIR = T[0]/I[0]/(pir+pid)


    return [err,pis1,pis2,pis3,RnotSIR,I,S,D,R,T]