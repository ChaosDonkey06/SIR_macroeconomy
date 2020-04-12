import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
###################################################### 
##Parameters, calibration targets and other settings##
######################################################

betta=0.96**(1/52)  #Weekly household discount factor
pid=7*0.005/18;     #Weekly probability of dying
pir=7*1/18-pid;     #Weekly probability of recovering
phii=0.8;           #Productivity of infected people
 
deltav=0/52;        #Weekly probability of discovering a vaccine
deltac=0/52;        #Weekly probability of discovering a treatment
kappa=0;            #Slope of pid-function in endog. pid scenario 
                    #(medical preparedness scenario)
 
#Calibration targets for hours and income
n_target=28;         #Weekly hours
inc_target=58000/52; #weekly income
 
#Calibation targets for shares of pis-terms in T-function in SIR model
pis3_shr_target=2/3;                   #share of T_0 jump due general infections
pis1_shr_target=(1-pis3_shr_target)/2; #share of T_0 jump due to consumption-based infections
pis2_shr_target=(1-pis3_shr_target)/2; #share of T_0 jump due to work-based infections
RplusD_target=0.60;                    #total share of people infected and then either recovered or dead after epidemic
 
pop_ini=1;          #Initial population
i_ini=0.001;        #Initial infected
 
HH=250;             #Number of periods to solve and simulate the model
 
#containment policy
muc = np.zeros((HH,1));    #exogenous path for muc over time.
                    #if you want e.g. a containment policy of
                    #10 percent for weeks 1,...52, then set muc(1:52)=0.1;
                    #
                    #Make sure that muc=0 at the end of the solution and
                    #simulation horizon. Steady state assumes muc=0;
                    #
                    #With optimal policy, muc path will be chosen to
                    #maximize PV utility (switch below).
                    
do_opt_policy=0;    #switch: if = 0, model is solved and simulated with
                    #given path for containment policy muc
                    #
                    #if =1, model is solved and simulated with optimal
                    #containment path muc. Path for muc set above will be
                    #overwritten with optimal path. COMPUTATIONS TAKE A
                    #WHILE IF YOU SET do_opt_taxes=1
                    
use_parallel=0;     #when optimal policy is computed, use_parallel=1 uses 
                    #parallel computing to maximize PV utility using fmincon.
 
#nonlinear solver and minimizer settings
opts_fsolve = optimoptions('fsolve','Display','iter','TolFun',1e-9); #options for fsolve
opts_fsolve_fmincon =optimoptions('fsolve','Display','off','TolFun',1e-9); #options for fsolve used opt. policy calcs. (fmincon)

if use_parallel==0:
    opts_fmincon=optimoptions('fmincon','Display','iter','TolFun',1e-7,'MaxFunctionEvaluations',2000,'FiniteDifferenceStepSize',1e-2) #options for fmincon w/o parallel comp.

elif use_parallel==1:
    #opts_fmincon=optimoptions('fmincon','Display','iter','TolFun',1e-7,'MaxFunctionEvaluations',2000,'UseParallel',true,'FiniteDifferenceStepSize',1e-2) #options for fmincon with parallel comp.
    #opts_fmincon=optimoptions('fmincon','Display','iter','TolFun',1e-6,'MaxFunctionEvaluations',10000,'UseParallel',true,'FiniteDifferenceStepSize',1e-2) #options for fmincon with parallel comp.
    #opts_fmincon=optimoptions('fmincon','Display','iter','TolFun',1e-6,'MaxFunctionEvaluations',5000,'UseParallel',true,'FiniteDifferenceStepSize',1e-3)
    opts_fmincon=optimoptions('fmincon','Display','iter','TolFun',1e-7,'MaxFunctionEvaluations',10000,'UseParallel',true)

############################
#Steady State Calculations##
############################

theta=1/n_target**2;     #calculate disutility of labor parameter theta
                        #so that pre-infection steady state labor is equal to
                        #desired target (using n=(1/theta)^(1/2) pre-infection 
                        #steady state equation)
A=inc_target/n_target;  #calculate parameter A such that income is equal to
                        #desired target, c=inc_target=A*n_target
 
#steady states
nrss            = (1/theta)**(1/2);           #labor recovered (same as post-infection steady state)
crss            = A*nrss;                    #consumption recovered
urss            = np.log(crss)-theta/2*nrss**2;  #utility recovered
Urss            = 1/(1-betta)*urss;          #PV utility recovered
UrssConsUnits   = Urss*crss;        #PV utility in cons. units (Urss*Marg.Util.Cons); value of life
niss            = (1/theta)**(1/2);           #labor infected
ciss            = phii*A*niss;               #consumption infected
uiss            = np.log(ciss)-theta/2*niss**2;  #utility infected
Uiss            = 1/(1-(1-deltac)*betta*(1-pir-pid))*(uiss +(1-deltac)*betta*pir*Urss+deltac*betta*Urss);  #PV utility infected
 
 
#Check level of present value utility
if Uiss-Urss>0: 
    print( 'Error: parameterization implies Uiss>Urss: {}'.format(Uiss-Urss) )
 
#calibrate the pis's in T-function
go_calibrate_pis
 
#initial guess for optimal muc and load last solution of optimal policy 
#allocations if you dont want to start maximization of PV utility from scratch.
if do_opt_policy==1:      
    muc_guess = np.zeros((HH,1))                  #initial guess for opt. cont. policy
    
    #load last_solution_opt_policy    #uncomment if you want to use last
                                      #solution as initial guess
    #muc_guess=muc_sol;
    #if numel(muc_guess)~=HH
    #    error('Initial guess for optimal policy loaded from disk has different dimension than HH.');
    #end
 
#initial guess of vectors of ns, ni and nr to solve nonlinear
#equilibrium equations
n_vec_guess=nrss*np.ones((3*HH,1)); #guess of vectors for ns,ni,nr
 
#If optimal policy is desired, find optimal path for muc to maximize PV utility
if do_opt_policy==1: #optimal policy
    
    #minimize negative PV utility (i.e. max PV utility) to find opt. path for muc; nonlinear
    #model equations are solved inside getU.m using function get_err.m
    LB=muc_guess*0-2; #upper bounds
    UB=muc_guess*0+2; #lower bounds

    muc_sol = minimize(getU,muc_guess,[],[],[],[],LB,UB,[],opts_fmincon,n_vec_guess,opts_fsolve_fmincon,A,theta,i_ini,pop_ini,pis1,pis2,pis3,pir,pid,betta,Uiss,HH,crss,nrss,Urss,phii,deltav,deltac,kappa)
    
    muc = np.zeros((HH,1))
    muc(0:len(muc_sol))=muc_sol
    #save last_solution_opt_policy muc_sol #save solution for possible use in subsequent maximization   

 
#Given either optimal path for muc or exogenous path for muc,
#solve nonlinear equilibrium model equations (i.e. adjust guesses ns,nr,ni)

[n_vec,fval,exitflag] = fsolve(@get_err,n_vec_guess,opts_fsolve,A,theta,i_ini,pop_ini,pis1,pis2,pis3,pir,pid,betta,Uiss,HH,crss,nrss,Urss,muc,phii,deltav,deltac,kappa);
if exitflag~=1:
    error('Fsolve could not solve the model')
    
#get allocations given either exogenous or optimal path for muc at ns,ni,nr
#solution

[err,I,S,R,D,T,Pop,cs,ns,Us,RnotSIRmacro,aggC,aggH,ci,cr,ni,nr,Ui,Ur,U,pid_endo] = get_err(n_vec,A,theta,i_ini,pop_ini,pis1,pis2,pis3,pir,pid,betta,Uiss,HH,crss,nrss,Urss,muc,phii,deltav,deltac,kappa)
disp(['Max. abs. error in equilib. equations:',num2str(max(abs(err)))]);
disp(' ')
RnotSIRmacro
 
#plotting
ia      = 2
ib      = 2
fsize   = 12
horz    = HH
time    = np.arange(horz-1)

import matplotlib.pyplot as plt
plt.figure
plt.subplot(ia,ib,1)
plt.plot(time,I(1:horz),'b-')
plt.title('Infected, I','FontSize',fsize)
plt.ylabel('Share of Initial Population','FontSize',fsize)
 
plt.subplot(ia,ib,2)
plt.plot(time,S(1:horz),'b-','LineWidth',2)
title('Susceptibles, S','FontSize',fsize)
 
plt.subplot(ia,ib,3)
plt.plot(time,R(1:horz),'b-','LineWidth',2)
title('Recovered, R','FontSize',fsize)
plt.ylabel('Share of Initial Population','FontSize',fsize)
plt.xlabel('Weeks','FontSize',fsize)
set(gca,'FontSize',fsize)
 
plt.subplot(ia,ib,4)
plt.plot(time,D(1:horz),'b-','LineWidth',2)
plt.title('Deaths, D','FontSize',fsize)
xlabel('Weeks','FontSize',fsize)

 
plt.figure
plt.subplot(ia,ib,3)
plt.plot(time,100*(cs(1:horz)-crss)/crss,'b-','LineWidth',2)
plt.plot(time,100*(ci(1:horz)-crss)/crss,'r--','LineWidth',2)
plt.plot(time,100*(cr(1:horz)-crss)/crss,'k-.','LineWidth',2)

# axis([0 horz-1 -(1-phi+0.02)*100 2])
plt.title('Consumption by Type','FontSize',fsize)
plt.ylabel('% Dev. from Initial Steady State','FontSize',fsize)
plt.xlabel('Weeks','FontSize',fsize)
#set(gca,'FontSize',fsize)
plt.legend(['Cons. Susceptibles','Cons. Infected','Cons. Recovered','Location','best'])
 
plt.subplot(ia,ib,4)
plt.plot(time,100*(ns(1:horz)-nrss)/nrss,'b-','LineWidth',2)
plt.plot(time,100*(ni(1:horz)-nrss)/nrss,'r--','LineWidth',2)
plt.plot(time,100*(nr(1:horz)-nrss)/nrss,'k-.','LineWidth',2)
#box off;
plt.title('Hours by Type','FontSize',fsize)
plt.xlabel('Weeks','FontSize',fsize)
#set(gca,'FontSize',fsize);
plt.legend('Hours Susceptibles','Hours Infected','Hours Recovered','Location','best');
#legend boxoff
 
plt.subplot(ia,ib,1)
plt.plot(time,0*100*(aggC(1:horz)-crss)/crss,'m:','LineWidth',1.5)
plt.plot(time,100*(aggC(1:horz)-crss)/crss,'b-','LineWidth',2)
#box off
plt.title('Aggregate Consumption, C','FontSize',fsize)
plt.ylabel('% Dev. from Initial Steady State','FontSize',fsize)
#set(gca,'FontSize',fsize);
 
plt.subplot(ia,ib,2)
plt.plot(time,0*100*(aggC(1:horz)-crss)/crss,'m:','LineWidth',1.5)
plt.plot(time,100*(aggH(1:horz)-nrss)/nrss,'b-','LineWidth',2)
#box off;
plt.title('Aggregate Hours, H','FontSize',fsize)
#set(gca,'FontSize',fsize);
 
#suptitle('The Evolution of an Epidemic');
orient landscape
print -dpdf -fillpage SIRmacro_epidemic_simulation_fig2
 
 
 
plt.figure
plt.subplot(3,1,1)
plt.plot(time,Us(1:HH),'b-','LineWidth',2) #; hold on;
plot(time,Ui(1:HH),'r--','LineWidth',2)    #;hold on;
plot(time,Ur(1:HH),'k-.','LineWidth',2)    #;hold on;
plot(time,U(1:HH),'m:','LineWidth',2)      #;hold off
#set(gca,'FontSize',fsize)                  #;
plt.legend(['Susceptibles','Infected','Recovered','Total','Location','best'])
#set(gca,'FontSize',fsize);
plt.title('Present Value Utility','FontSize',fsize);
#legend boxoff;
 
plt.subplot(3,1,2)
plt.plot(time,0*muc,'k:','LineWidth',1.5)  #; hold on
plt.plot(time,muc,'b-','LineWidth',2)      #; hold off
#set(gca,'FontSize',fsize);
plt.title('Containment Policy, \mu_c','FontSize',fsize)
plt.ylabel('%','FontSize',fsize)
 
plt.subplot(3,1,3)
plt.plot(time,100*pid*18/7,'k:','LineWidth',1.5)     #;  hold on
plt.plot(time,100*pid_endo*18/7,'b-','LineWidth',2)  #; hold off
#set(gca,'FontSize',fsize);
plt.title('Mortality Rate, \pi_d','FontSize',fsize)
plt.xlabel('Weeks','FontSize',fsize)
plt.ylabel('%','FontSize',fsize)
 
#suptitle('The Evolution of an Epidemic')
#orient portrait
#print -dpdf -fillpage SIRmacro_epidemic_simulation_fig3
 
 
# Save workspace
save results_baseline
 
# output some data used in paper and robustness table
aggCons_trough_percent                  = np.min((100*(aggC-crss)/crss))
aggCons_avg_first_year_percent          = np.mean((100*(aggC(1:52)-crss)/crss))
terminal_one_minus_susceptibles_percent = 100*(1-S(end))
peak_infection_percent                  = np.max(100*I)
terminal_death_share_percent            = 100*D(end)
terminal_number_deaths_US_millions      = terminal_death_share_percent/100*330
 

