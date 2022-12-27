import numpy as np

import scipy.interpolate as sci
import scipy.linalg as scl
import scipy.integrate as scn
import scipy.optimize as sco
import scipy.signal as scs
import scipy.stats as sct

def adminf(om,A,B):

    Tc,Fs = (100,10)
    time = np.arange(0,Tc,1/Fs)
    omo = np.linspace(min(om),max(om),500)
    f = sci.interp1d(om,B)
    Bo = f(omo)
    f = sci.interp1d(om,A)
    Ao = f(omo)

    kt = []
    for i,t in enumerate(time):
        kt = np.append(kt,2/np.pi*scn.simps(Bo*np.cos(omo*t),omo))

    Ainf = 0
    for i,w in enumerate(omo):
        Ainf = Ainf+Ao[i]+1/w*scn.simps(kt*np.sin(w*time),time)

    return(Ainf/len(omo),(omo,Ao))

def afun(p,t):
    return p[0]/t**2+p[1]/t**4-p[2]/t**5

def befun(p,om):
#    return p[0]*np.exp(p[1]*(om-0*p[2]))
    return p[0]/np.exp(p[1]*om)

def bfun(p,t):
    return p[0]/t**2+p[1]/t**4+p[2] #/t**8

def res_b(p,om,B):
    res = np.abs(befun(p,om) - B)**.5
    return res

def res_a(p,om,A):
    res = np.abs(befun(p,om) - A)**.5
    return res

def approx(om,B,dom,omE,per):

    nom = len(B)
    k = np.where(B==max(B))[0][0]
    if per==0:
        per = 1-(k+int(nom/20))/nom
    if per<.25:
        per = .25

    nB = int(per*nom)
    nB = 2
    inb = [-2,-1]

    norm = max(B)
    B = B/norm
    x,y = om[inb],B[inb]

    oMin = np.ceil(min(om)/dom)*dom
    omo = np.arange(oMin,x[0],dom)
    KND = 'linear'
    f = sci.interp1d(om,B,kind=KND)
    Bo = f(omo)

    oMin = np.floor(x[0]/dom)*dom
    omx = np.arange(oMin,omE,dom)

    p0 = [1,1]

    METHOD,LOSS = 'trf','soft_l1'
    res = sco.least_squares(res_b,p0,args=(x,y),method=METHOD,loss=LOSS)

    po = res['x']
    Bx = befun(po,omx)

    print('pars:',po)
    #if per==0:
        #p0 = [1,0]
        #METHOD,LOSS = 'trf','soft_l1'
        #res = sco.least_squares(res_b,p0,args=(x,y),method=METHOD,loss=METHOD)
        #po = res['x']
        #Bx = befun(po,omx)
    #else:
        #p0 = [1,1,0]
        #res = sco.least_squares(res_a,p0,args=(x,y),method='lm',loss='linear')
#
        #po = res['x']
        #Bx = afun(po,omx)

    omo = np.append(omo,omx)
    Bo  = np.append(Bo,Bx)

    return(omo,Bo*norm)

def damfn(p,s):
    P,Q = (0,0)
    nb = int(len(p)/2)
    for i,b in enumerate(p[:nb]):
        P = P + b*s**(2*(i+1))

    for i,d in enumerate(p[nb:]):
        Q = Q + d*s**(2*(i))
    Q = Q + s**(len(p)+2)
    return(P/Q)

def damf(s,*p):
    P,Q = (0,0)
    nb = int(len(p)/2)
    for i,b in enumerate(p[:nb]):
        P = P + b*s**(2*(i+1))

    for i,d in enumerate(p[nb:]):
        Q = Q + d*s**(2*(i))
    Q = Q + s**(len(p)+2)
    return(P/Q)

def admf(s,*p):
    P,Q = (0,0)
    nb = int(len(p)/2)
    for i,b in enumerate(p[:nb]):
        P = P + b*s**(2*(i))

    for i,d in enumerate(p[nb:]):
        Q = Q + d*s**(2*(i))
    Q = Q + s**(len(p)+2)
    return(P/Q)

def ktf(tim,om,B):
    kt = []
    for t in tim:
        kt = np.append(kt,2/np.pi*scn.simps(B*np.cos(om*t),om))
    return(kt)

def ktfn(tim,om,B):
    kt = []
    omo = np.linspace(min(om),max(om),101)
    fb = sci.interp1d(om,B)
    for t in tim:
        if t == 0:
            M = 2
        else:
            M = 1
        fkt = (M+np.sign(t))/np.pi*scn.simps(fb(omo)*np.cos(omo*t),omo)
        #kt = np.append(kt,2/np.pi*scn.simps(B*np.cos(om*t),om))
        kt = np.append(kt,fkt)
    return(kt)

def komf(tim,om,kt,Anf,Bnf):

    omo = np.linspace(min(om),max(om),501)
    Ak,Bk = [],[] 
    for i,o in enumerate(omo):
        ckt =  kt*np.cos(o*tim)
        skt = kt*np.sin(o*tim)
        Bk = np.append(Bk,scn.simps(ckt,tim))
        Ak = np.append(Ak,-scn.simps(skt,tim)/o)
    fa = sci.interp1d(omo,Ak)
    fb = sci.interp1d(omo,Bk)
    return (Anf-fa(om),fb(om)+Bnf)

def komfn(tim,om,kt,Anf,Bnf):
    omo = np.linspace(min(om),max(om),501)
    Kt = []
    for i,o in enumerate(om):
        Kt = np.append(Kt,scn.simps(kt*np.exp(-1j*o*tim),tim))

    return np.imag(Kt)/om+Anf,np.real(Kt)+Bnf

def era_sys(y,Fs,rnMax,pStat):

    n  = len(y)
    rn = 2

    r2,rmse,Einf = 0,.02,1e2
    H = scl.hankel(y[1:]/Fs)

    #print('\nirf:',n)

    U,S,V = scl.svd(H)

    while ((r2<.998) or (rmse>5e-2) or (np.log(Einf)>4.1)):

        Un  = U[:n-2,:rn]
        Vn  = V.T[:n-2,:rn]
        Um  = U[1:n-1,:rn]
        Ss  = np.sqrt(S[:rn].reshape(rn,1))
        iSs = 1 / Ss
        Ub  = Un.T@Um

        At = Ub*(iSs@Ss.T)
        Bt = Vn[0,:].reshape(rn,1)*Ss
        Ct = Un[0,:].reshape(1,rn)*Ss.T
        Dt = y[0] #/Fs


        CA = 1/2/Fs
        CB = 1
        CC = -CA
        CD = 1

        iM = scl.inv(CA*np.eye(rn)-CC*At)
        An = np.dot(CB*At-CD*np.eye(rn),iM)
        Bn = (CA*CB-CC*CD)*iM@Bt
        Cn = Ct@iM
        Dn = Dt+CC*np.dot(Ct@iM,Bt)

        tt,kt = scs.impulse((An,Bn,Cn,Dn),T=np.arange(0,n/Fs,1/Fs))
        stat  = sct.pearsonr(kt,y)
        rmse = np.sqrt(np.mean((y-kt)**2))/np.std(y)
        r2 = stat[0]**2
        k = np.where(y!=0)[0]
        E1   = np.mean(np.abs((y[k]-kt[k])/y[k]))
        Einf = np.max(np.abs((y[k]-kt[k])/y[k]))
        if rn>=rnMax:
            break
        else:
            rn+=1
    if pStat:
        print('  E1:',E1)
        print('Einf:',Einf)
        print('RMSE:',rmse)
        print('  R2:',r2)
        print('Rank:',rn-1)

    (Ph,Qh) = scs.ss2tf(An,Bn,Cn,Dn)

    return((Ph,Qh),(An,Bn,Cn,Dn))

def synth(Ts,Fs,rao,wave,SEED):

    if SEED:
        rnds = np.random.seed(SEED)

    dom = 2*np.pi/Ts
    om,am,ph = rao
    oMin,oMax = np.ceil(min(om)/dom)*dom,np.floor(max(om)/dom)*dom

    ts = np.arange(0,Ts,1/Fs)
    omi = np.arange(oMin,oMax,dom)
    nomi = len(omi)

    fa = sci.interp1d(om,am)
    fp = sci.interp1d(om,ph)

    amp = fa(omi)
    pha = fp(omi)

    fi = omi/2/np.pi
    s,f = Jonswap(wave,fi)

    sr  = s*amp**2
    ar  = np.sqrt(2*sr/Ts)

    pr = (1-2*np.random.random(nomi))*np.pi+pha

    y = 0
    for o,a,p in zip(omi,ar,pr):
        y += a*np.cos(o*ts+p)

    return (y,ts)

def state_space(om,A,B,NS,idx,PLT):

    pDir = 'png/'
    ir,ic = idx
    Fs   = 10.
    Tres = 100.
    N    = int(Fs*Tres)
    omE  = 2*np.pi*Fs/2
    dom  = omE/N

    om = np.append(1e-6,om)
    A  = np.append(A[0],A)
    B  = np.append(0,B)
    Ainf = adminf(om,A,B)
    #print('Ainf:',Ainf)
    nPow = int(np.log10(np.abs(Ainf)))
    norm = 10**nPow

    A,Ainf,B = (A/norm,Ainf/norm,B/norm)

    ###### Extrapolation to short waves
    Ao,omo = approx(om,A-Ainf,dom,omE)
    Ao = Ao + Ainf
    Bo,omo = approx(om,B,dom,omE)

    Npb = 2*NS-1
    Npb_b = int(Npb/2)
    p0 = np.ones(Npb)
    p0[Npb_b] = 0
    lLim = np.array([-np.inf for i in range(Npb)])
    uLim = [np.inf for i in range(Npb)]
    w = np.zeros_like(Bo)+1e-3
    w = None
    #pbo, pc = sco.curve_fit(damf,omo,Bo,p0,sigma=w,absolute_sigma=True,method='dogbox')
    #pbo = np.round(pbo,4)

    del_omo = np.abs(omo-1.)
    k = np.where(del_omo==min(del_omo))[0][0]

    def funp(p,omo,Bo):
        w = np.ones_like(Bo)
        #w[:k] = .9+.1*omo[:k]**2

        res = w*np.abs(damfn(p,omo)**2-Bo**2)**2
        return(res)

    def jacp(p,omo,Bo):
        J = np.empty((omo.size,p.size))
        nd = int(len(p)/2)
        den = np.polyval(p[nd:],omo)
        num = np.polyval(p[:nd],omo)

        for i in range(0,nd):
            J[:,i] = omo**i / den
        for i in range(nd,len(p)):
            J[:,i] = -num*omo**i / den**2

        return J

    res = sco.least_squares(funp,p0,args=(omo,Bo),method='trf',loss='soft_l1')

#    print('cfit:',pbo)
    pbo = np.round(res['x'],6)
    #print(' lsq:',pbo)

    Npa = Npb+1
    Npa_a = int(Npa/2)
    p0 = np.zeros(Npa)

    #p0[-Npa_a:] = pbo[Npb_b:]
    #p0[Npb_b] = Ainf

    lLim = p0 - 1e-3
    uLim = p0 + 1e-3
    lLim[:Npa_a] = -np.inf
    uLim[:Npa_a] = np.inf
    w = 1/np.abs(Ao)**(1/1)
    #w = np.zeros_like(Ao)+1e-2
    w = None
    #pao, pc = sco.curve_fit(admf,omo,Ao-Ainf,p0,bounds=(tuple(lLim),tuple(uLim)),sigma=w,absolute_sigma=True)

    def funp(p,omo,Ao,Ainf):
        w = np.ones_like(Ao)
        res = w*np.abs(damfn(p,omo)-(Ao-Ainf))**2
        return(res)

    res = sco.least_squares(funp,p0,args=(omo,Ao,Ainf),method='trf',loss='soft_l1')

#    print('cfit:',pao)
    pao = np.round(res['x'],6)
    #print(' lsq:',pao)
    #if ir==ic:
    if PLT:
        fig,(axa,axb) = plt.subplots(2,figsize=(5,4))
        axb.plot(om,B,'-*')
        axb.plot(omo,damf(omo,*pbo))
        axb.set_xlabel(r'$\omega$ [rad/s]')
        axb.set_xscale('log')
        axb.set_ylabel('B$_{}%i%i}$'%(ir,ic))

        axa.plot(om,A-Ainf,'-*')
        axa.plot(omo,admf(omo,*pao))
        axa.set_xscale('log')
        axa.set_ylabel(r'A$_{%i%i}$-A$_\infty$'%(ir,ic))
        axa.annotate('1e%i'%nPow,(1.1*min(om),0.9*max(A)))
        fig.savefig(pDir+'coef/Coef%i%i.png'%(ir,ic),dpi=150)
        plt.close(fig)

    #aVec = np.flip(pao[:Npa_a])
    bVec = np.append(np.flip(pbo[:Npb_b]),0)
    dVec = np.append(1,np.flip(pbo[Npb_b:]))

    tet,gam,_ = scs.residue(bVec,dVec,rtype='min',tol=1e-18)

    if 10*ir+ic == 31:
        print(gam)

    lmb = np.sqrt(-gam+0j)

    k = np.where(np.real(lmb)>0)[0]
    lmb[k] = -lmb[k]
    psi = -tet/lmb

    Px,Qx = scs.invres(psi,lmb,[])
    SS = scs.tf2ss(norm*Px,Qx)
    if PLT:

        K = B + 1j*om*(A-Ainf)
        omi = np.linspace(min(om),max(om),301)
        Ki = np.polyval(Px,1j*omi)/np.polyval(Qx,1j*omi)
        fig,(axa,axp) = plt.subplots(2,figsize=(5,4))
        axa.set_title('$K_{%i%i}$'%(ir,ic))
        axa.plot(om,np.abs(K),'*-')
        axa.plot(omi,np.abs(Ki))
        axa.annotate('1e%i'%nPow,(1.1*min(om),0.9*max(np.abs(K))))
        axp.plot(om,np.angle(K),'*-')
        axp.plot(omi,np.angle(Ki))
        fig.savefig(pDir+'retard/K%i%i.png'%(ir,ic),dpi=150)
        plt.close(fig)

    return((Px*norm,Qx),SS)


def prony(F, Fs, m):
    """ Input : real arrays F - Impulse Response Function (IRF)
              : real number Fs - sampling frequency of IRF
              : integer m - the number of modes in the exponential fit
       Output : arrays a and b such that F(t) ~ sum ai exp(bi*t)"""

    # Solve LLS problem in step 1
    # Amat is (N-m)*m and bmat is N-m*1
    N    = len(F)
    Amat = np.zeros((N-m, m))
    bmat = F[m:]

    for jcol in range(m):
        Amat[:, jcol] = F[m-jcol-1:N-1-jcol]

    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    d = sol[0]

    # Solve the roots of the polynomial in step 2
    # first, form the polynomial coefficients
    c = 1j*np.zeros(m+1)
    c[m] = 1.+0j
    for i in range(1,m+1):
        c[m-i] = -d[i-1]

    u = np.roots(np.flip(c))
    b_est = np.log(u)*Fs

    # Set up LLS problem to find the "a"s in step 3
    Amat = 1j*np.zeros((N, m))
    bmat = F

    for irow in range(N):
        Amat[irow, :] = u**irow

    sol = np.linalg.lstsq(Amat, bmat ,rcond=None)
    a_est = sol[0]

    return (a_est, b_est)
