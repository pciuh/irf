#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:19:21 2020

@author: pciuh
"""

import numpy as np
import pandas as pd

import scipy.interpolate as sci
import scipy.linalg as scl
import scipy.integrate as scn
import scipy.optimize as sco
import scipy.signal as scs
import scipy.stats as sct

import matplotlib.pyplot as plt


G=9.80665

def moments(x,y,N):
    m = np.zeros(N)
    for i in range(N):
        m[i] = np.trapz(y*x**i,x)
    return(m)
    
def countomega(iFil):
    
    bStr = 'NO.  FREQUENCY FREQUENCY'
    eStr = '***** HYDRODYNAMIC DATABASE NOW USING ENCOUNTER*****'

    fil=open(iFil,'r')
    c=0
    Lin = fil.readlines()
    for line in Lin:
        if line.strip()[0:27]==bStr:
            sb=c
            
        if line.strip()[0:54]==eStr:
            se=c
        c=c+1    
    
    return(se-sb-7)


def readomega(iFil,nom):
############################   Reads frequencies from AQWA *.LIS files

    fil=open(iFil,'r')
    Lin=fil.readlines()
    c=0
    ############## INETRTIA MATRIX

    for line in Lin:
        if line.strip()[0:27] == 'NO.  FREQUENCY FREQUENCY':
            sb=c
            #break
        c=c+1

    om  = np.zeros(nom)
    ome = np.zeros(nom)
    sb  = sb + 2
    for i in range(nom):
        om[i]  = float(Lin[sb+i].strip()[5:15])
        ome[i] = float(Lin[sb+i].strip()[15:25])
    
    return(om,ome)
    

def readcoeff(iFil,nom):
############################   Reads Coefficients from AQWA *.LIS files

    fil=open(iFil,'r')
    Lin=fil.readlines()

    c=0
    ############## INETRTIA MATRIX

    for line in Lin:
        if line.strip()[0:14] == 'INERTIA MATRIX':
            sb=c
            #break
        c=c+1

    ############## MASS OF STRUCTURE

    imx=np.zeros((6,6))

    ms=float(Lin[sb-17].strip()[24:36])
    for ir in range(3):
        imx[ir,ir]=ms

        im={}
    ind=[2,0,0]
    for ir in range(3):
        tmat=np.array([float(x) for x in Lin[sb+ir*2].split()[ind[ir]:]])
        imx[ir+3,3:6]=tmat.T
    im={'IM':imx}

    c=0
    for line in Lin:
        if line.strip() == 'STIFFNESS MATRIX':
            sb=c+7
            break
        c=c+1

    ############## STIFFNESS MATRIX

    smx=[]
    for ir in range(6):
        smx=np.append(smx,np.array([float(x) for x in Lin[sb+ir*2].split()[1:]]))

    sm={'SM':smx.reshape(6,6)}
#    print(sm)


    ############## ADDED MASS AND DAMPING MATRIX

    c=0
    for line in Lin:
#        if line.strip()==rSTR:
        if line.strip()=='ADDED  MASS':
            ib=c+7
            break
        c=c+1

    NL=45
    amx=[]
    am={}
    cmx=[]
    cm={}
    c=ib
    for i in range(nom):
        om=float(Lin[c-10].split()[3])
        amx=[]
        cmx=[]
        for ir in range(6):
            amx=np.append(amx,np.array([float(x) for x in Lin[c+ir*2].split()[1:]]))
            cmx=np.append(cmx,np.array([float(x) for x in Lin[20+c+ir*2].split()[1:]]))
        c=c+NL
        am[om]=amx
        cm[om]=cmx
    return(am,cm,sm,im)

def readrao(iFil,rTYP,nom):
############################   Reads RAOs from AQWA *.LIS files
    fil=open(iFil,'r')
    Lin=fil.readlines()

    if rTYP=='R':
        rSTR='R.A.O.S-VARIATION WITH WAVE PERIOD/FREQUENCY'
    elif rTYP=='FD':
        rSTR='DIFFRACTION FORCES-VARIATION WITH WAVE PERIOD/FREQUENCY'
    elif rTYP=='FK':
        rSTR='FROUDE KRYLOV FORCES-VARIATION WITH WAVE PERIOD/FREQUENCY'
    elif rTYP=='FA':
        rSTR='FROUDE KRYLOV + DIFFRACTION FORCES-VARIATION WITH WAVE PERIOD/FREQUENCY'
    else:
        print('Wrong Switch!')
        
    c=0
    ib=0
    for line in Lin:
        if line.strip()==rSTR:
            ib=c+7
        c=c+1

    STR='STRUCTURE    FREQUENCY FREQUENCY  PERIOD     WAVE      WAVE    MAX ELEM    DEPTH RATIOS    PARAMETERS'
    c=0
    for line in Lin:
        if line.strip()==STR:
            fb=c+3
        c=c+1
    
    ie=ib+nom
    fe=fb+nom
    ome=np.zeros(nom)
    om=np.zeros(nom)
    rao=np.zeros((nom,12))

    pref=['XX','YY','ZZ','RX','RY','RZ']
    suff=['A','P']
    
    keys=[]
    for pr in pref:
        for sf in suff:
            keys=np.append(keys,pr+sf)
        
    dcr={}
    c=0
    for line in Lin[fb:fe]:
        om[c]=line[16:24]
        c=c+1
    c=0   
    for line in Lin[ib:ie]:
        ome[c]=line[8:13]
        rao[c]=np.array([float(x) for x in line[24:130].split()])
        
        c=c+1
    dcr=dict(zip(keys,rao.T))
    df=pd.DataFrame(dcr)
    df.insert(loc=0,column='OME',value=ome)
    df.insert(loc=0,column='OM',value=om)

    return (df)
def secord(iFil,nom):

    fil1 = open(iFil,'r')
    Lin = fil1.readlines()

    rStr = ['SURGE(X)','SWAY(Y)','YAW(RZ)']

    qtf = []
    for rst in rStr:
        c=0
        for line in Lin:
            if line.strip()==rst:
                ib=c+1
                ie=c+1+nom
            c=c+1
        c = 0
        for line in Lin[ib:ie]:
            qtf = np.append(qtf,np.array([float(x) for x in line[20:30].split()]))

        qtf = qtf.reshape(3,-1)
        dcr = dict(zip(rStr,qtf))
        df = pd.DataFrame(dcr)
    return(df)


def calc_faw(iFil,IND):
############################   Calculates mean second order forces
    
    file1 = open(iFil, 'r') 
    Lines = file1.readlines() 
    cs=((8,80))

    line=str(Lines[1:2])
    nf=int(line[6:8])

    nl=np.size(Lines)
    nc=5
    nt=int((nl-nc)/4)


    count=int(np.ceil(nf/6))+2
    count0=count

    qtf=np.zeros((nl))
    om=np.array([])
    for line in Lines[2:count]:
        om=np.append(om,np.array(line[cs[0]:cs[1]].split(),dtype=np.float64))

    for line in Lines[count:]:
        sline=np.array(line[cs[0]:cs[1]].split(),dtype=np.float64)
        qtf[count-count0]=sline[IND]
        count=count+1

    Pd=qtf[0:-3:4]
#    print(Pd)
    Qd=qtf[1:-2:4]
    Ps=qtf[3:-1:4]
    Qs=qtf[4::4]

    Pdm=np.zeros(nf)
    Qdm=np.zeros(nf)
    Psm=np.zeros(nf)
    Qsm=np.zeros(nf)
    for i in range(nf):
        ind=i*nf+i
        Pdm[i]=Pd[ind]
        Qdm[i]=Qd[ind]
        Psm[i]=Ps[ind]
        Qsm[i]=Qs[ind]
    
    
    Fdm=np.sqrt(Pdm**2+Qdm**2)*np.sign(Pdm)
    Fdm=Pdm
    PHd=np.arctan2(Qdm,Pdm)

    Fsm=np.sqrt(Psm**2+Qsm**2)*np.sign(Psm)
    PHs=np.arctan2(Qsm,Psm)
    
    Fdms=Pdm*np.cos(PHd)+Qdm*np.sin(PHd)
    Fsms=Psm*np.cos(PHs)+Qsm*np.sin(PHs)

    return(om,Fdm,Fsm,PHd,PHs)
    
def calc_rao(iFil,nom,nC):
############################   Calculates RAO from input coefficients & forces

    fil=open(iFil,'r')
    Lin=fil.readlines()    

    c=0
    ############## INETRTIA MATRIX

    for line in Lin:
        if line.strip()[0:91] == '4. SMALL ANGLE STABILITY PARAMETERS':
            sb=c+5
            #break
        c=c+1

    GMX=float(Lin[sb].strip()[50:60])
    
    am,cm,sm,im=readcoeff(iFil,nom)
    om,_=readomega(iFil,nom)

    fk=readrao(iFil,'FK',nom)
    fd=readrao(iFil,'FD',nom)

    per=np.array([float(x) for x in am.keys()])
    ome=2*np.pi/per
    fdd=np.array(fk.to_dict('split')['data'])
    fka=fdd[:,2:-1:2]
    fkp=fdd[:,3::2]*np.pi/180

    fdd=np.array(fd.to_dict('split')['data'])
    fda=fdd[:,2:-1:2]
    fdp=fdd[:,3::2]*np.pi/180

    pref=['XX','YY','ZZ','RX','RY','RZ']
    suff=['A','P']

    keys=['OME']
    for pr in pref:
        for sf in suff:
            keys=np.append(keys,pr+sf)


    vres=np.zeros((nom,13))
    fres=np.zeros((nom,13))

    for i in range(nom):

        omo=ome[i]
        fao=fka[i]*np.exp(1j*fkp[i])+fda[i]*np.exp(1j*fdp[i])
        imo=im['IM']
        smo=sm['SM']
        amo=np.array(am[per[i]]).reshape(6,6)
        cmo=np.array(cm[per[i]]).reshape(6,6)

        if nC>0:

#            print('mu.in: {:6.5f}'.format(nC))
#            print('bef: {:.1e}'.format(cmo[3,3]))

            mn=np.sqrt(G*imo[0,0]*GMX*(imo[3,3]+amo[3,3]))
            b44=2*nC*mn
            cmo[3,3]=b44
#            print('aft: {:.1e}'.format(cmo[3,3]))


        x=smo-(imo+amo)*omo**2-1j*cmo*omo
        xm=np.asmatrix(x)
        h=np.linalg.inv(x)
#        hm=xm.I
        xo=h@fao.T
#        xom=np.dot(hm,fao)

        xa=np.abs(xo)
        xp=np.angle(xo)*180/np.pi

        faa=np.abs(fao)
        fap=np.angle(fao)*180/np.pi

        fres[i,0]      = omo
        fres[i,1:-1:2] = faa
        fres[i,2::2]   = fap

        vres[i,0]=omo
        vres[i,1:-1:2]=xa
        vres[i,2::2]=xp
        vres[i,[7,9,11]]=vres[i,[7,9,11]]*180/np.pi

    df=dict(zip(keys,fres.T))
    dfa=pd.DataFrame(df)
    dfa.insert(loc=0,column='OM',value=om)

    dr=dict(zip(keys,vres.T))
    drao=pd.DataFrame(dr)
    drao.insert(loc=0,column='OM',value=om)

    return(drao,dfa)

def hydcoef(iFil,nom):
    aStr = 'ADDED MASS-VARIATION WITH WAVE PERIOD/FREQUENCY'
    bStr = 'DAMPING-VARIATION WITH WAVE PERIOD/FREQUENCY'
    cNam = [11,22,33,44,55,66,13,15,24,26,35,46]
    aNam = ['A%i'%x for x in cNam]
    bNam = ['B%i'%x for x in cNam]
    nc   = len(cNam)

    fil=open(iFil,'r')
    Lin=fil.readlines()

    c=0
    ############## Added Mass
    for line in Lin:
        if line.strip() == aStr:
            ba=c+5
            #break
        c=c+1
    c=0
    ############## Added Mass
    for line in Lin:
        if line.strip() == bStr:
            bb=c+5
            #break
        c=c+1

    adm,dmp = [],[]

    ea = ba+nom
    eb = bb+nom

    for line in Lin[ba:ea]:
        adm = np.append(adm,np.array([float(x) for x in line[12:].split()]))

    for line in Lin[bb:eb]:
        dmp = np.append(dmp,np.array([float(x) for x in line[12:].split()]))

    adm,dmp = (adm.reshape(-1,12),dmp.reshape(-1,12))

    dcr = dict(zip(aNam,adm.T))
    dfa = pd.DataFrame(dcr)

    dcr = dict(zip(bNam,dmp.T))
    dfb = pd.DataFrame(dcr)

    return(dfa,dfb)



def dampc(iFil,nom,pDZ,wave):
##############################   Finding damping basend on wave input
    nfi = 11
    fi  =np.linspace(0,25,nfi)
    mu  = np.polyval(pDZ,fi) #pDZ[1]*fi+pDZ[0]
    sKind='quadratic'

    rms=np.zeros(nfi)
    for i,dz in enumerate(mu):

        drao,_=calc_rao(iFil,nom,dz)
        om=drao.OM.to_numpy()
        rx=drao.RXA.to_numpy()
        om=np.insert(om,0,0.01)
        rx=np.insert(rx,0,0)
        om=np.append(om,2*max(om))
        rx=np.append(rx,0)
        omo=np.linspace(min(om),max(om),101)
        frx=sci.interp1d(om,rx,kind=sKind)
        rxo=frx(omo)
        f=omo/2/np.pi
        s=Jonswap(wave, f)
        rxs=s*rxo**2

        rms[i]=np.sqrt(np.trapz(rxs,f))

    MN=np.sqrt(2)
#    print('MN: {:.3f}'.format(MN))
    fia=MN*rms
    dfi=np.abs(fi-fia)
    N=np.where(dfi==min(dfi))
    ind=N[0]
#    print(ind)

    nInt=3
    idfi=np.zeros(nInt)
    ifi=np.zeros(nInt)
    ifa=np.zeros(nInt)
    for i in np.arange(-1,2,1):
        ifi[i]=fi[ind+i]    
        ifa[i]=fia[ind+i]


    idfi=ifa-ifi
    pfi=np.polyfit(idfi,ifi,nInt-1)
    fio=np.polyval(pfi,0)
    MU = np.polyval(pDZ,fio)

    return(MU)    

def Jonswap(*args):

    wave = args[0]
    Hs    = wave['HS']
    Tp    = wave['TP']
    Gam   = wave['GAM']
    fp    = 1/Tp

    fmin = (0.6477+0.005357*Gam-0.0002625*Gam**2)*fp                ####  0.1 energy threshold
    fmax = (6.3204-0.4377*Gam+0.05261*Gam**2-0.002839*Gam**3)*fp    #### 99.9 energy threshold

    if len(args)<2:
        f = np.linspace(fmin,fmax,303)
    else:
        f = args[1]

    f[f==0]=1e-10

    tau = 0.07*np.ones_like(f)
    tau[f>fp] = 0.09

    alf = 0.0081

    alf = 0.0624/(0.23+0.0336*Gam-0.185/(1.9+Gam))

    A = alf*Hs**2*f**-5*fp**4
    #A = alf*G**2/(2*np.pi)**4/f**-5
    B = np.exp(-5/4*(fp/f)**4)
    C = np.exp(-(f-fp)**2/(2*tau**2*fp**2))

    return(A*B*Gam**C,f)

def jonswap(wave, ft):

    Hs    = wave['HS']
    Tp    = wave['TP']
    gamma = wave['GAM']

    fp = 1/Tp
    beta = 0.06238/(0.230+0.0336*gamma-0.185*(1.9+gamma)**(-1))
    d = np.exp(-(ft/fp-1)**2/(2*0.07**2))
    s = beta*Hs**2*Tp**(-4)*ft**(-5)*np.exp(-5/4*(Tp*ft)**(-4))*gamma**d
    s[ft==0] = 0
    return (s)

def rotmat(fi,teta,psi):

    vr,vp,vy=[np.eye(3,3,dtype=complex) for i in range(3)]

    vr[1,:]=np.array([0+0j,np.cos(fi),-np.sin(fi)])
    vr[2,:]=np.array([0+0j,np.sin(fi), np.cos(fi)])

    vp[0,:]=np.array([ np.cos(teta),0+0j,np.sin(teta)])    
    vp[2,:]=np.array([-np.sin(teta),0+0j,np.cos(teta)])    

    vy[0,:]=np.array([ np.cos(psi),-np.sin(psi),0+0j])    
    vy[1,:]=np.array([ np.sin(psi), np.cos(psi),0+0j])    

    va=(vy@vp)@vr

    return(va)

def kinematics(ome,xx):
    v=1j*ome*xx
    a=-ome**2*xx
    return(a,v)

def new_coor(df,vi):

    om=df.OM.to_numpy()
    ome=df.OME.to_numpy()
    xx=(df.XXA*np.exp(1j*df.XXP*np.pi/180)).to_numpy()
    yy=(df.YYA*np.exp(1j*df.YYP*np.pi/180)).to_numpy()
    zz=(df.ZZA*np.exp(1j*df.ZZP*np.pi/180)).to_numpy()
    rx=(df.RXA*np.exp(1j*df.RXP*np.pi/180)).to_numpy()
    ry=(df.RYA*np.exp(1j*df.RYP*np.pi/180)).to_numpy()
    rz=(df.RZA*np.exp(1j*df.RZP*np.pi/180)).to_numpy()
    nf=len(om)

    sn,vn,an=[np.zeros((nf,3),dtype=complex) for i in range(3)]
    snv,vnv,anv=[np.zeros((nf,7),dtype=float) for i in range(3)]

    for i in range(nf):
        vt=np.array([xx[i],yy[i],zz[i]])    ####### wektor przesuniec
        vr=rotmat(rx[i]*np.pi/180,ry[i]*np.pi/180,rz[i]*np.pi/180)        ####### wektor obrotow
        vv=vt+vr@vi                         ####### wektor obranego punktu do analizy
        sn[i,:]=vv-vi
        an[i,:],vn[i,:]=kinematics(ome[i],vv-vi)

    anv[:,0]=om
    anv[:,1:-1:2]=np.abs(an)
    anv[:,2::2]=np.degrees(np.angle(an))

    vnv[:,0]=om
    vnv[:,1:-1:2]=np.abs(vn)
    vnv[:,2::2]=np.degrees(np.angle(vn))

    snv[:,0]=om
    snv[:,1:-1:2]=np.abs(sn)
    snv[:,2::2]=np.degrees(np.angle(sn))

    pref=['XX','YY','ZZ']
    suff=['A','P']

    keys=['OM']
    for pr in pref:
        for sf in suff:
            keys=np.append(keys,pr+sf)

    dca=dict(zip(keys,anv.T))
    dfa=pd.DataFrame(dca)
    dca=dict(zip(keys,vnv.T))
    dfv=pd.DataFrame(dca)
    dca=dict(zip(keys,snv.T))
    dfs=pd.DataFrame(dca)

    return(dfa,dfv,dfs)

def relmot(nrao):

    zna=nrao['ZZA']
    znp=nrao['ZZP']
    zwa=np.ones_like(znp.to_numpy())
    zwp=np.zeros_like(zwa)

    zn=zna*np.exp(1j*np.radians(znp))
    zw=zwa*np.exp(1j*zwp)
    sn=zn-zw

    sna=np.abs(sn)
    snp=np.degrees(np.angle(sn))

    keys=['ZNA','ZNP','SNA','SNP']
    vec=[zna,znp,sna,snp]
    dc=dict(zip(keys,vec))

    return(pd.DataFrame(dc))

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
