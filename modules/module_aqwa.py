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

