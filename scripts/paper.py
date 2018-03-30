from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.constants import G
import astropy.units as u
import corner
import myutils

import scipy
from scipy.special import gamma, hyp2f1, gammainc

from dm import Sample, trim_chain

speclabels = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
logglabels = {0: 'dwarfs', 1: 'giants'}

def hrd():
    """"""
    s = Sample()
    
    plt.close()
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.plot(s.JH, s.MJ, 'ko', ms=1, alpha=0.01, rasterized=True)
    
    plt.xlabel('J - H')
    plt.ylabel('$M_J$')
    plt.xlim(-0.2, 1.2)
    plt.ylim(7, -6)
    
    plt.sca(ax[1])
    plt.plot(s.JH, s.MJ, 'ko', ms=1, alpha=0.01, rasterized=True, label='')
    
    logg = [s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 1, 1]
    teff = [2, 3, 5, 6]
    Npop = len(teff)
    icol = np.array([0.22*(i+0.5) for i in range(Npop)])[::-1]
    
    for i in range(Npop):
        psel = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20)
        color = mpl.cm.magma(icol[i])
        plt.plot(s.JH[psel], s.MJ[psel], 'o', color=color, ms=1, rasterized=True, label='{} {}'.format(speclabels[teff[i]], logglabels[logg_id[i]]))
    
    plt.legend(loc=4, fontsize='x-small', markerscale=5, handlelength=0.4)
    plt.xlabel('J - H')
    plt.xlim(-0.2, 1.2)
    plt.ylim(7, -6)
    
    plt.tight_layout()
    plt.savefig('../tex/hrd.pdf', dpi=300)

def velocities():
    """"""
    s = Sample()
    
    select = s.dwarf & s.spectype[2] & (s.verr[:,2]<20) & (s.x[:,2].value>0.) & (s.x[:,2].value<0.1)

    
    # cylindrical coordinates
    vz = s.v[:,2].value
    
    vx = s.v[:,0].value
    vy = s.v[:,1].value
    thx = np.arctan2(s.x[:,1].value, s.x[:,0].value)
    thv = np.arctan2(s.v[:,1].value, s.v[:,0].value)
    vr = np.sqrt(vx**2 + vy**2) * np.cos(thx+thv)
    
    print(np.shape(vr))
    
    vxe = s.verr[:,0]
    vye = s.verr[:,1]
    vze = s.verr[:,2]
    vre = np.sqrt((vx*vxe/vr)**2 + (vy*vye/vr)**2) * np.abs(np.cos(thx+thv))
    
    wz = vze**-2
    wr = vre**-2
    print(np.percentile(wr[select], [5,50,95]))
    print(np.percentile(wz[select], [5,50,95]))
    
    print(np.percentile(vre[select], [5,50,95]))
    print(np.percentile(vze[select], [5,50,95]))
    
    N = 50
    zbins = np.linspace(-100,100,N)
    rbins = np.linspace(-200,200,N)
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(8,8), gridspec_kw = {'height_ratios':[0.2, 1], 'width_ratios':[1, 0.2], 'hspace':0.1, 'wspace':0.1}, sharex='col', sharey='row')
    
    ax[0][1].axis('off')
    
    plt.sca(ax[0][0])
    plt.hist(vr[select], weights=wr[select], normed=True, bins=rbins, color='k', alpha=0.3)
    
    plt.sca(ax[1][1])
    plt.hist(vz[select], weights=wz[select], normed=True, bins=zbins, orientation='horizontal', color='k', alpha=0.3)
    
    plt.sca(ax[1][0])
    plt.plot(vr[select], vz[select], 'k.', ms=1)
    
    plt.xlim(rbins[0], rbins[-1])
    plt.ylim(zbins[0], zbins[-1])
    
    plt.xlabel('$V_R$ (km s$^{-1}$)')
    plt.ylabel('$V_Z$ (km s$^{-1}$)')

    #plt.tight_layout()

def fulljeans_bestfit(logg=1, teff=5, l=39, nstart=0):
    """Plot best-fit model after fulljeans solution obtained"""
    
    if type(logg) is list:
        logglabel = '.'.join(str(x) for x in logg)
        tefflabel = '.'.join(str(x) for x in teff)
        llabel = '.'.join(str(x) for x in l)
    else:
        logglabel = logg
        tefflabel = teff
        llabel = l
        logg = [logg]
        teff = [teff]
        l = [l]
    
    njoint = 3
    nindividual = 6    

    extension = ''
    dname = '../data/chains/fulljeans_logg{}_teff{}_z{}_s0'.format(logglabel, tefflabel, llabel)
    d = np.load('{}{}.npz'.format(dname, extension))
    chain = d['chain']
    lnp = d['lnp']
    
    id_best = np.argmax(lnp)
    xall = chain[id_best]
    
    Npop = len(logg)
    icol = np.array([0.22*(i+0.5) for i in range(Npop)])[::-1]
    
    plt.close()
    fig, ax = plt.subplots(2,3, figsize=(15,7), gridspec_kw = {'height_ratios':[5,2]}, sharex=True, squeeze=False)

    for i in range(Npop):
        # parameters
        #xlist = [xall[v] for v in range(njoint)] + [xall[v] for v in range(njoint + nindividual*i, njoint + (nindividual*i+1))]
        xlist = [xall[v] for v in range(njoint)] + [xall[v] for v in range(njoint+nindividual*i,njoint+nindividual*(i+1))]
        x = np.array(xlist)
        
        # data
        t = Table.read('../data/profile_ell_logg{}_teff{}_z{}_s0.fits'.format(logg[i], teff[i], l[i]))
        #t = t[t['z']<1]
        
        zmin = 0.2
        zmax = 1.
        if teff[i]<5:
            zmax = 0.5
            zmin = 0.1

        mask = (t['z']>zmin) & (t['z']<zmax)
        tm = t[mask]
        #nue = tm['nueff']/np.sqrt(tm['n'])
    
        # best fit
        z0 = 1*u.kpc
        z = np.linspace(0,1.2,100)*u.kpc
    
        nuzbest = x[6]*np.exp(-z.value/x[7])
        szbest = full_sz(z=z, sigs=x[0]*u.Msun*u.pc**-2, H=x[1]*u.kpc, rhodm=x[2]*u.Msun*u.pc**-3, D=x[3]*u.km**2*u.s**-2, n=x[4], R0=x[5]*u.kpc, nu0=x[6]*u.kpc**-3, h=x[7]*u.kpc, sz0=x[8]*u.km*u.s**-1)
        srzbest = x[3]*(z/z0)**x[4]
    
        nuzbest_ = x[6]*np.exp(-tm['zeff']/x[7])
        szbest_ = full_sz(z=tm['z']*u.kpc, sigs=x[0]*u.Msun*u.pc**-2, H=x[1]*u.kpc, rhodm=x[2]*u.Msun*u.pc**-3, D=x[3]*u.km**2*u.s**-2, n=x[4], R0=x[5]*u.kpc, nu0=x[6]*u.kpc**-3, h=x[7]*u.kpc, sz0=x[8]*u.km*u.s**-1).value
        srzbest_ = x[3]*(tm['z']*u.kpc/z0)**x[4]
    
        a = 0.2
        color = mpl.cm.magma(icol[i])
    
        plt.sca(ax[0][0])
        plt.plot(t['zeff'], t['nueff'], 'o', color=color, alpha=a)
        plt.errorbar(t['zeff'], t['nueff'], yerr=t['nue'], fmt='none', color=color, alpha=a)
        plt.plot(tm['zeff'], tm['nueff'], 'o', color=color)
        plt.errorbar(tm['zeff'], tm['nueff'], yerr=tm['nue'], fmt='none', color=color)
        plt.plot(z, nuzbest, '-', color=color, lw=2)
    
        plt.gca().set_yscale('log')
        plt.ylabel('$\\nu$ (kpc$^{-3}$)')

        plt.sca(ax[1][0])
        plt.axhline(0, color='r')
        plt.plot(tm['zeff'], 1-tm['nueff']/nuzbest_, 'o', color=color)
        plt.errorbar(tm['zeff'], 1-tm['nueff']/nuzbest_, yerr=tm['nue']/nuzbest_, fmt='none', color=color)
        plt.xlabel('Z (kpc)')
        plt.ylabel('$\Delta$ $\\nu$ / $\\nu$')
        plt.ylim(-1,1)
    
        plt.sca(ax[0][1])
        plt.plot(t['z'], t['sz'], 'o', color=color, alpha=a)
        plt.errorbar(t['z'], t['sz'], yerr=t['sze'], fmt='none', color=color, alpha=a)
        plt.plot(tm['z'], tm['sz'], 'o', color=color)
        plt.errorbar(tm['z'], tm['sz'], yerr=tm['sze'], fmt='none', color=color)
        plt.plot(z, szbest, '-', color=color, lw=2)
    
        plt.xlim(0,1.2)
        plt.ylim(0,50)
        plt.ylabel('$\sigma_{z}$ (km s$^{-1}$)')
    
        plt.sca(ax[1][1])
        plt.axhline(0, color='r')
        plt.plot(tm['zeff'], 1-tm['sz']/szbest_, 'o', color=color)
        plt.errorbar(tm['z'], 1-tm['sz']/szbest_, yerr=tm['sze']/szbest_, fmt='none', color=color)
        plt.xlabel('Z (kpc)')
        plt.ylabel('$\Delta$ $\sigma_z$ / $\sigma_z$')
        plt.ylim(-1,1)
    
        plt.sca(ax[0][2])
        plt.plot(t['z'], t['srz'], 'o', color=color, alpha=a)
        plt.errorbar(t['z'], t['srz'], yerr=t['srze'], fmt='none', color=color, alpha=a)
        plt.plot(tm['z'], tm['srz'], 'o', color=color)
        plt.errorbar(tm['z'], tm['srz'], yerr=tm['srze'], fmt='none', color=color)
        plt.plot(z, srzbest, '-', color=color, lw=2)

        plt.ylabel('$\sigma_{Rz}$ (km s$^{-1}$)')
        plt.ylim(-400,100)
    
        plt.sca(ax[1][2])
        plt.axhline(0, color='r')
        plt.plot(tm['zeff'], 1-tm['srz']/srzbest_, 'o', color=color)
        plt.errorbar(tm['z'], 1-tm['srz']/srzbest_, yerr=tm['srze']/srzbest_, fmt='none', color=color)
        plt.xlabel('Z (kpc)')
        plt.ylabel('$\Delta$ $\sigma_{Rz}$ / $\sigma_{Rz}$')
        plt.ylim(-1,1)
    
    plt.tight_layout()
    plt.savefig('../tex/fulljeans_bestfit_t{}.pdf'.format(i))

def full_sz(z=np.nan, A=15.3*u.km*u.s**-1*u.kpc**-1, B=-11.9*u.km*u.s**-1*u.kpc**-1, sigg=13.2*u.Msun*u.pc**-2, Rsun=8.3*u.kpc, z0=1*u.kpc, sigs=12*u.Msun*u.pc**-2, H=0.2*u.kpc, rhodm=0.01*u.Msun*u.pc**-3, D=324*u.km**2*u.s**-2, n=1.16, R0=1*u.kpc, nu0=1e6*u.kpc**-3, h=0.3*u.kpc, sz0=10*u.km*u.s**-1):
    """"""
    
    if np.any(~np.isfinite(z)):
        z = np.linspace(0,2,100)*u.kpc
    
    nuz = nu0*np.exp(-z/h)
    C = sz0**2 * nu0

    sz2 = (C/nuz + nu0/nuz * ( -(4*np.pi*G*rhodm - 2*(B**2 - A**2)) * h * (h - np.exp(-(z/h).value)*(h+z))
                             - 2*np.pi*G * (sigs + sigg) * (h - h*np.exp(-(z/h).value))
                             - 2*np.pi*G*sigs * h*H/(h + H) * (np.exp(-((z*(h+H))/(h*H)).value) - 1)
                             - D*(1/Rsun - 2/R0)*z0**-n * h**(n+1) * scipy.special.gammainc(n+1, (z/h).value)
        )).to(u.km**2*u.s**-2)
    
    sz = np.sqrt(sz2)
    
    return sz

def pdf_fulljeans(logg=[0,0,1,1], teff=[2,3,5,6], l=[99,99,39,39], nstart=0, nwalkers=200, extension=''):
    """Plot triangle plot with samples of R,z velocity ellipsoid parameters"""
    
    if type(logg) is list:
        logg = '.'.join(str(x) for x in logg)
        teff = '.'.join(str(x) for x in teff)
        l = '.'.join(str(x) for x in l)
    
    if len(extension)>0:
        prefix = extension + '/'
        suffix = '_' + extension
    else:
        prefix = extension
        suffix = extension
    
    dname = '../data/chains/{}fulljeans_logg{}_teff{}_z{}_s0'.format(prefix, logg, teff, l)
    d = np.load('{}.npz'.format(dname))
    chain = d['chain']
    
    nstep, ndim = np.shape(chain)
    nstep = int(nstep/nwalkers)
    
    samples = trim_chain(chain, nwalkers, nstart, ndim)
    
    labels = ['$\Sigma_d$', '$h_d$', '$\\rho_{dm}$', '$A_{tilt}$', '$n_{tilt}$', '$R_{tilt}$', '$\\nu_0$', 'h', '$\sigma_{z,0}$']
    plt.close()
    #fig = corner.corner(samples[:,:3], cmap='gray', quantiles=[0.16,0.50,0.84], angle=0, plot_contours=True, plot_datapoints=False, smooth1d=True, labels=labels, show_titles=True, verbose=True, bins=50, title_fmt='.3f', range=[[7, 40], [0.09,0.4], [0, 0.02]])
    fig = corner.corner(samples[:,:3], cmap='gray', quantiles=[0.16,0.50,0.84], angle=0, plot_contours=True, plot_datapoints=False, smooth1d=True, labels=labels, show_titles=True, verbose=True, bins=50, title_fmt='.3f')
    
    plt.savefig('../tex/pdf_fulljeans{}.pdf'.format(suffix))

def h_prior():
    """Bobylev & Bajkova"""
    h = np.array([0.305, 0.311, 0.308, 0.3, 0.313, 0.309])
    herr = np.array([0.003, 0.003, 0.005, 0.002, 0.002, 0.001])
    w = herr**-0.5
    
    print(np.average(h, weights=w))
    print(np.average(herr, weights=w))
    

