from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table
import astropy.units as u
import corner

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
    print(xall)
    
    N = len(logg)
    plt.close()
    fig, ax = plt.subplots(2,3, figsize=(15,7), gridspec_kw = {'height_ratios':[5,2]}, sharex=True, squeeze=False)

    for i in range(2):
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
    
        plt.sca(ax[0][0])
        plt.plot(t['zeff'], t['nueff'], 'ko', alpha=a)
        plt.errorbar(t['zeff'], t['nueff'], yerr=t['nue'], fmt='none', color='k', alpha=a)
        plt.plot(tm['zeff'], tm['nueff'], 'ko')
        plt.errorbar(tm['zeff'], tm['nueff'], yerr=tm['nue'], fmt='none', color='k')
        plt.plot(z, nuzbest)
    
        plt.gca().set_yscale('log')
        plt.ylabel('$\\nu$ (kpc$^{-3}$)')

        plt.sca(ax[1][0])
        plt.axhline(0, color='r')
        plt.plot(tm['zeff'], tm['nueff']-nuzbest_, 'ko')
        plt.errorbar(tm['zeff'], tm['nueff']-nuzbest_, yerr=tm['nue'], fmt='none', color='k')
        plt.xlabel('Z (kpc)')
        plt.ylabel('$\Delta$ $\\nu$')
    
        plt.sca(ax[0][1])
        plt.plot(t['z'], t['sz'], 'ko', alpha=a)
        plt.errorbar(t['z'], t['sz'], yerr=t['sze'], fmt='none', color='k', alpha=a)
        plt.plot(tm['z'], tm['sz'], 'ko')
        plt.errorbar(tm['z'], tm['sz'], yerr=tm['sze'], fmt='none', color='k')
        plt.plot(z, szbest)
    
        plt.xlim(0,1.2)
        plt.ylim(0,50)
        plt.ylabel('$\sigma_{z}$ (km s$^{-1}$)')
    
        plt.sca(ax[1][1])
        plt.axhline(0, color='r')
        plt.plot(tm['zeff'], tm['sz']-szbest_, 'ko')
        plt.errorbar(tm['z'], tm['sz']-szbest_, yerr=tm['sze'], fmt='none', color='k')
        plt.xlabel('Z (kpc)')
        plt.ylabel('$\Delta$ $\sigma_z$')
    
        plt.sca(ax[0][2])
        plt.plot(t['z'], t['srz'], 'ko', alpha=a)
        plt.errorbar(t['z'], t['srz'], yerr=t['srze'], fmt='none', color='k', alpha=a)
        plt.plot(tm['z'], tm['srz'], 'ko')
        plt.errorbar(tm['z'], tm['srz'], yerr=tm['srze'], fmt='none', color='k')
        plt.plot(z, srzbest)

        plt.ylabel('$\sigma_{Rz}$ (km s$^{-1}$)')
        plt.ylim(-400,100)
    
        plt.sca(ax[1][2])
        plt.axhline(0, color='r')
        plt.plot(tm['zeff'], tm['srz']-srzbest_, 'ko')
        plt.errorbar(tm['z'], tm['srz']-srzbest_, yerr=tm['srze'], fmt='none', color='k')
        plt.xlabel('Z (kpc)')
        plt.ylabel('$\Delta$ $\sigma_{Rz}$')
    
    plt.tight_layout()
    plt.savefig('../tex/fulljeans_bestfit_t{}.pdf'.format(i))

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
    fig = corner.corner(samples[:,:3], cmap='gray', quantiles=[0.16,0.50,0.84], angle=0, plot_contours=True, plot_datapoints=False, smooth1d=True, labels=labels, show_titles=True, verbose=True, bins=50, title_fmt='.3f', range=[[7, 40], [0.09,0.4], [0, 0.02]])
    
    plt.savefig('../tex/pdf_fulljeans{}.pdf'.format(suffix))

def h_prior():
    """Bobylev & Bajkova"""
    h = np.array([0.305, 0.311, 0.308, 0.3, 0.313, 0.309])
    herr = np.array([0.003, 0.003, 0.005, 0.002, 0.002, 0.001])
    w = herr**-0.5
    
    print(np.average(h, weights=w))
    print(np.average(herr, weights=w))
    
    
