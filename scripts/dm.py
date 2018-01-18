from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy
from astropy.table import Table

import astropy.units as u
import astropy.coordinates as coord
import gala.coordinates as gc
from astropy.constants import G

import scipy.optimize
from scipy.special import gamma, hyp2f1
from scipy.integrate import quad
import gaia_tools.select
import healpy as hp
import mwdust

import imp
triangle = imp.load_source('triangle', '../../python/triangle.py/triangle.py')

import myutils
import multiprocessing
import emcee
import os.path
import time

import warnings
warnings.filterwarnings('ignore')

from os.path import expanduser
home = expanduser('~')

speclabels = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
logglabels = {0: 'dwarfs', 1: 'giants'}

class Sample():
    def __init__(self, name='distances_PJM2017.csv', crtheta=90, vdisk=220*u.km/u.s, sdisk=180*u.km/u.s, label='McMillan'):
        self.data = Table.read('{}/data/gaia/{}'.format(home, name))
        self.MJ = self.data['Jmag_2MASS'] - 5*np.log10(self.data['distance']) + 5
        self.JH = self.data['Jmag_2MASS'] - self.data['Hmag_2MASS']
        self.JK = self.data['Jmag_2MASS'] - self.data['Kmag_2MASS']
        
        # dwarf -- giant separation
        x = np.array([0.437, 0.359])
        y = np.array([2.6, 1.7])
        pf = np.polyfit(x, y, 1)
        self.poly = np.poly1d(pf)
        self.dwarf = (self.MJ>self.poly(self.JH)) | (self.MJ>2.5)
        self.giant = ~self.dwarf

        # spectral-type separation
        self.speclabels = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
        self.jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 0.79])
        #self.jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 1.])
        self.spectype = np.empty((len(self.jh_bins)-1, len(self.data)), dtype=bool)
        for i in range(np.size(self.jh_bins)-1):
            self.spectype[i] = (self.JH>=self.jh_bins[i]) & (self.JH<self.jh_bins[i+1])
        
        # detailed spectral type
        st = Table.read('../data/spectypes.txt', format='ascii.commented_header', fill_values=[('...', np.nan), ('....', np.nan), ('.....', np.nan)])
        distance = np.abs(self.JH[:,np.newaxis] - st['JH'])
        spec_id = np.argmin(distance, axis=1)
        self.spt = st['SpT'][spec_id]
        
        # Galactocentric coordinates
        c = coord.SkyCoord(ra=np.array(self.data['RAdeg'])*u.deg, dec=np.array(self.data['DEdeg'])*u.deg, distance=self.data['distance']*u.pc)
        cgal = c.transform_to(coord.Galactocentric)

        self.x = (np.transpose([cgal.x, cgal.y, cgal.z])*u.pc).to(u.kpc)
        self.v = np.transpose(gc.vhel_to_gal(c.icrs, rv=self.data['HRV']*u.km/u.s, pm=[np.array(self.data['pmRA_TGAS']), np.array(self.data['pmDE_TGAS'])]*u.mas/u.yr)).to(u.km/u.s)
    
        # Uncertainties
        if os.path.isfile('../data/uncertainties.npz'):
            d = np.load('../data/uncertainties.npz')
            self.xmed = d['xmed']
            self.xerr = d['xerr']
            self.vmed = d['vmed']
            self.verr = d['verr']
        else:
            self.xmed = self.x
            self.xerr = 0.01 * self.x
            self.vmed = self.v
            self.verr = 0.2 * self.v
    
        self.toomre(vdisk, sdisk)
        
        # stellar params edits
        self.data['age'] = self.data['age'] * 1e-9
        self.data['eage'] = self.data['eage'] * 1e-9
        self.data['Fe'] = np.genfromtxt(self.data['Fe'])
        self.data['Mg'] = np.genfromtxt(self.data['Mg'])
        
        # selection function
        # tgas
        fin = '../data/tgas_completeness.npz'
        if os.path.isfile(fin):
            self.cf_tgas = np.load(fin)['cf']
        else:
            self.cf_tgas = self.tgas_completeness()
        
        # rave
        fin = '../data/rave_completeness.npz'
        if os.path.isfile(fin):
            self.cf_rave = np.load(fin)['cf']
        else:
            self.cf_rave = self.rave_completeness()
        
        # joint
        self.cf = self.cf_tgas * self.cf_rave
    
    def tgas_completeness(self, verbose=False):
        """Determine completeness of TGAS--RAVE match, TGAS side, from Bovy+2017"""
        
        tsf = gaia_tools.select.tgasSelect()
        
        weight = np.zeros(len(self.data))
        for i in range(np.size(weight)):
            weight[i] = tsf(self.data['Jmag_2MASS'][i], self.data['Jmag_2MASS'][i] - self.data['Kmag_2MASS'][i], self.data['RAdeg'][i], self.data['DEdeg'][i])
            if verbose: print(i, weight[i])
        if verbose: print('{} out of {} stars have completeness determined'.format(np.sum(weight>0), np.size(weight)))
        
        np.savez('../data/tgas_completeness', cf=weight)
        
        return weight

    def rave_completeness(self):
        """Determine completeness of TGAS--RAVE match, RAVE side, from Wojno+2017"""
        J = self.data['Jmag_2MASS']
        K = self.data['Kmag_2MASS']
        I = J + J - K + 0.2*np.exp(5*(J - K - 1.2)) + 0.12
        
        # row
        nside = 32
        nested = True
        pix_id = hp.ang2pix(nside, np.radians(90 - self.data['DEdeg']), np.radians(self.data['RAdeg']), nest=nested)
        
        # column
        dmag = 0.1
        mag_min = 0
        mag_id = np.int64(np.floor((I - mag_min)/dmag))
        mag_string = np.array(['CF{:04.1f}'.format(x/10) for x in mag_id])
        
        t = Table.read('/home/ana/data/rave_cf.fits')
        
        Nstar = len(self.data)
        cf = np.empty(Nstar)
        for i in range(Nstar):
            cf[i] = t[pix_id[i]][mag_string[i]]
        
        np.savez('../data/rave_completeness', cf=cf)
        
        return cf
    
    def toomre(self, vdisk, sdisk):
        self.vxz = np.sqrt(self.v[:,0]**2 + self.v[:,2]**2)
        self.vy = self.v[:,1]
        
        self.halo = np.sqrt(self.vxz**2 + (self.vy - vdisk)**2)>sdisk
        self.disk = ~self.halo
        
        self.prograde = self.vy>0
        self.retrograde = ~self.prograde
    
def uncertainties(istart=0, iend=None):
    """Bootstrap (uncorrelated) observational uncertainties to get uncertainties in Cartesian positions"""
    
    s = Sample()
    N = len(s.data)
    np.random.seed(592)
    Nrand = 500
    if iend==None:
        iend = N
    
    Nbatch = iend - istart
    x = np.empty((Nbatch,3))*u.kpc
    xerr = np.empty((Nbatch,3))*u.kpc
    v = np.empty((Nbatch,3))*u.km/u.s
    verr = np.empty((Nbatch,3))*u.km/u.s
    
    t1 = time.time()
    for i, j in enumerate(range(istart, iend)):
        ra = np.random.randn(Nrand)*0.003/3600 + s.data['RAdeg'][j]
        dec = np.random.randn(Nrand)*0.003/3600 + s.data['DEdeg'][j]
        dist = np.random.randn(Nrand)*s.data['edistance'][j] + s.data['distance'][j]
        hrv = np.random.randn(Nrand)*s.data['eHRV'][j] + s.data['HRV'][j]
        pmra = np.random.randn(Nrand)*s.data['pmRA_error_TGAS'][j] + s.data['pmRA_TGAS'][j]
        pmdec = np.random.randn(Nrand)*s.data['pmDE_error_TGAS'][j] + s.data['pmDE_TGAS'][j]
        
        c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.pc)
        cgal = c.transform_to(coord.Galactocentric)

        x_ = (np.transpose([cgal.x, cgal.y, cgal.z])*u.pc).to(u.kpc)
        v_ = np.transpose(gc.vhel_to_gal(c.icrs, rv=hrv*u.km/u.s, pm=[pmra, pmdec]*u.mas/u.yr)).to(u.km/u.s)
        
        x[i] = np.median(x_, axis=0)
        xerr[i] = np.std(x_, axis=0)
        v[i] = np.median(v_, axis=0)
        verr[i] = np.std(v_, axis=0)
        
        #print(i, x[i] - s.x[j], xerr[i])
        #print(i, v[i] - s.v[i], verr[i])
    
    t2 = time.time()
    dt = t2 - t1
    print(dt, (dt*u.s*len(s.data)/Nbatch).to(u.h))
    
    np.savez('../data/uncertainties.{}'.format(istart), xmed=x.value, xerr=xerr.value, vmed=v.value, verr=verr.value)

def concatenate_uncertainties():
    """Merge individually saved files with Cartesian uncertainties"""
    istart = [0, 73000, 146000]
    
    for i, j in enumerate(istart):
        f = '../data/uncertainties.{}.npz'.format(j)
        d = np.load(f)
        if i==0:
            xmed = d['xmed']
            xerr = d['xerr']
            vmed = d['vmed']
            verr = d['verr']
        else:
            xmed = np.vstack([xmed, d['xmed']])
            xerr = np.vstack([xerr, d['xerr']])
            vmed = np.vstack([vmed, d['vmed']])
            verr = np.vstack([verr, d['verr']])
    
    np.savez('../data/uncertainties', xmed=xmed, xerr=xerr, vmed=vmed, verr=verr)

def edit_rave_cf():
    """Make RAVE DR5 completeness function easily readable"""
    t = Table.read('/home/ana/data/RAVE_completeness_PBP.csv')
    
    names = t.colnames
    Ncol = len(names)
    Nstar = len(t)
    dtype = ['f' for x in range(Ncol)]
    dtype[0] = 'i'
    
    tout = Table(np.array([t[names[0]]] + [np.zeros(Nstar) for x in range(Ncol-1)]).T, names=names, dtype=dtype)
    
    for c in names[1:]:
        nan = np.array(t[c])=='NULL'
        #print(c, nan)
        if np.sum(nan):
            tout[c][nan] = np.nan
        tout[c][~nan] = t[c][~nan]
    
    tout.pprint()
    tout.write('/home/ana/data/rave_cf.fits', overwrite=True)

def dust():
    """"""
    _BASE_NSIDE = 2**5
    lcen = 90
    bcen = 20
    dist = np.array([0.1, 0.2, 0.5])
    
    dmap3d= mwdust.Zero(filter='2MASS J')
    pixarea, aj= dmap3d.dust_vals_disk(lcen, bcen, dist, hp.pixelfunc.nside2resol(_BASE_NSIDE)/np.pi*180.)
    
    print(pixarea, aj, hp.pixelfunc.nside2pixarea(_BASE_NSIDE))


class Survey():
    def __init__(self, name, qrel=1, qrv=50, nonnegative=True, crtheta=90, observed=True, vdisk=220*u.km/u.s, sdisk=180*u.km/u.s, label=''):
        self.name = name
        self._data = Table.read('/home/ana/projects/mrich_halo/data/{}_abb.fits'.format(self.name))
        self.vdisk = vdisk
        self.sdisk = sdisk
        if len(label)==0:
            label = name
        self.label = label
        
        self.quality_cut(qrel=qrel, qrv=qrv, nonnegative=nonnegative, observed=observed)
        self.configuration_coords(observed=observed)
        self.phase_coords()
        self.define_counterrot(crtheta)
        self.toomre(vdisk, sdisk)
        
    def quality_cut(self, qrel=1, qrv=50, nonnegative=True, observed=True):
        if observed:
            quality = ((np.abs(self._data['pmra_error']/self._data['pmra'])<qrel) 
                & (np.abs(self._data['pmdec_error']/self._data['pmdec'])<qrel) 
                & (np.abs(self._data['hrv_error'])<qrv)
                & (np.abs(self._data['parallax_error']/self._data['parallax'])<2*qrel))
            if nonnegative:
                quality = quality & (self._data['parallax']>0) & np.isfinite(self._data['pmra'])
            self.data = self._data[quality]
        else:
            self.data = self._data
    
    def configuration_coords(self, observed=True):
        if observed:
            c = coord.SkyCoord(ra=np.array(self.data['ra'])*u.deg, dec=np.array(self.data['dec'])*u.deg, distance=1/np.array(self.data['parallax'])*u.kpc)
            cgal = c.transform_to(coord.Galactocentric) 

            self.x = np.transpose([cgal.x, cgal.y, cgal.z])*u.kpc
            self.v = np.transpose(gc.vhel_to_gal(c.icrs, rv=self.data['hrv']*u.km/u.s, pm=[np.array(self.data['pmra']), np.array(self.data['pmdec'])]*u.mas/u.yr))
        else:
            self.x = -self.data['x']*u.kpc
            self.v = self.data['v']*u.km/u.s
    
    def phase_coords(self):
        self.Ek = 0.5*np.linalg.norm(self.v, axis=1)**2
        self.vtot = np.linalg.norm(self.v, axis=1)

        self.L = np.cross(self.x, self.v, axis=1)
        self.L2 = np.linalg.norm(self.L, axis=1)
        self.Lperp = np.sqrt(self.L[:,0]**2 + self.L[:,1]**2)
        self.ltheta = np.degrees(np.arccos(self.L[:,2]/self.L2))
        self.lx = np.degrees(np.arccos(self.L[:,0]/self.L2))
        self.ly = np.degrees(np.arccos(self.L[:,1]/self.L2))
    
    def define_counterrot(self, crtheta):
        self.crtheta = crtheta
        self.counter_rotating = self.ltheta < self.crtheta
        self.rotating = self.ltheta > 180 - self.crtheta
    
    def toomre(self, vdisk, sdisk):
        self.vxz = np.sqrt(self.v[:,0]**2 + self.v[:,2]**2)
        self.vy = self.v[:,1]
        
        self.halo = np.sqrt(self.vxz**2 + (self.vy - vdisk)**2)>sdisk
        self.disk = ~self.halo

def hrd():
    """Plot the 2MASS HRD for TGAS--RAVE cross-match"""
    
    t = Table.read('/home/ana/data/gaia/distances_PJM2017.csv')
    t.pprint()
    print(t.colnames)

    MJ = t['Jmag_2MASS'] - 5*np.log10(t['distance']) + 5
    
    # bins
    jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 0.79])
    jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 1])
    speclabels = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    x = np.array([0.437, 0.359])
    y = np.array([2.6, 1.7])
    pf = np.polyfit(x, y, 1)
    poly = np.poly1d(pf)
    dwarf = (MJ>poly(t['Jmag_2MASS'] - t['Hmag_2MASS'])) | (MJ>2.5)
    giant = ~dwarf
    
    plt.close()
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.plot(t['Jmag_2MASS'] - t['Hmag_2MASS'], MJ, 'ko', ms=1, alpha=0.01, rasterized=True)
    plt.plot(x, y, 'r-')
    for c in jh_bins:
        plt.axvline(c, color='k', lw=2, alpha=0.2)
    
    plt.xlabel('J - H')
    plt.ylabel('$M_J$')
    plt.xlim(-0.5, 1.5)
    plt.ylim(8, -5)

    plt.sca(ax[1])
    plt.plot(t['Jmag_2MASS'][giant] - t['Hmag_2MASS'][giant], MJ[giant], 'ro', ms=1, rasterized=True, label='Giants')
    plt.plot(t['Jmag_2MASS'][dwarf] - t['Hmag_2MASS'][dwarf], MJ[dwarf], 'bo', ms=1, rasterized=True, label='Dwarfs')
    for i, c in enumerate(jh_bins):
        plt.axvline(c, color='k', lw=2, alpha=0.2)
        if i>0:
            plt.text(c-0.02, -4, speclabels[i-1], ha='right', fontsize='x-small')
    
    plt.legend(fontsize='small', markerscale=4, handlelength=0.4)
    plt.xlabel('J - H')
    plt.xlim(-0.5, 1.5)
    plt.ylim(8, -5)
    
    plt.tight_layout()
    plt.savefig('../plots/hrd_2mass.png')

def inventory():
    """"""
    s = Sample()
    for i, sl in enumerate(s.speclabels):
        print(sl, np.sum(s.spectype[i]), np.sum(s.spectype[i][s.giant]), np.sum(s.spectype[i][s.dwarf]))

def ages():
    """"""
    s = Sample()
    age_bins = np.linspace(0, 13.7, 10)
    
    plt.close()
    fig, ax = plt.subplots(2,3, figsize=(12,7))
    
    plt.sca(ax[0][0])
    plt.scatter(s.JH, s.MJ, c=s.data['age'], vmin=0, vmax=13.7)
    
    plt.xlabel('J - H')
    plt.ylabel('$M_J$')
    plt.xlim(-0.5, 1.5)
    plt.ylim(8, -5)
    
    plt.sca(ax[1][0])
    #finite = (s.data['Mg']!='NULL') & (s.data['Fe']!='NULL')
    finite = np.isfinite(s.data['Fe'])
    afinite = np.isfinite(s.data['Fe']) & np.isfinite(s.data['Mg'])
    colors = ['r', 'b']
    alphas = [0.01, 1]
    for i, c in enumerate([s.disk, s.halo]):
        plt.plot(s.data['age'][finite & c], s.data['Fe'][finite & c], 'o', ms=1, color=colors[i], rasterized=True, alpha=alphas[i])
    plt.plot(s.data['age'][finite & s.halo & s.prograde], s.data['Fe'][finite & s.halo & s.prograde], 'wo', ms=4, mec='b')
    plt.plot(s.data['age'][finite & s.halo & s.retrograde], s.data['Fe'][finite & s.halo & s.retrograde], 'bo', ms=4, mec='b')
    
    young = (s.data['age']<4) & (s.data['age']>0)
    plt.errorbar(s.data['age'][finite & s.halo & s.prograde & young], s.data['Fe'][finite & s.halo & s.prograde & young], xerr=s.data['age'][finite & s.halo & s.prograde & young], fmt='none', color='k')
    plt.errorbar(s.data['age'][finite & s.halo & s.retrograde & young], s.data['Fe'][finite & s.halo & s.retrograde & young], xerr=s.data['age'][finite & s.halo & s.retrograde & young], fmt='none', color='k')
    plt.xlim(13.7, 0)
    plt.ylim(-3, 1)
    plt.xlabel('Age (Gyr)')
    plt.ylabel('[Fe/H]')
    
    plt.sca(ax[0][1])
    plt.hist(s.data['age'], bins=age_bins, color='k', histtype='step', lw=2)
    plt.hist(s.data['age'][s.dwarf], bins=age_bins, histtype='step', lw=2)
    plt.hist(s.data['age'][s.giant], bins=age_bins, histtype='step', lw=2)
    plt.xlabel('Age (Gyr)')
    plt.ylabel('Number')
    
    plt.sca(ax[1][1])
    plt.hist(s.data['age'], bins=age_bins, color='k', histtype='step', lw=2, normed=True, label='All')
    plt.hist(s.data['age'][s.dwarf], bins=age_bins, histtype='step', lw=2, normed=True, label='Dwarfs')
    plt.hist(s.data['age'][s.giant], bins=age_bins, histtype='step', lw=2, normed=True, label='Giants')
    plt.xlabel('Age (Gyr)')
    plt.ylabel('Density (Gyr$^{-1}$)')
    plt.legend(fontsize='small')
    
    plt.sca(ax[0][2])
    plt.plot(s.data['age'][finite], s.data['Fe'][finite], 'ko', ms=1, alpha=0.01)
    plt.xlim(13.7, 0)
    plt.ylim(-3, 1)
    plt.xlabel('Age (Gyr)')
    plt.ylabel('[Fe/H]')
    
    plt.sca(ax[1][2])
    plt.plot(s.data['age'][afinite], s.data['Mg'][afinite].astype(np.float) - s.data['Fe'][afinite].astype(np.float), 'ko', ms=1, alpha=0.01)
    plt.xlim(13.7, 0)
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Age (Gyr)')
    plt.ylabel('[Mg/Fe]')

    plt.tight_layout()
    plt.savefig('../plots/feh_age.png')

def nuz():
    """"""
    s = Sample()
    sf = np.load('../data/completeness.npz')['sf']
    
    z_bins = np.arange(0, 4, 0.2)
    z = myutils.bincen(z_bins)
    z_bins_dwarf = np.arange(0, 2, 0.1)
    z_dwarf = myutils.bincen(z_bins_dwarf)
    z_bins_list = [z_bins_dwarf, z_bins]
    z_list = [z_dwarf, z]
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    ms = ['o', '^']
    #ms = ['-', ':']
    color = [mpl.cm.viridis(x/7) for x in range(7)]
    logglabels = ['dwarfs', 'giants']
    
    for i, logg in enumerate([s.dwarf, s.giant]):
        plt.sca(ax[i])
        for j, stype in enumerate(s.spectype):
            selection = logg & stype & (sf>0)
            if np.sum(selection)>1000:
                hz, be = np.histogram(np.abs(s.x[:,2][selection].value), bins=z_bins_list[i], weights=1/sf[selection])
                plt.plot(z_list[i], hz, ms[i], color=color[j], label='{} {}'.format(s.speclabels[j], logglabels[i]))
                #pf = np.polyfit(z[1:], np.log(hz[1:]), 1)
                #if np.all(np.isfinite(pf)): print(pf)
                #poly = np.poly1d(pf)
                #plt.plot(z, np.exp(poly(z)), 'k-')
                
                hz, be = np.histogram(np.abs(s.x[:,2][selection].value), bins=z_bins_list[i])
                plt.plot(z_list[i], hz, ms[i], color='w', mec=color[j], label=''.format(s.speclabels[j], logglabels[i]))
                #pf = np.polyfit(z[1:], np.log(hz[1:]), 1)
                #if np.all(np.isfinite(pf)): print(pf)
                #poly = np.poly1d(pf)
                #plt.plot(z, np.exp(poly(z)), 'k--')
    
    plt.sca(ax[0])
    plt.legend(fontsize='small')
    plt.gca().set_yscale('log')
    plt.xlabel('|Z| (kpc)')
    plt.ylabel('N(Z)')
    plt.xlim(0,2)
    plt.ylim(1,1e5)
    
    plt.sca(ax[1])
    plt.legend(fontsize='small')
    plt.gca().set_yscale('log')
    plt.xlabel('|Z| (kpc)')
    plt.ylabel('N(Z)')
    plt.xlim(0,4)
    plt.ylim(1,1e5)
    
    #plt.plot(z, 1e5*np.exp(-z/0.1), 'k-')
    #plt.plot(z, 1e5*np.exp(-z/0.13), 'k-')
    #plt.plot(z, 1e5*np.exp(-z/0.15), 'k-')
    
    plt.tight_layout()
    plt.savefig('../plots/nuz_spectype.png')

def population_info(logg='giants'):
    """"""
    s = Sample()
    age = 2
    old = s.data['age'] - s.data['eage']>age
    
    if logg=='giants':
        s_logg = s.giant
        s_spectype = [s.spectype[x] for x in [5,6]]
        z_bins = np.arange(0, 4, 0.2)
        zmin = 0.5
        zmax = 2
        szmax = 60
    elif logg=='dwarfs':
        s_logg = s.dwarf
        s_spectype = [s.spectype[x] for x in [3,4]]
        z_bins = np.arange(0,2,0.03)
        zmin=0.2
        zmax = 0.5
        szmax = 30
    elif logg=='age':
        s_logg = s.dwarf | s.giant
        s_spectype = [~old, old]
        z_bins = np.arange(0, 4, 0.1)
        zmin = 0.2
        zmax = 2
        szmax = 50
    
    z = myutils.bincen(z_bins)
    
    local = (np.sqrt((s.x[:,0] + 8.3*u.kpc)**2 + s.x[:,1]**2)<2*u.kpc)
    finite = np.isfinite(s.data['Fe'])
    Npop = len(s_spectype)
    
    feh_bins = np.linspace(-3,1,30)
    age_bins = np.linspace(0,14,30)
    
    colors = [mpl.cm.magma(x/3) for x in range(3)]
    
    d = 3
    Ncol = 5
    plt.close()
    fig, ax = plt.subplots(Npop, 5, figsize=(Ncol*d, Npop*d), sharex='col')
    
    for i in range(Npop):
        selection = finite & local & (s.cf>0) & s_logg & s_spectype[i]
        
        # HRD location
        plt.sca(ax[i][0])
        plt.plot(s.JH, s.MJ, 'ko', ms=1, alpha=0.01)
        plt.plot(s.JH[selection], s.MJ[selection], 'ro', ms=1, alpha=1)
        plt.xlim(-0.5, 1.5)
        plt.ylim(8, -5)
        
        plt.title('{} stars'.format(np.sum(selection)), fontsize='small')
        if i==Npop-1: plt.xlabel('J - H')
        plt.ylabel('$M_J$')
        
        # Age distribution
        plt.sca(ax[i][1])
        plt.hist(s.data['age'][selection], bins=age_bins, color='k', lw=2, histtype='step', normed=True)
        
        plt.xlim(0, 14)
        plt.title('{:.0f}% older than {:g} Gyr'.format(100*np.sum(old[selection])/np.size(old[selection]), age), fontsize='small')
        plt.ylabel('Density')
        if i==Npop-1: plt.xlabel('Age')
        
        # Metallicity distribution function
        plt.sca(ax[i][2])
        plt.hist(s.data['Fe'][selection], bins=feh_bins, color='k', lw=2, histtype='step', normed=True)
        
        fep = np.percentile(s.data['Fe'][selection], [33,66])
        fei = np.empty((3, np.sum(selection)), dtype=np.bool)
        fei[0] = s.data['Fe'][selection]<fep[0]
        fei[1] = (s.data['Fe'][selection]>=fep[0]) & (s.data['Fe'][selection]<fep[1])
        fei[2] = s.data['Fe'][selection]>=fep[1]
        
        plt.axvspan(-3, fep[0], color=colors[0], alpha=0.3)
        plt.axvspan(fep[0], fep[1], color=colors[1], alpha=0.3)
        plt.axvspan(fep[1], 1, color=colors[2], alpha=0.3)
        
        plt.xlim(-3, 1)
        plt.ylabel('Density')
        if i==Npop-1: plt.xlabel('[Fe/H]')
        
        # Density profile
        plt.sca(ax[i][3])
        h = np.zeros(3)
        for j, ind in enumerate(fei):
            hz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins, weights=1/s.cf[selection][ind])
            nz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
            nonzero = (nz>30) & (z>zmin)

            plt.plot(z[nonzero], hz[nonzero], 'o', color=colors[j])
            plt.errorbar(z, hz, yerr=np.sqrt(nz), fmt='none', color=colors[j])
            
            pf = np.polyfit(z[nonzero], np.log(hz[nonzero]), 1)
            poly = np.poly1d(pf)
            h[j] = 1/np.abs(pf[0])
            plt.plot(z, np.exp(poly(z)), '-', color=colors[j])
        
        plt.xlim(0, zmax)
        plt.ylim(1, 1e5)
        plt.ylabel('Number')
        if i==Npop-1: plt.xlabel('|Z| (kpc)')
        plt.gca().set_yscale('log')
        
        # Velocity dispersion profile
        plt.sca(ax[i][4])
        for j, ind in enumerate(fei):
            hz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins, weights=1/s.cf[selection][ind])
            nz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
            nonzero = (nz>30) & (z>zmin)
            idx  = np.digitize(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
            
            Nb = np.size(z)
            sz = np.empty(Nb)
            n = np.empty(Nb)
            sze = np.empty(Nb)
            for l in range(Nb):
                if np.sum(idx==l+1):
                    vmin, vmax = np.percentile(s.v[:,2][selection][ind][idx==l+1].value, [5,95])
                    i_ = (s.v[:,2][selection][ind][idx==l+1].value>vmin) & (s.v[:,2][selection][ind][idx==l+1].value<vmax)
                    sz[l] = np.std(s.v[:,2][selection][ind][idx==l+1][i_].value)
            
            plt.plot(z[nonzero], sz[nonzero], 'o', color=colors[j])
            plt.errorbar(z, sz, yerr=np.median(s.verr[:,2][selection][ind]), fmt='none', color=colors[j])
            
            sz_pred = sigmaz(h[j]*u.kpc, z*u.kpc, rhodm=0.01*u.Msun*u.pc**-3, sigs=40*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=0.2*u.kpc)
            plt.plot(z, sz_pred, '-', color=colors[j])
        
        plt.xlim(0, zmax)
        plt.ylim(0, szmax)
        plt.ylabel('$\sigma_Z$ (km s$^{-1}$)')
        if i==Npop-1: plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()
    plt.savefig('../plots/pop_properties_{:s}.png'.format(logg))

def mgiants():
    """"""
    s = Sample()
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(9,9))
    
    finite = np.isfinite(s.data['Fe'])
    selection = finite & s.giant & s.spectype[-1]
    #selection = finite & s.dwarf & s.spectype[5]
    print(np.sum(selection))
    print('[Fe/H] percentiles', np.percentile(s.data['Fe'][selection], [33,50,66]))
    print('age percentiles: ', np.percentile(s.data['age'][selection], [33,50,66]))
    
    fep = np.percentile(s.data['Fe'][selection], [33,66])
    fei = np.empty((3, np.sum(selection)), dtype=np.bool)
    fei[0] = s.data['Fe'][selection]<fep[0]
    fei[1] = (s.data['Fe'][selection]>=fep[0]) & (s.data['Fe'][selection]<fep[1])
    fei[2] = s.data['Fe'][selection]>=fep[1]
    
    plt.sca(ax[0][0])
    plt.hist(s.data['Fe'][selection])
    
    plt.sca(ax[0][1])
    plt.hist(s.data['age'][selection], bins=np.arange(0,14,0.5))
    
    plt.sca(ax[1][0])
    plt.plot(s.data['age'][selection], s.data['Fe'][selection], 'ko')
    plt.errorbar(s.data['age'][selection], s.data['Fe'][selection], xerr=s.data['eage'][selection], fmt='none')
    #print(np.median(s.data['eage'][selection]))
    print('1sigma older than 2 Gyr', np.sum(s.data['age'][selection] - s.data['eage'][selection]>2))
    
    plt.sca(ax[1][1])
    z_bins = np.arange(0, 4, 0.3)
    z = myutils.bincen(z_bins)
    for i, ind in enumerate(fei):
        hz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins, weights=s.cf[selection][ind])
        nz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        plt.plot(z, hz, 'o', color=mpl.cm.magma(i/3))
        plt.errorbar(z, hz, yerr=np.sqrt(nz), fmt='none', color=mpl.cm.magma(i/3))
        
        nonzero = (nz>0) & (z>0.5)
        pf = np.polyfit(z[nonzero], np.log(hz[nonzero]), 1)
        if np.all(np.isfinite(pf)): print(pf)
        poly = np.poly1d(pf)
        plt.plot(z, np.exp(poly(z)), '-', color=mpl.cm.magma(i/3))
        
    plt.gca().set_yscale('log')
    
    plt.tight_layout()

def mgiants_data():
    """"""
    s = Sample()
    finite = np.isfinite(s.data['Fe'])
    selection = finite & s.giant & s.spectype[-1] & (s.cf>0) & (s.data['age']-s.data['eage']>1) & (np.sqrt((s.x[:,0] + 8.3*u.kpc)**2 + s.x[:,1]**2)<2*u.kpc)
    #selection = finite & (s.cf>0) & (s.data['age']-s.data['eage']>5) & (np.sqrt((s.x[:,0] + 8.3*u.kpc)**2 + s.x[:,1]**2)<2*u.kpc)
    #selection = finite & s.dwarf & s.spectype[4] & (np.sqrt((s.x[:,0] + 8.3*u.kpc)**2 + s.x[:,1]**2)<2*u.kpc)
    print(np.sum(selection))

    fep = np.percentile(s.data['Fe'][selection], [33,66])
    fei = np.empty((3, np.sum(selection)), dtype=np.bool)
    fei[0] = s.data['Fe'][selection]<fep[0]
    fei[1] = (s.data['Fe'][selection]>=fep[0]) & (s.data['Fe'][selection]<fep[1])
    fei[2] = s.data['Fe'][selection]>=fep[1]
    
    z_bins = np.arange(0, 4, 0.15)
    z = myutils.bincen(z_bins)
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    plt.sca(ax[0])
    h = np.zeros(3)
    for i, ind in enumerate(fei):
        hz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins, weights=s.cf[selection][ind])
        nz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        nonzero = (nz>50) & (z>0.4)
        plt.plot(z[nonzero], hz[nonzero], 'o', color=mpl.cm.magma(i/3))
        plt.errorbar(z, hz, yerr=np.sqrt(nz), fmt='none', color=mpl.cm.magma(i/3))
        
        pf = np.polyfit(z[nonzero], np.log(hz[nonzero]), 1)
        h[i] = 1/np.abs(pf[0])
        if np.all(np.isfinite(pf)): print(pf, h[i])
        poly = np.poly1d(pf)
        plt.plot(z, np.exp(poly(z)), '-', color=mpl.cm.magma(i/3))

    plt.gca().set_yscale('log')
    plt.xlim(0, 2)
    plt.ylim(1, 1e4)
    
    plt.sca(ax[1])
    for i, ind in enumerate(fei):
        hz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins, weights=s.cf[selection][ind])
        nz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        nonzero = (nz>50) & (z>0.4)
        idx  = np.digitize(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        
        Nb = np.size(z)
        sz = np.empty(Nb)
        n = np.empty(Nb)
        sze = np.empty(Nb)
        for l in range(Nb):
            if np.sum(idx==l+1):
                vmin, vmax = np.percentile(s.v[:,2][selection][ind][idx==l+1].value, [0,100])
                #vmin, vmax = np.percentile(s.v[:,2][selection][ind][idx==l+1].value, [10,90])
                #vmin, vmax = np.percentile(s.v[:,2][selection][ind][idx==l+1].value, [20,80])
                i_ = (s.v[:,2][selection][ind][idx==l+1].value>vmin) & (s.v[:,2][selection][ind][idx==l+1].value<vmax)
                sz[l] = np.std(s.v[:,2][selection][ind][idx==l+1][i_].value)
        
        plt.plot(z[nonzero], sz[nonzero], 'o', color=mpl.cm.magma(i/3))
        plt.errorbar(z, sz, yerr=np.median(3*s.data['eHRV'][selection][ind]), fmt='none', color=mpl.cm.magma(i/3))
        
        sz_pred = sigmaz(h[i]*u.kpc, z*u.kpc, rhodm=0.015*u.Msun*u.pc**-3, sigs=40*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=0.2*u.kpc)
        plt.plot(z, sz_pred, '-', color=mpl.cm.magma(i/3))
    
    plt.xlim(0, 2)
    plt.ylim(0, 80)
    
    plt.tight_layout()

def sigmaz(h, z, sigs=41*u.Msun*u.pc**-2, H=0.2*u.kpc, sigg=13.2*u.Msun*u.pc**-2, rhodm=0.006*u.Msun*u.pc**-3):
    """Return a sigma_z profile for a population of scale height h, evaluated at distances z from the disk plane
    Parameters:
    h - tracer scale height (astropy unit)
    z - array-like of distances (astropy unit)"""
    
    sigma = np.sqrt(2*np.pi*G*h*(sigg + sigs*(1 - H/(h+H)*np.exp(-z/H))) + 4*np.pi*G*h*rhodm*(z+h)).to(u.km/u.s)
    
    return sigma


from astroML.density_estimation import XDGMM

def vz_xd(signed=False, dz=0.04, full=False, test=False):
    """Extreme deconvolution of Vz velocities"""
    
    s = Sample()
    
    if signed:
        z_bins = np.arange(-4, 4+dz, dz)
    else:
        z_bins = np.arange(0, 4+dz, dz)
        s.x[:,2] = np.abs(s.x[:,2])
    z = myutils.bincen(z_bins)
    Nb = np.size(z)

    Nrow = 5
    Ncol = np.int(np.ceil(Nb/Nrow))
    d = 5
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    if full:
        Npop = 1
    
    if test:
        Npop = 1
    
    ncomp = 10
    np.random.seed(4091)
    
    for i in range(Npop):
        plt.close()
        fig, ax = plt.subplots(Nrow,Ncol, figsize=(Ncol*d, Nrow*d), sharex=True, squeeze=False)
    
        if full:
            selection = (s.verr[:,2]<20)
        else:
            selection = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20)
        hz, be = np.histogram(s.x[:,2][selection].value, bins=z_bins, weights=s.cf[selection])
        nz, be = np.histogram(s.x[:,2][selection].value, bins=z_bins)
        idx  = np.digitize(s.x[:,2][selection].value, bins=z_bins)
        
        alpha = np.ones((Nb, ncomp)) * np.nan
        mu = np.ones((Nb, ncomp)) * np.nan
        var = np.ones((Nb, ncomp)) * np.nan
        
        for l in range(Nb):
            if np.sum(idx==l+1)>ncomp:
                vz = np.array([s.v[:,2][selection][idx==l+1].value]).T
                vze = s.verr[:,2][selection][idx==l+1][:,np.newaxis, np.newaxis]
                
                clf = XDGMM(ncomp, n_iter=100)
                clf.fit(vz, vze)
                
                mu[l] = clf.mu[:,0]
                var[l] = clf.V[:,0,0]
                alpha[l] = clf.alpha
                
                sig2 = np.sum(clf.V[:,0,0] * clf.alpha)
                
                print(l, z[l], np.sqrt(sig2), np.std(vz), np.median(np.abs(vz - np.median(vz))))
                
                plt.sca(ax[int(l/Ncol)][l%Ncol])
                v_ = np.arange(-100, 100, 15)
                vc_ = myutils.bincen(v_)
                nv, b_ = np.histogram(vz, bins=v_)
                poisson = nv**-0.5
                plt.fill_between(vc_, poisson, -poisson, color='k', alpha=0.2)

                hv, b_ = np.histogram(vz, bins=v_, normed=True)
                hv_gmm = np.empty(np.size(vc_))
                for c in range(ncomp):
                    hv_gmm += clf.alpha[c] * np.exp(-0.5*(vc_ - clf.mu[c,0])**2/clf.V[c,0,0]) / (np.sqrt(2*np.pi) * clf.V[c,0,0])
                
                res = 1 - hv_gmm/hv
                plt.plot(vc_, res, 'ko')
                
                #plt.ylim(-1,1)
                plt.xlim(-100,100)
        
        if full:
            np.savez('../data/vz_xd{}_full_z{}_s{:1d}'.format(ncomp, l, signed), mu=mu, var=var, alpha=alpha)
        else:
            np.savez('../data/vz_xd{}_logg{}_teff{}_z{}_s{:1d}'.format(ncomp, logg_id[i], teff[i], l, signed), mu=mu, var=var, alpha=alpha)
    
        plt.tight_layout()
        if full:
            plt.savefig('../plots/vz_xd{}_full_z{}_s{:1d}.png'.format(ncomp, l, signed))
        else:
            plt.savefig('../plots/vz_xd{}_logg{}_teff{}_z{}_s{:1d}.png'.format(ncomp, logg_id[i], teff[i], l, signed))

def vzvr_xd(test=True, dz=0.04, signed=False, off=0, ncomp=10):
    """Extreme deconvolution of VzVR velocities"""
    
    s = Sample()
    
    if signed:
        z_bins = np.arange(-4, 4+dz, dz)
    else:
        z_bins = np.arange(0, 4+dz, dz)
        s.x[:,2] = np.abs(s.x[:,2])
    
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    Nrow = 5
    Ncol = np.int(np.ceil(Nb/Nrow))
    d = 5
    
    logg = [s.giant, s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [1, 0, 0, 0, 1, 1]
    teff = [6, 2, 3, 4, 5, 6]
    Npop = len(teff)
    
    if test:
        Npop = 1
        #Nb = 1
    
    np.random.seed(4091)
    
    for i in range(Npop):
        plt.close()
        #fig, ax = plt.subplots(Nrow,Ncol, figsize=(Ncol*d, Nrow*d), sharex=True, squeeze=False)
        plt.figure(figsize=(8,6))
    
        selection = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20)
        hz, be = np.histogram(s.x[:,2][selection].value, bins=z_bins, weights=s.cf[selection])
        nz, be = np.histogram(s.x[:,2][selection].value, bins=z_bins)
        idx  = np.digitize(s.x[:,2][selection].value, bins=z_bins)
        
        alpha = np.ones((Nb, ncomp)) * np.nan
        mu = np.ones((Nb, ncomp)) * np.nan
        var = np.ones((Nb, ncomp)) * np.nan
        
        mvr = np.ones(Nb) * np.nan
        mvz = np.ones(Nb) * np.nan
        mvrz = np.ones(Nb) * np.nan
        svrz = np.ones(Nb) * np.nan
        
        for l in range(Nb):
            if np.sum(idx==l+1)>ncomp:
                vz = np.array([s.v[:,2][selection][idx==l+1].value]).T
                vze = s.verr[:,2][selection][idx==l+1][:,np.newaxis, np.newaxis]
                
                vx = np.array([s.v[:,0][selection][idx==l+1].value])
                vy = np.array([s.v[:,1][selection][idx==l+1].value])
                thx = np.arctan2(s.x[:,1][selection][idx==l+1].value, s.x[:,0][selection][idx==l+1].value)
                thv = np.arctan2(s.v[:,1][selection][idx==l+1].value, s.v[:,0][selection][idx==l+1].value)
                vr = (np.sqrt(vx**2 + vy**2) * np.cos(thx+thv)).T
                
                vxe = s.verr[:,0][selection][idx==l+1]
                vye = s.verr[:,1][selection][idx==l+1]
                vre = np.sqrt((vx[:,0]*vxe/vr[:,0])**2 + (vy[:,0]*vye/vr[:,0])**2)[:,np.newaxis, np.newaxis]
                
                #print(np.shape(vz), np.shape(vze), np.shape(vr), np.shape(vre))
                
                vrz = vr*vz
                vrze = np.sqrt((vr[:,0]*vze[:,0,0])**2 + (vz[:,0]*vre[:,0,0])**2)[:,np.newaxis, np.newaxis]
                #print(np.median(vr), np.median(vz), np.median(vrz))
                
                #med = np.median(vrz)
                mvr[l] = np.median(vr)
                mvz[l] = np.median(vz)
                fsig = (vr-mvr)*(vz-mvz)
                mvrz[l] = np.nanmean(fsig)
                svrz[l] = np.nanstd(fsig)
                svrz[l] = np.median(vr*vz)
                #print(l, mvrz[l], svrz[l])
                #print(i, l, med, np.std(vrz), np.median(vr), np.median(vz), mvrz, svrz)
                
                #clf = XDGMM(ncomp, n_iter=100)
                #clf.fit(vrz, vrze)
                
                #mu[l] = clf.mu[:,0]
                #var[l] = clf.V[:,0,0]
                #alpha[l] = clf.alpha
                
                #med = np.sum(mu[l] * alpha[l])
                #print(i, l, med*2/8.3)
                
                #sig2 = np.sum(clf.V[:,0,0] * clf.alpha)
                #print(l, np.sqrt(sig2), np.std(vrz), np.median(np.abs(vrz - np.median(vrz))))
                
                ##plt.sca(ax[int(l/Ncol)][l%Ncol])
                ##plt.hist(vr*vz, bins=np.linspace(-3000,3000,30))
                ##plt.axvline(med, color='k', lw=2)
                
                #v_ = np.arange(-100, 100, 15)
                #vc_ = myutils.bincen(v_)
                #nv, b_ = np.histogram(vz, bins=v_)
                #poisson = nv**-0.5
                #plt.fill_between(vc_, poisson, -poisson, color='k', alpha=0.2)

                #hv, b_ = np.histogram(vz, bins=v_, normed=True)
                #hv_gmm = np.empty(np.size(vc_))
                #for c in range(ncomp):
                    #hv_gmm += clf.alpha[c] * np.exp(-0.5*(vc_ - clf.mu[c,0])**2/clf.V[c,0,0]) / (np.sqrt(2*np.pi) * clf.V[c,0,0])
                
                #res = 1 - hv_gmm/hv
                #plt.plot(vc_, res, 'ko')
                
                ##plt.ylim(-1,1)
                #plt.xlim(-100,100)
        
        plt.plot(z, mvz, 'r-', label='$<V_{Z}>$')
        plt.plot(z, mvr, 'b-', label='$<V_{R}>$')
        #plt.plot(z, mvrz, 'm-', label='$<V_{RZ}>$')
        plt.plot(z, svrz, 'm-', label='$<V_{R}V_{Z}>$')
        
        sig = svrz - mvz*mvr
        plt.plot(z, sig, 'k-', label='$<V_{R}V_{Z}>$ - $<V_{R}>$$<V_{Z}>$')
        
        plt.xlim(0,2)
        plt.ylim(-300,300)
        
        plt.legend(fontsize='medium')
        plt.xlabel('Z (kpc)')
        plt.ylabel('Velocity$^2$ (km/s)$^2$')
        #np.savez('../data/vrz_xd{}_logg{}_teff{}_z{}_s{:1d}'.format(ncomp, logg_id[i], teff[i], l, signed), mu=mu, var=var, alpha=alpha)
    
        plt.tight_layout()
        plt.savefig('../plots/vrz_xd{}_logg{}_teff{}_z{}_s{:1d}.png'.format(ncomp, logg_id[i], teff[i],l, signed), dpi=200)


def dataset_populations_xd(Nboot=100, ncomp=10, dz=0.04, signed=False, test=False):
    """"""
    
    s = Sample()
    sfok = s.cf>0
    
    if signed:
        z_bins = np.arange(-4, 4+dz, dz)
    else:
        z_bins = np.arange(0, 4+dz, dz)
        s.x[:,2] = np.abs(s.x[:,2])

    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    Nb_aux = np.size(z_bins) - 2
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    if test:
        Npop = 1
    
    for i in range(Npop):
        selection = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20) #& (s.data['Fe']>-0.5)#& sfok
        #if i>1:
            #finite = np.isfinite(s.data['Fe'])
            #selection = selection & finite & (s.data['Fe']<-0.7)
        
        hz, be = np.histogram(s.x[:,2][selection & sfok].value, bins=z_bins, weights=1/(s.cf[selection & sfok]))
        nz, be = np.histogram(s.x[:,2][selection].value, bins=z_bins)
        idx  = np.digitize(s.x[:,2][selection].value, bins=z_bins)
        #idx  = np.digitize(-s.x[:,2][selection].value, bins=z_bins)
        
        n = np.zeros(Nb)
        vz = np.ones(Nb)*np.nan
        sz = np.ones(Nb)*np.nan
        vze = np.ones(Nb)*np.nan
        sze = np.ones(Nb)*np.nan
        vza = np.empty(Nb)
        vzae = np.empty(Nb)
        sza = np.empty(Nb)
        szae = np.empty(Nb)
        feh = np.empty(Nb)
        age = np.empty(Nb)
        neff = np.zeros(Nb)
        zeff = np.zeros(Nb)
        vrz = np.ones(Nb)*np.nan
        vrze = np.ones(Nb)*np.nan
        srz = np.ones(Nb)*np.nan
        srze = np.ones(Nb)*np.nan

        d = np.load('../data/vz_xd{}_logg{}_teff{}_z{}_s{:1d}.npz'.format(ncomp, logg_id[i], teff[i], Nb-1, signed))
        drz = np.load('../data/vrz_xd{}_logg{}_teff{}_z{}_s{:1d}.npz'.format(ncomp, logg_id[i], teff[i], Nb-1, signed))
        #if signed:
            #d = np.load('../data/vz_xd{}_logg{}_teff{}_z{}_s{:1d}.npz'.format(ncomp, logg_id[i], teff[i], Nb-1, signed))
            #drz = np.load('../data/vrz_xd{}_logg{}_teff{}_z{}_s{:1d}.npz'.format(ncomp, logg_id[i], teff[i], Nb-1, signed))
        #else:
            #d = np.load('../data/vz_xd{}_logg{}_teff{}_z{}.npz'.format(ncomp, logg_id[i], teff[i], Nb-2))
            #drz = np.load('../data/vrz_xd{}_logg{}_teff{}_z{}.npz'.format(ncomp, logg_id[i], teff[i], Nb-2))
        #deff = np.load('../data/veff_logg{}_spt{}_z{}.npz'.format(logg_id[i], teff[i], Nb_aux))
        
        tgas = Table.read('../data/nu_tgas_logg{}_teff{}_z{}_s{:1d}.fits'.format(logg_id[i], teff[i], Nb_aux, signed))
        
        for l in range(Nb-1):
            if np.sum(idx==l+1):
                # extreme deconvolution values
                if np.all(np.isfinite(d['alpha'][l])):
                    vz[l] = np.sum(d['alpha'][l] * d['mu'][l])
                    vze[l] = np.abs(vz[l] / np.sqrt(nz[l]))
                    sz[l] = np.sqrt(np.sum(d['alpha'][l] * d['var'][l]))
                    sze[l] = sz[l] / np.sqrt(nz[l])
                    vrz[l] = np.sum(drz['alpha'][l] * drz['mu'][l])
                    vrze[l] = np.abs(vrz[l] / np.sqrt(nz[l]))
                    srz[l] = (np.sum(drz['alpha'][l] * drz['var'][l]))**0.25
                    srze[l] = srz[l] / np.sqrt(nz[l])
                
                # bootstrap
                Nstar = np.size(s.v[:,2][selection][idx==l+1])
                vzboot = s.v[:,2][selection][idx==l+1][:,np.newaxis].value + s.verr[:,2][selection][idx==l+1][:,np.newaxis]*np.random.randn(Nstar, Nboot)
                vza[l] = np.median(vzboot)
                vzae[l] = 0.5 * (np.percentile(vzboot, [84]) - np.percentile(vzboot, [16]))
                
                szboot = np.sqrt(np.median(np.abs(vzboot - np.median(vzboot, axis=0)), axis=0)**2 - np.median(vzboot, axis=0)**2)
                sza[l] = np.median(szboot)
                szae[l] = 0.5 * (np.percentile(szboot, [84]) - np.percentile(szboot, [16]))
                feh[l] = np.nanmedian(s.data['Fe'][selection][idx==l+1])
                age[l] = np.nanmedian(s.data['age'][selection][idx==l+1])
                
                #neff[l] = deff['n'][Nb-l] / deff['veff'][Nb-l]
                #zeff[l] = np.abs(deff['zcen'][Nb-l])
                
                nz[l] = tgas['n'][l]
                neff[l] = tgas['nu'][l]
                zeff[l] = tgas['z'][l]
        
        t = Table([z, hz, nz, vz, vze, sz, sze, vrz, vrze, srz, srze, feh, age, neff, zeff], names=('z', 'nu', 'n', 'vz', 'vze', 'sz', 'sze', 'vrz', 'vrze', 'srz', 'srze', 'feh', 'age', 'nueff', 'zeff'))
        t.write('../data/profile_xd{}_logg{}_teff{}_z{}_s{:1d}.fits'.format(ncomp, logg_id[i], teff[i], Nb_aux, signed), overwrite=True)
        t.pprint()

def veff_xd(teff=2, logg=0, dz=0.04):
    """Get effective volume in xd framework"""
    
    s = Sample()
    sfok = s.cf>0

    # grid distance
    d_bins = np.linspace(0,4,200)
    delta_d = d_bins[1] - d_bins[0]
    d_cen = myutils.bincen(d_bins)
    Nd = np.size(d_bins) - 1
    
    # grid sky
    nside = 8
    nested = True
    pix_id = hp.ang2pix(nside, np.radians(90 - s.data['DEdeg']), np.radians(s.data['RAdeg']), nest=nested)
    pixels = np.unique(pix_id)
    Npix = np.size(pixels)
    theta, phi = hp.pix2ang(nside, pixels, nest=nested)
    ra = phi
    dec = 0.5*np.pi - theta
    
    z_bins = np.arange(-4, 4+dz, dz)
    z_cen = myutils.bincen(z_bins)
    Nz = np.size(z_bins) - 1
    l = np.int64(np.size(z_bins)/2 - 2)
    
    R_bins = np.array([-1,1])
    Nr = np.size(R_bins) - 1
    V = (z_bins[1:] - z_bins[:-1]) * 1**2 * np.pi
    Veff = (z_bins[1:] - z_bins[:-1]) * 0
    N = np.zeros(Nz)
    Veff = np.zeros(Nz)
    
    d_cen_4d = np.tile(d_cen, Npix).reshape(Npix, Nd).T.flatten()
    ra_4d = np.tile(ra, Nd)
    dec_4d = np.tile(dec, Nd)
    pix_4d = np.tile(pixels, Nd)

    c = coord.SkyCoord(ra=ra_4d*u.rad, dec=dec_4d*u.rad, distance=d_cen_4d*u.kpc)
    cgal = c.transform_to(coord.Galactocentric)
    
    inR = (np.sqrt((cgal.x+8.3*u.kpc)**2 + (cgal.y)**2)<1*u.kpc)
    
    Veff_id = myutils.wherein(z_bins, cgal.z.to(u.kpc).value) - 1
    Veff_filtered = np.float64(np.copy(Veff_id))
    Veff_filtered[~inR] = np.nan
    Veffs = np.int64(np.unique(Veff_filtered[np.isfinite(Veff_filtered)]))
    
    pop = {0: s.dwarf, 1: s.giant}

    # select population
    sel_pop = pop[logg] & s.spectype[teff] & (s.verr[:,2]<20) & (np.sqrt((s.x[:,0].value+8.3)**2 + s.x[:,1].value**2)<1)
    
    # volume elements
    for v in Veffs:
        sel_veff = Veff_id==v
        pixels_veff = pix_4d[sel_veff]
        distance_veff = d_cen_4d[sel_veff]
        print(v, np.shape(pixels_veff))
    
        # grid sky
        sel_pix = np.zeros(np.size(pix_id), dtype=bool)
        for ip, p in enumerate(pixels_veff):
            sel_pix = sel_pix | (pix_id==p)
        
        # grid distance
        for d in distance_veff:
            sel_d = (s.data['distance']*1e-3>=d - delta_d) & (s.data['distance']*1e-3<d + delta_d)
            
            selection = sel_pix & sel_d & sel_pop
            #Veff[v] += np.nanmean(s.cf_tgas[selection]) * d**2
            veff_ = np.nanmean(s.cf_rave[selection])
            if np.isfinite(veff_):
                Veff[v] += veff_ * d**2
            N[v] += np.sum(selection)
    
    Veff *= (z_bins[1:] - z_bins[:-1]) * 1**2 * np.pi
    np.savez('../data/veff_logg{}_spt{}_z{}'.format(logg, teff, l), n=N, veff=Veff, zcen=z_cen)
    
    print(z_cen, V, Veff, N)
    
    plt.close()
    plt.figure()
    
    plt.plot(z_cen, N/V, 'ko')
    plt.plot(z_cen, N/Veff, 'wo', mec='k')
    plt.gca().set_yscale('log')
    
    plt.xlabel('Z (kpc)')
    plt.ylabel('Density (kpc$^{-3}$)')
    plt.xlim(-4,4)
    
    plt.tight_layout()
    plt.savefig('../plots/veff_logg{}_spt{}.png'.format(logg, teff))

def nu_tgas(teff=2, logg=0, dz=0.04, signed=False):
    """"""
    tgas = Table.read('/home/ana/data/gaia/tgas-source.fits')
    tgas_2mass = Table.read('/home/ana/data/gaia_tools/Gaia/dstn_match/tgas-matched-2mass.fits.gz')

    JH = tgas_2mass['j_mag'] - tgas_2mass['h_mag']
    jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 0.79])
    spectype = np.empty((len(jh_bins)-1, len(tgas_2mass)), dtype=bool)
    for i in range(np.size(jh_bins)-1):
        spectype[i] = (JH>=jh_bins[i]) & (JH<jh_bins[i+1])
    
    # dwarf -- giant separation
    MJ = tgas_2mass['j_mag'] - 5*np.log10(1000./tgas['parallax']) + 5
    x = np.array([0.437, 0.359])
    y = np.array([2.6, 1.7])
    pf = np.polyfit(x, y, 1)
    poly = np.poly1d(pf)
    dwarf = (MJ>poly(JH)) | (MJ>2.5)
    giant = ~dwarf
    loggid = {0: dwarf, 1: giant}
    
    a = spectype[teff] & (tgas_2mass['matched']==True) & (tgas_2mass['j_mag']>0) & (tgas_2mass['j_cmsig']>0) & loggid[logg] & np.isfinite(MJ)
    tgas_2mass = tgas_2mass[a]    
    tgas = tgas[a]
    MJ = MJ[a]
    
    c = coord.SkyCoord(ra=np.array(tgas['ra'])*u.deg, dec=np.array(tgas['dec'])*u.deg, distance=1./tgas['parallax']*u.kpc)
    cgal = c.transform_to(coord.Galactocentric)
    x = (np.transpose([cgal.x, cgal.y, cgal.z])*u.kpc).to(u.kpc)
    
    if signed:
        z_bins = np.arange(-4, 4+dz, dz)
        fvol = cyl_vol_func
        label = 'Z'
        xlims = [-2,2]
    else:
        z_bins = np.arange(0, 4+dz, dz)
        fvol = abscyl_vol_func
        x[:,2] = np.abs(x[:,2])
        label = '|Z|'
        xlims = [0,2]
    
    z_cen = myutils.bincen(z_bins)
    l = np.size(z_bins) - 2

    tsf = gaia_tools.select.tgasSelect()
    tesf = gaia_tools.select.tgasEffectiveSelect(tsf, dmap3d=mwdust.Zero(), MJ=MJ, JK=tgas_2mass['j_mag']-tgas_2mass['k_mag'])
    statistical = tsf.determine_statistical(tgas, tgas_2mass['j_mag'], tgas_2mass['k_mag'])
    
    # calculate effective volume
    veff = np.zeros_like(z_cen)
    dxy = 1
    for i in range(np.size(z_cen)):
        veff[i] = tesf.volume(lambda x_, y_, z_: fvol(x_, y_, z_, xymax=dxy, zmin=z_bins[i], zmax=z_bins[i+1]), ndists=101, xyz=True, relative=True)
    
    # calculate density
    close = np.sqrt((x[:,0].value + 8.3)**2 + (x[:,1].value)**2) < dxy
    toinclude = statistical & close
    nz, be = np.histogram(x[:,2].value, bins=z_bins, weights=np.float64(toinclude))
    nu_raw = nz/(np.pi*dxy**2*(z_bins[1]-z_bins[0]))
    nu = nz/(veff*np.pi*dxy**2*(z_bins[1]-z_bins[0]))
    
    # save profile
    tout = Table(np.array([z_cen, nu, nu_raw, nz, veff]).T, names=('z', 'nu', 'nu_raw', 'n', 'veff'))
    tout.write('../data/nu_tgas_logg{}_teff{}_z{}_s{:1d}.fits'.format(logg, teff, l, signed), overwrite=True)
    
    ## fit sech^2 profile
    #p0 = [1e5, 0.1]
    #finite = np.isfinite(nu)
    #popt_raw, pcov_raw = scipy.optimize.curve_fit(sech_profile, z_cen[finite], nu_raw[finite], p0=p0)
    #print(popt_raw)
    #popt, pcov = scipy.optimize.curve_fit(sech_profile, z_cen[finite], nu[finite], p0=p0)
    #print(popt)
    
    # plot
    plt.close()
    plt.figure(figsize=(8,6))

    plt.plot(z_cen, nu_raw, 'wo', mec='k', label='Raw')
    plt.plot(z_cen, nu, 'ko', label='Corrected')
    #plt.plot(z_cen, sech_profile(z_cen, *popt), label='')
    #plt.plot(z_cen, sech_profile(z_cen, *popt_raw), label='')
    
    plt.title('{:s} {:s}'.format(speclabels[teff], logglabels[logg]))
    plt.xlabel('{} (kpc)'.format(label))
    plt.ylabel('$\\nu$ (kpc$^{-3}$)')
    plt.legend(fontsize='medium')
    
    plt.xlim(*xlims)
    plt.gca().set_yscale('log')
    plt.gca().set_ylim(bottom=1)
    
    plt.tight_layout()
    plt.savefig('../plots/nu_tgas_logg{}_teff{}_z{}_s{:1d}.png'.format(logg, teff, l, signed))


# vrvz variations
def vrvz_gradient(logg=1, teff=6, dz=0.1, signed=1):
    """"""
    
    if signed:
        z_bins = np.arange(-4, 4+dz, dz)
    else:
        z_bins = np.arange(0, 4+dz, dz)

    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    Nb_aux = np.size(z_bins) - 2
    
    x = np.array([-7.5,-7.9,-8.3,-8.7,-9.1])
    Nx = np.size(x)
    
    vrz = np.ones((Nx, Nb))*np.nan
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    for i, x0 in enumerate(x):
        fname = '../data/vrz_xd10_logg{}_teff{}_x0{:.1f} kpc_z{}_s{:d}.npz'.format(logg, teff, x0, Nb_aux, signed)
        if os.path.isfile(fname):
            drz = np.load(fname)

            for l in range(Nb):
                vrz[i][l] = np.sum(drz['alpha'][l] * drz['mu'][l])
    
    Ntot = np.sum(np.any(np.isfinite(vrz), axis=0))

    xvec = np.empty(0)
    zvec = np.empty(0)
    vrzvec = np.empty(0)
    for l in range(Nb):
        if np.any(np.isfinite(vrz[:,l])):
            c = mpl.cm.viridis(l/Nb)
            if z[l]>0:
                c = 'g'
            else:
                c = 'b'
            plt.plot(np.abs(x), vrz[:,l], 'o', color=c, ms=10, lw=2) #, label='z = {:.2f} kpc'.format(z[l]))
            
            finite = np.isfinite(vrz[:,l])
            xvec = np.concatenate((xvec, x[finite]))
            zvec = np.concatenate((zvec, np.repeat(z[l], np.sum(finite))))
            vrzvec = np.concatenate((vrzvec, vrz[:,l][finite]))
    
    cols = ['g', 'b']
    labels = ['N', 'S']
    if signed:
        above = zvec>0
        for e, sub in enumerate([above, ~above]):
            p = np.polyfit(np.abs(xvec[sub]), vrzvec[sub], 1)
            print(p)
            
            poly = np.poly1d(p)
            plt.plot(np.abs(x), poly(np.abs(x)), '-', c=cols[e], label='{}: {:.0f}'.format(labels[e], p[0]) + ' km$^2$ s$^{-2}$ kpc$^{-1}$', lw=2)
    else:
        p = np.polyfit(np.abs(xvec), vrzvec, 1)
        print(p)
    
    plt.legend(fontsize='small', ncol=1)
    plt.xlim(7.3,9.3)
    plt.xlabel('R (kpc)')
    plt.ylabel('$V_RV_Z$ (km s$^{-1}$)$^2$')
    
    plt.savefig('../plots/vrvz_gradient_logg{}_teff{}_z{}_s{:d}.png'.format(logg, teff, Nb_aux, signed))

def dataset_populations_xd_rvrz(Nboot=100, ncomp=10, dz=0.04, signed=False, test=False):
    """"""
    s = Sample()
    
    if signed:
        z_bins = np.arange(-4, 4+dz, dz)
    else:
        z_bins = np.arange(0, 4+dz, dz)

    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    Nb_aux = np.size(z_bins) - 2
    l = np.argmin(np.abs(z))
    
    x = np.array([-7.5,-7.9,-8.3,-8.7,-9.1])
    Nx = np.size(x)
    dr = 0.2*u.kpc
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    if test:
        Npop = 1
    
    for i in range(Npop):
        vrz = np.ones(Nx)*np.nan
        vrze = np.ones(Nx)*np.nan
        
        for j, x0 in enumerate(x):
            fname = '../data/vrz_xd10_logg{}_teff{}_x0{:.1f} kpc_z{}_s{:d}.npz'.format(logg_id[i], teff[i], x0, Nb_aux, signed)
            if os.path.isfile(fname):
                drz = np.load(fname)
                vrz[j] = np.sum(drz['alpha'][l] * drz['mu'][l])
                
                selection = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20) & ((s.x[:,0] - x0*u.kpc)**2 + s.x[:,1]**2<dr**2)
                n = np.sum(selection)
                vrze[j] = np.abs(vrz[j] / np.sqrt(n))
        
        t = Table([np.abs(x), vrz, vrze], names=('R', 'dvrz', 'dvrze'))
        t.write('../data/rvrz_xd{}_logg{}_teff{}_z{}_s{:1d}.fits'.format(ncomp, logg_id[i], teff[i], Nb_aux, signed), overwrite=True)
        t.pprint()

def nu_tgas_fit(teff=2, logg=0):
    """"""
    t = Table.read('../data/nu_tgas_logg{}_teff{}.fits'.format(logg, teff))
    
    # fit sech^2 profile
    p0 = [1e5, 0.1]
    finite = np.isfinite(t['nu'])
    popt_raw, pcov_raw = scipy.optimize.curve_fit(sech_profile, t['z'][finite], t['nu_raw'][finite], p0=p0)
    popt, pcov = scipy.optimize.curve_fit(sech_profile, t['z'][finite], t['nu'][finite], p0=p0)
    
    # fit dual sech^2 profile
    p0_duo = [1e5, 0.1, 0.1, 1]
    popt_raw_duo, pcov_raw_duo = scipy.optimize.curve_fit(twosech_profile, t['z'][finite], t['nu_raw'][finite], p0=p0_duo)
    popt_duo, pcov_duo = scipy.optimize.curve_fit(twosech_profile, t['z'][finite], t['nu'][finite], p0=p0_duo)
    #print(popt_duo)
    
    # plot
    plt.close()
    plt.figure(figsize=(8,6))

    plt.plot(t['z'], t['nu_raw'], 'wo', mec='k', label='Raw')
    plt.plot(t['z'], sech_profile(t['z'], *popt_raw), 'b-', label='sech$^2$, $h_1$={:.0f}'.format(np.abs(popt_raw[-1])*1e3))
    plt.plot(t['z'], twosech_profile(t['z'], *popt_raw_duo), 'b--', label='Dual sech$^2$, $h_1$={:.0f} pc, $h_2$={:.0f} pc'.format(np.abs(popt_raw_duo[-2])*1e3, np.abs(popt_raw_duo[-1])*1e3))
    
    plt.plot(t['z'], t['nu'], 'ko', label='Corrected')
    plt.plot(t['z'], sech_profile(t['z'], *popt), 'g-', label='sech$^2$, $h_1$={:.0f}'.format(np.abs(popt[-1])*1e3))
    plt.plot(t['z'], twosech_profile(t['z'], *popt_duo), 'g--', label='Dual sech$^2$, $h_1$={:.0f} pc, $h_2$={:.0f} pc'.format(np.abs(popt_duo[-2])*1e3, np.abs(popt_duo[-1])*1e3))
    
    plt.title('{:s} {:s}'.format(speclabels[teff], logglabels[logg]))
    plt.xlabel('|Z| (kpc)')
    plt.ylabel('$\\nu$ (kpc$^{-3}$)')
    plt.legend(fontsize='small')
    
    plt.xlim(0,2)
    plt.gca().set_yscale('log')
    plt.gca().set_ylim(bottom=1)
    
    plt.tight_layout()
    plt.savefig('../plots/nu_tgas_logg{}_teff{}.png'.format(logg, teff))

def abscyl_vol_func(X, Y, Z, xymin=0, xymax=0.15, zmin=0.05, zmax=0.15):
    """A function that bins in cylindrical annuli around the Sun, symmetric wrt the plane"""
    
    xy = np.sqrt(X**2 + Y**2)
    out = np.zeros_like(X)
    out[(xy >= xymin)*(xy < xymax)*(np.abs(Z) >= zmin)*(np.abs(Z) < zmax)] = 1
    
    return out

def cyl_vol_func(X, Y, Z, xymin=0, xymax=0.15, zmin=0.05, zmax=0.15):
    """A function that bins in cylindrical annuli around the Sun"""
    
    xy = np.sqrt(X**2 + Y**2)
    out = np.zeros_like(X)
    out[(xy >= xymin)*(xy < xymax)*(Z >= zmin)*(Z < zmax)] = 1
    
    return out

def sech_profile(x, *p):
    """sech^2 profile"""
    
    y = p[0] * (np.cosh(x/(2*p[1])))**-2
    
    return y

def twosech_profile(x, *p):
    """sech^2 profile"""
    
    y = p[0] * ((1-p[1]) * (np.cosh(x/(2*p[2])))**-2 + p[1] * (np.cosh(x/(2*p[3])))**-2)
    
    return y

def vz_distribution(verbose=True):
    """"""
    
    s = Sample()
    finite = np.isfinite(s.data['Fe'])
    selection = finite & s.giant & s.spectype[-1] & (s.cf>0) & (s.data['age']-s.data['eage']>1)
    
    fep = np.percentile(s.data['Fe'][selection], [33,66])
    fei = np.empty((3, np.sum(selection)), dtype=np.bool)
    fei[0] = s.data['Fe'][selection]<fep[0]
    fei[1] = (s.data['Fe'][selection]>=fep[0]) & (s.data['Fe'][selection]<fep[1])
    fei[2] = s.data['Fe'][selection]>=fep[1]
    
    z_bins = np.arange(0, 4, 0.04)
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    Nrow = 3
    Ncol = np.int(np.ceil(Nb/Nrow))
    d = 2.5
    #print(Nb)
    
    plt.close()
    fig, ax = plt.subplots(Nrow,Ncol, figsize=(Ncol*d, Nrow*d), sharex=True)
    
    for i, ind in enumerate(fei):
        hz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins, weights=s.cf[selection][ind])
        nz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        idx  = np.digitize(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        
        for l in range(Nb):
            if verbose: print(i, l)
            plt.sca(ax[int(l/Ncol)][l%Ncol])
            if np.sum(idx==l+1):
                plt.hist(s.v[:,2][selection][ind][idx==l+1], bins=np.linspace(-250,250,20), histtype='step', color=mpl.cm.magma(i/3), normed=True)
                
                # fitting
                fout = '../data/chains/vz_feh{}_z{}.chain'.format(i,l)
                fit_vz(s.v[:,2][selection][ind][idx==l+1].value, s.verr[:,2][selection][ind][idx==l+1], save=True, fout=fout)
                
            if i==0: plt.title('{} kpc'.format(z[l]), fontsize='medium')
        
    plt.tight_layout(h_pad=0, w_pad=0)

def vz_populations(verbose=True):
    """"""
    
    s = Sample()
    
    z_bins = np.arange(0, 4, 0.04)
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    Nrow = 3
    Ncol = np.int(np.ceil(Nb/Nrow))
    d = 2.5
    #print(Nb)
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    
    for i in range(Npop):
        plt.close()
        fig, ax = plt.subplots(Nrow,Ncol, figsize=(Ncol*d, Nrow*d), sharex=True)
    
        selection = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20)
        hz, be = np.histogram(np.abs(s.x[:,2][selection].value), bins=z_bins, weights=s.cf[selection])
        nz, be = np.histogram(np.abs(s.x[:,2][selection].value), bins=z_bins)
        idx  = np.digitize(np.abs(s.x[:,2][selection].value), bins=z_bins)
        
        for l in range(Nb):
            if verbose: print(i, l)
            plt.sca(ax[int(l/Ncol)][l%Ncol])
            if np.sum(idx==l+1):
                plt.hist(s.v[:,2][selection][idx==l+1], bins=np.linspace(-100,100,30), histtype='step', color=mpl.cm.magma(i/Npop), normed=True)
                
                # sigma clipping
                vz = s.v[:,2][selection][idx==l+1]
                ind = np.zeros(np.size(vz), dtype=bool)
                finite = np.isfinite(vz)
                ind[finite] = True
                nind = np.arange(np.size(ind), dtype=int)
                nsig = 4
                outliers = 1
                laux = 0
                while (outliers>0) & (laux<10):
                    std = np.std(vz[ind])
                    mean = np.mean(vz[ind])
                    out = (vz[ind]>mean + nsig*std) | (vz[ind]<mean - nsig*std)
                    outliers = np.sum(out)
                    sl = nind[ind][out]
                    ind[sl] = False
                    #print(l, outliers, std, mean, np.sum(ind))
                    laux += 1
                
                # fitting
                fout = '../data/chains/vz_logg{}_teff{}_z{}.chain'.format(logg_id[i], teff[i],l)
                fit_vz(s.v[:,2][selection][idx==l+1][ind].value, s.verr[:,2][selection][idx==l+1][ind], save=True, fout=fout)
                
            if i==0: plt.title('{} kpc'.format(z[l]), fontsize='medium')
        
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.savefig('../plots/vz_logg{}_teff{}_z{}.png'.format(logg_id[i], teff[i],l))

def fit_vz(vz, vze, save=False, fout='', nstep=200, nburn=200, nwalkers=50):
    """"""
    
    init = np.array([2., 10.])
    ndim = np.size(init)
    pos = [init + init*1e-10*np.random.randn(ndim) for i in range(nwalkers)]
    threads = 3
    pool = multiprocessing.Pool(threads)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_vz, pool=pool, args=(vz, vze))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    
    if save:
        np.savez(fout, lnp=sampler.flatlnprobability, chain=sampler.flatchain)
    
    pool.close()

def lnprob_vz(p, x, xerr):
    """"""
    lp = lnprior_vz(p)
    if ~np.isfinite(lp):
        return -np.inf
    else:
        sz2 = p[1]**2 + xerr**2
        delta = np.sqrt((x - p[0])**2/sz2)
        #lnp = -0.5 * np.sum(np.log(4*np.pi*sz2) + delta**2)
        lnp = -0.5 * np.sum(np.log(4*np.pi*sz2) + delta**2.5/(1+delta))
        if np.isfinite(lnp):
            return lp + lnp
        else:
            return -np.inf

def lnprior_vz(p):
    """Require positive sigma_z, <v_z> close to 0"""
    if (p[1]>0) & (np.abs(p[0])<10):
        return 0.
    else:
        return -np.inf

def dataset(verbose=False):
    """Combine and save datasets"""
    
    s = Sample()
    finite = np.isfinite(s.data['Fe'])
    selection = finite & s.giant & s.spectype[-1] & (s.cf>0) & (s.data['age']-s.data['eage']>1)
    
    fep = np.percentile(s.data['Fe'][selection], [33,66])
    fei = np.empty((3, np.sum(selection)), dtype=np.bool)
    fei[0] = s.data['Fe'][selection]<fep[0]
    fei[1] = (s.data['Fe'][selection]>=fep[0]) & (s.data['Fe'][selection]<fep[1])
    fei[2] = s.data['Fe'][selection]>=fep[1]
    
    z_bins = np.arange(0, 4, 0.2)
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    
    for i, ind in enumerate(fei):
        hz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins, weights=1/s.cf[selection][ind])
        nz, be = np.histogram(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        idx  = np.digitize(np.abs(s.x[:,2][selection][ind].value), bins=z_bins)
        
        n = np.zeros(Nb)
        vz = np.empty(Nb)
        sz = np.empty(Nb)
        vze = np.empty(Nb)
        sze = np.empty(Nb)
        sza = np.empty(Nb)
        
        for l in range(Nb):
            if verbose: print(i, l)
            #plt.sca(ax[int(l/Ncol)][l%Ncol])
            if np.sum(idx==l+1):
                n[l] = np.sum(idx==l+1)
                sampler = np.load('../data/chains/vz_feh{}_z{}.chain.npz'.format(i,l))
                p = np.percentile(sampler['chain'], [16,50,84], axis=0)
                vz[l] = p[1,0]
                sz[l] = p[1,1]
                vze[l] = 0.5*(p[2,0] - p[0,0])
                sze[l] = 0.5*(p[2,1] - p[0,1])
                
                vmin, vmax = np.percentile(s.v[:,2][selection][ind][idx==l+1].value, [15,85])
                i_ = (s.v[:,2][selection][ind][idx==l+1].value>vmin) & (s.v[:,2][selection][ind][idx==l+1].value<vmax)
                sza[l] = np.std(s.v[:,2][selection][ind][idx==l+1][i_].value)
                #sza[l] = np.std(s.v[:,2][selection][ind][idx==l+1].value)
        
        t = Table([z, hz, nz, vz, vze, sz, sze, sza], names=('z', 'nu', 'n', 'vz', 'vze', 'sz', 'sze', 'sza'))
        t.write('../data/profile_mgiants_feh{}.fits'.format(i), overwrite=True)

def dataset_populations(Nboot=100):
    """"""
    
    s = Sample()
    sfok = s.cf>0
    
    z_bins = np.arange(0, 4, 0.04)
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    
    for i in range(Npop):
        selection = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20)
        
        hz, be = np.histogram(np.abs(s.x[:,2][selection & sfok].value), bins=z_bins, weights=1/(s.cf[selection & sfok]))
        nz, be = np.histogram(np.abs(s.x[:,2][selection].value), bins=z_bins)
        idx  = np.digitize(np.abs(s.x[:,2][selection].value), bins=z_bins)
        
        n = np.zeros(Nb)
        vz = np.empty(Nb)
        sz = np.empty(Nb)
        vze = np.empty(Nb)
        sze = np.empty(Nb)
        sza = np.empty(Nb)
        szae = np.empty(Nb)
        
        for l in range(Nb):
            if np.sum(idx==l+1):
                n[l] = np.sum(idx==l+1)
                sampler = np.load('../data/chains/vz_logg{}_teff{}_z{}.chain.npz'.format(logg_id[i], teff[i], l))
                p = np.percentile(sampler['chain'], [16,50,84], axis=0)
                vz[l] = p[1,0]
                sz[l] = p[1,1]
                vze[l] = 0.5*(p[2,0] - p[0,0])
                sze[l] = 0.5*(p[2,1] - p[0,1])
                
                #vmin, vmax = np.percentile(s.v[:,2][selection][idx==l+1].value, [5,95])
                #i_ = (s.v[:,2][selection][idx==l+1].value>vmin) & (s.v[:,2][selection][idx==l+1].value<vmax)
                #sza[l] = np.std(s.v[:,2][selection][idx==l+1][i_].value)
                #sza[l] = np.sqrt(np.std(s.v[:,2][selection][idx==l+1][i_].value)**2 - np.mean(s.v[:,2][selection][idx==l+1][i_].value)**2)
                
                # bootstrap
                Nstar = np.size(s.v[:,2][selection][idx==l+1])
                vzboot = s.v[:,2][selection][idx==l+1][:,np.newaxis].value + s.verr[:,2][selection][idx==l+1][:,np.newaxis]*np.random.randn(Nstar, Nboot)
                szboot = np.sqrt(np.std(vzboot, axis=0)**2 - np.mean(vzboot, axis=0)**2)
                #szboot = np.std(vzboot, axis=0)
                #szboot = np.median(np.abs(vzboot - np.median(vzboot, axis=0)), axis=0)
                #szboot = np.sqrt(np.median(np.abs(vzboot - np.median(vzboot, axis=0)), axis=0)**2 - np.median(vzboot, axis=0)**2)
                sza[l] = np.median(szboot)
                szae[l] = 0.5 * (np.percentile(szboot, [84]) - np.percentile(szboot, [16]))
        
        nan = sz==0
        sz[nan] = np.nan
        sze[nan] = np.nan
        t = Table([z, hz, nz, vz, vze, sz, sze, sza, szae], names=('z', 'nu', 'n', 'vz', 'vze', 'sz', 'sze', 'sza', 'szae'))
        t.write('../data/profile_logg{}_teff{}.fits'.format(logg_id[i], teff[i]), overwrite=True)
        t.pprint()

def dataset_populations_analytic(Nboot=100):
    """"""
    
    s = Sample()
    sfok = s.cf>0
    
    z_bins = np.arange(0, 4, 0.04)
    #z_bins = np.arange(0, 2, 0.1)
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    
    for i in range(Npop):
        selection = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20) #& (s.data['Fe']>-0.5)#& sfok
        #if i>1:
            #finite = np.isfinite(s.data['Fe'])
            #selection = selection & finite & (s.data['Fe']<-0.7)
        
        hz, be = np.histogram(np.abs(s.x[:,2][selection & sfok].value), bins=z_bins, weights=1/(s.cf[selection & sfok]))
        nz, be = np.histogram(np.abs(s.x[:,2][selection].value), bins=z_bins)
        idx  = np.digitize(np.abs(s.x[:,2][selection].value), bins=z_bins)
        #idx  = np.digitize(-s.x[:,2][selection].value, bins=z_bins)
        
        n = np.zeros(Nb)
        vz = np.empty(Nb)
        sz = np.empty(Nb)
        vze = np.empty(Nb)
        sze = np.empty(Nb)
        sza = np.empty(Nb)
        szae = np.empty(Nb)
        feh = np.empty(Nb)
        age = np.empty(Nb)
        
        for l in range(Nb):
            if np.sum(idx==l+1):
                # bootstrap
                Nstar = np.size(s.v[:,2][selection][idx==l+1])
                vzboot = s.v[:,2][selection][idx==l+1][:,np.newaxis].value + s.verr[:,2][selection][idx==l+1][:,np.newaxis]*np.random.randn(Nstar, Nboot)
                vz[l] = np.median(vzboot)
                vze[l] = 0.5 * (np.percentile(vzboot, [84]) - np.percentile(vzboot, [16]))
                
                #szboot = np.sqrt(np.mean(np.abs(vzboot - np.mean(vzboot, axis=0)), axis=0)**2 - np.mean(vzboot, axis=0)**2)
                #szboot = np.std(vzboot, axis=0)
                #szboot = np.median(np.abs(vzboot - np.median(vzboot, axis=0)), axis=0)
                szboot = np.sqrt(np.median(np.abs(vzboot - np.median(vzboot, axis=0)), axis=0)**2 - np.median(vzboot, axis=0)**2)
                sza[l] = np.median(szboot)
                szae[l] = 0.5 * (np.percentile(szboot, [84]) - np.percentile(szboot, [16]))
                #szae[l] = sza[l]/np.sqrt(nz[l])
                #szae[l] = 0.5*np.median(s.verr[:,2][selection][idx==l+1])
                feh[l] = np.nanmedian(s.data['Fe'][selection][idx==l+1])
                age[l] = np.nanmedian(s.data['age'][selection][idx==l+1])
        
        t = Table([z, hz, nz, vz, vze, sza, szae, feh, age], names=('z', 'nu', 'n', 'vz', 'vze', 'sza', 'szae', 'feh', 'age'))
        t.write('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]), overwrite=True)
        t.pprint()


def check_vzfits():
    """"""
    
    s = Sample()
    
    z_bins = np.arange(0, 1, 0.04)
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    Nrow = 3
    Ncol = np.int(np.ceil(Nb/Nrow))
    d = 2.5
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    
    # velocity binning
    v_bins = np.arange(-100,100,5)
    v = myutils.bincen(v_bins)
    Nv = len(v)
    vb = np.linspace(-100,100,100)
    
    for i in range(Npop):
        selection = logg[i] & s.spectype[teff[i]]
        selection_precise = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<2)
        t = Table.read('../data/profile_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        tm = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t.pprint()
        #tm.pprint()
        
        plt.close()
        fig, ax = plt.subplots(Nrow,Ncol, figsize=(Ncol*d, Nrow*d), sharex=True)

        idx = np.digitize(np.abs(s.x[:,2][selection].value), bins=z_bins)
        idx_precise = np.digitize(np.abs(s.x[:,2][selection_precise].value), bins=z_bins)
        
        for l in range(Nb):
            plt.sca(ax[int(l/Ncol)][l%Ncol])
            if np.sum(idx==l+1):
                plt.hist(s.v[:,2][selection][idx==l+1].value, bins=v_bins, histtype='step', lw=1.5, normed=True, label='Data')
                n, be = np.histogram(s.v[:,2][selection][idx==l+1].value, bins=v_bins, density=True)
                n[n==0] = np.nan
                #plt.hist(s.v[:,2][selection_precise][idx_precise==l+1].value, bins=v_bins, histtype='step', lw=1.5, normed=True, label='Data,\n$\sigma_{Vz}$ < 1 km/s')

                # mad sigma
                sz = tm['sza'][l]
                delta = np.abs(vb - tm['vz'][l])/sz
                gauss = (4*np.pi*sz**2)**-0.5*np.exp(-0.5*np.abs(delta)**2)
                modgauss = (4*np.pi*sz**2)**-0.5*np.exp(-0.5*np.abs(delta)**2.5/(1+np.abs(delta)))
                plt.plot(vb, gauss, '-', label='MAD Gauss')
                plt.plot(vb, modgauss, '-', label='MAD ~Gauss')
                
                delta = np.abs(v - t['vz'][l])/sz
                ym = (4*np.pi*sz**2)**-0.5*np.exp(-0.5*np.abs(delta)**2.5/(1+np.abs(delta)))
                #plt.plot(v, 1 - n/ym, 'o', label='MAD ~Gauss')

                # standard deviation sigma
                sz = t['sza'][l]
                delta = np.abs(vb - t['vz'][l])/sz
                gauss = (4*np.pi*sz**2)**-0.5*np.exp(-0.5*np.abs(delta)**2)
                plt.plot(vb, gauss, '-', label='Gauss')
                
                # gaussian sigma
                sz = t['sz'][l]
                delta = np.abs(vb - t['vz'][l])/sz
                gauss = (4*np.pi*sz**2)**-0.5*np.exp(-0.5*np.abs(delta)**2)
                #plt.plot(vb, gauss, '-', label='Gauss')
                modgauss = (4*np.pi*sz**2)**-0.5*np.exp(-0.5*np.abs(delta)**2.5/(1+np.abs(delta)))
                plt.plot(vb, modgauss, '-', label='~Gauss')
                
                delta = np.abs(v - t['vz'][l])/sz
                yg = (4*np.pi*sz**2)**-0.5*np.exp(-0.5*np.abs(delta)**2.5/(1+np.abs(delta)))
                #plt.plot(v, 1 - n/yg, 'o', label='~Gauss')

                #diff = (n - yg)**2 - (n - ym)**2
                ##print(z[l], np.nansum(diff))
                #if np.nansum(diff)>0:
                    #plt.plot(v, diff, 'o', color='orange', label='~Gauss - MAD ~Gauss')
                #else:
                    #plt.plot(v, diff, 'o', color='royalblue', label='~Gauss - MAD ~Gauss')
                #mad_better = diff<0
                #plt.plot(v[mad_better], diff[mad_better], 'ro')
            
            plt.title('|Z| = {:.2f} kpc'.format(z[l]), fontsize='small')
            
            if int(l/Ncol)==Nb-1:
                plt.xlabel('$V_Z$ (km s$^{-1}$)')
            #plt.gca().set_yscale('log')
            #plt.ylim(-6e-5, 6e-5)
            
            if l==0:
                plt.legend(fontsize='xx-small')
            
            #if np.sum(idx==l+1):
                #print('{:d} {:4.2f} {:6.4g}'.format(i, z[l], np.sqrt(np.nansum((yg - n)**2)/np.size(v)) - np.sqrt(np.nansum((ym - n)**2)/np.size(v))))
    
        plt.tight_layout()
        plt.savefig('../plots/vzdistribution_logg{}_teff{}.png'.format(logg_id[i], teff[i]), dpi=200)

def plot_populations():
    """"""
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    
    Ncol = 2
    Nrow = Npop
    d = 3
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d*1.5,Nrow*d), sharex=True, sharey='col')
    
    for i in range(Npop):
        t = Table.read('../data/profile_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        
        plt.sca(ax[i][0])
        plt.plot(t['z'], t['n'], 'ko', ms=3)
        plt.fill_between(t['z'], t['n'] - np.sqrt(t['n']), t['n'] + np.sqrt(t['n']), color='k', alpha=0.3)
        
        # fit density
        z = np.linspace(0,4,100)
        nonzero = (t['z']>0.2) & (t['n']>50)
        pf = np.polyfit(t['z'][nonzero], np.log(t['n'][nonzero]), 1)
        poly = np.poly1d(pf)
        h = 1/np.abs(pf[0])
        plt.plot(z, np.exp(poly(z)), '-', color='r')
        #print(h)
        
        plt.xlim(0,2)
        plt.ylabel('N')
        
        plt.gca().set_yscale('log')
        
        plt.sca(ax[i][1])
        plt.plot(t['z'], t['sz'], 'wo', mec='k', ms=3, label='Gaia + RAVE (Gaussian)')
        plt.fill_between(t['z'], t['sz'] - t['sze'], t['sz'] + t['sze'], color='k', alpha=0.1)
        
        plt.plot(t['z'], t['sza'], 'ko', ms=3, label='Gaia + RAVE (moment)')
        plt.fill_between(t['z'], t['sza'] - t['szae'], t['sza'] + t['szae'], color='k', alpha=0.3)
        
        # predicted velocity dispersion
        sz_pred = sigmaz(h*u.kpc, z*u.kpc, rhodm=0.006*u.Msun*u.pc**-3, sigs=40*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=0.2*u.kpc)
        plt.plot(z, sz_pred, '-', color='r', label='Fiducial $\\rho_{DM}$')
        
        sz_pred = sigmaz(h*u.kpc, z*u.kpc, rhodm=0*u.Msun*u.pc**-3, sigs=40*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=0.2*u.kpc)
        plt.plot(z, sz_pred, '--', color='r', label='No $\\rho_{DM}$')
        
        plt.ylim(0,60)
        plt.ylabel('$\sigma_Z$ (km s$^{-1}$)')
        if i==0:
            plt.legend(fontsize='x-small')
    
    for i in range(2):
        plt.sca(ax[Npop-1][i])
        plt.xlabel('|Z| (kpc)')
        
    plt.tight_layout()
    plt.savefig('../plots/profiles_populations.png')

def plot_populations_analytic(Ncol=5, ncomp=10):
    """"""
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    #Npop = 3
    logg_label = ['dwarfs', 'giants']
    
    if Ncol<2:
        Ncol = 2
    Nrow = Npop
    d = 3
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*1.3*d,Nrow*d), sharex=True, sharey='col')
    
    for i in range(Npop):
        #if i<3:
        tx = Table.read('../data/profile_xd{}_logg{}_teff{}_z79_s1.fits'.format(ncomp, logg_id[i], teff[i]))
        tm = Table.read('../data/profile_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        t = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        
        plt.sca(ax[i][0])
        #plt.plot(t['z'], t['n'], 'wo', ms=0, label='{} {}'.format(speclabels[teff[i]], logg_label[logg_id[i]]))
        ##plt.plot(t['z'], t['n'], 'wo', mec='k', ms=3, label='')
        #plt.plot(t['z'], t['nu'], 'ko', ms=3, label='')
        #plt.fill_between(t['z'], t['nu'] - np.sqrt(t['n']), t['nu'] + np.sqrt(t['n']), color='k', alpha=0.3)
        plt.plot(tx['z'], tx['nueff'], 'ro', ms=3, label='')
        plt.fill_between(tx['z'], tx['nueff']*(1-1/np.sqrt(tx['n'])), tx['nueff']*(1+1/np.sqrt(tx['n'])), color='k', alpha=0.3)
        
        # fit density
        z = np.linspace(0,4,100)
        nonzero = (tx['z']>0.) & (tx['z']<0.65) #& (t['n']>50)
        #print(len(t), np.size(t['z']), np.size(tx[]))
        pf = np.polyfit(tx['zeff'][nonzero], np.log(tx['nueff'][nonzero]), 1)
        poly = np.poly1d(pf)
        h = 1/np.abs(pf[0])
        #if i==0:
            #h = 0.073
        #if i==1:
            #h = 0.096
        #if i==2:
            #h = 0.17
        #if i==3:
            #h = 0.15
        #if i==4:
            #h = 0.2
        plt.plot(z, np.exp(poly(z)), '-', color='r')
        #plt.ylim(1e2, 1e8)
        print(h)
        
        plt.axvspan(0.43, 0.56, color='b', alpha=0.2, zorder=0)
        plt.legend(frameon=False, fontsize='small')
        plt.xlim(-3,3)
        plt.ylabel('N')
        
        plt.gca().set_yscale('log')
        
        plt.sca(ax[i][1])
        #plt.plot(tm['z'], np.sqrt(tm['sz']**2 - tm['vz']**2), 'o', color='0.75', ms=3, label='')
        #plt.plot(tm['z'], tm['sz'], 'wo', mec='k', ms=3, label='Data (fit)')
        #if i<3:
        plt.plot(tx['z'], tx['sz'], 'ro', mec='r', ms=3, label='Data (XD)')
        #plt.fill_between(t['z'], t['sz'] - t['sze'], t['sz'] + t['sze'], color='k', alpha=0.3)
        #plt.plot(t['z'], t['sza'], 'ko', ms=3, label='Data (MAD)')
        #plt.fill_between(t['z'], t['sza'] - t['szae'], t['sza'] + t['szae'], color='k', alpha=0.3)
        
        # predicted velocity dispersion
        #sz_pred = sigmaz(h*u.kpc, z*u.kpc, rhodm=0.006*u.Msun*u.pc**-3, sigs=25*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=0.1*u.kpc)
        sz_pred = sigmaz(h*u.kpc, z*u.kpc, rhodm=0.006*u.Msun*u.pc**-3, sigs=40*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=0.2*u.kpc)
        plt.plot(z, sz_pred, '-', color='r', label='Fiducial $\\rho_{DM}$')
        
        sz_pred = sigmaz(h*u.kpc, z*u.kpc, rhodm=0*u.Msun*u.pc**-3, sigs=40*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=0.2*u.kpc)
        plt.plot(z, sz_pred, '--', color='r', label='No $\\rho_{DM}$')
        
        plt.axvspan(0.43, 0.56, color='b', alpha=0.2, zorder=0)
        
        plt.ylim(0,30)
        plt.ylabel('$\sigma_Z$ (km s$^{-1}$)')
        if i==0:
            plt.legend(fontsize='x-small')
        
        if Ncol>2:
            plt.sca(ax[i][2])
            plt.plot(tm['z'], tm['vz'], 'wo', mec='k', ms=3)
            plt.plot(tx['z'], tx['vz'], 'ko', ms=3)
            plt.fill_between(tx['z'], tx['vz'] - tx['vze'], tx['vz'] + tx['vze'], color='k', alpha=0.3)
            plt.axvspan(0.43, 0.56, color='b', alpha=0.2, zorder=0)

            plt.ylim(-10, 10)
            plt.ylabel('$V_Z$ (km s$^{-1}$)')
        
        if Ncol>3:
            plt.sca(ax[i][3])
            plt.plot(tx['z'], tx['feh'], 'ko', ms=3)
            plt.axvspan(0.43, 0.56, color='b', alpha=0.2, zorder=0)

            plt.ylim(-0.7, 0.1)
            plt.ylabel('[Fe/H]')
        
        if Ncol>4:
            plt.sca(ax[i][4])
            plt.plot(tx['z'], tx['age'], 'ko', ms=3)
            plt.axvspan(0.43, 0.56, color='b', alpha=0.2, zorder=0)

            plt.ylim(10, 0)
            plt.ylabel('Age (Gyr)')
    
    for i in range(2):
        plt.sca(ax[Npop-1][i])
        plt.xlabel('|Z| (kpc)')
        
    plt.tight_layout()
    plt.savefig('../plots/profiles_populations_analytic.png')

def plot_dataset():
    """"""
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(16,5))
    
    colors = [mpl.cm.magma(x/3) for x in range(3)]
    h = np.empty(3)
    zmin = 0.4
    zmax = 2
    
    for i in range(3):
        t = Table.read('../data/profile_mgiants_feh{}.fits'.format(i))
        nonzero = (t['z']>zmin) & (t['z']<zmax)
        plt.sca(ax[0])
        plt.plot(t['z'][nonzero], t['n'][nonzero], 'o', color=colors[i])
        plt.fill_between(t['z'], t['n']-np.sqrt(t['n']), t['n']+np.sqrt(t['n']), color=colors[i], alpha=0.3)
        
        pf = np.polyfit(t['z'][nonzero], np.log(t['n'][nonzero]), 1)
        poly = np.poly1d(pf)
        h[i] = 1/np.abs(pf[0])
        plt.plot(t['z'], np.exp(poly(t['z'])), '-', color=colors[i])
        
        plt.sca(ax[1])
        plt.plot(t['z'], t['sz'], 'o', color=colors[i])
        plt.plot(t['z'], t['sza'], 'wo', mec=colors[i])
        plt.fill_between(t['z'], t['sz']-t['sze'], t['sz']+t['sze'], color=colors[i], alpha=0.3)
        
        print(h[i])
        sz_pred = sigmaz(h[i]*u.kpc, t['z']*u.kpc, rhodm=0.01*u.Msun*u.pc**-3, sigs=0*u.Msun*u.pc**-2, sigg=0*u.Msun*u.pc**-2, H=0.2*u.kpc)
        plt.plot(t['z'], sz_pred, '-', color=colors[i])
        
        plt.sca(ax[2])
        plt.plot(t['z'], t['vz'], 'o', color=colors[i])
        plt.fill_between(t['z'], t['vz']-t['vze'], t['vz']+t['vze'], color=colors[i], alpha=0.3)
    
    plt.sca(ax[0])
    plt.xlim(0,2)
    plt.ylim(1, 1e5)
    plt.gca().set_yscale('log')
    plt.xlabel('|Z| (kpc)')
    plt.ylabel('Number')
    
    plt.sca(ax[1])
    plt.xlim(0,2)
    plt.ylim(0,60)
    plt.xlabel('|Z| (kpc)')
    plt.ylabel('$\sigma_Z$ (km/s)')
    
    plt.sca(ax[2])
    plt.xlim(0,2)
    plt.ylim(-10,10)
    plt.xlabel('|Z| (kpc)')
    plt.ylabel('$V_Z$ (km/s)')
    
    plt.tight_layout()
    plt.savefig('../plots/mgiants_profiles.png')


# velocity ellipsoid

def ellipsoid_z(test=True, dz=0.04, nmin=20, signed=False):
    """Extreme deconvolution of VzVR velocities"""
    
    s = Sample()
    
    if signed:
        z_bins = np.arange(-4, 4+dz, dz)
    else:
        z_bins = np.arange(0, 4+dz, dz)
        s.x[:,2] = np.abs(s.x[:,2])
    
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    
    #Nrow = 5
    #Ncol = np.int(np.ceil(Nb/Nrow))
    #d = 5
    
    logg = [s.dwarf, s.dwarf, s.dwarf, s.giant, s.giant]
    logg_id = [0, 0, 0, 1, 1]
    teff = [2, 3, 4, 5, 6]
    Npop = len(teff)
    
    if test:
        Npop = 1
        Nb = 10
    
    np.random.seed(4091)
    
    # cylindrical coordinates
    vz = s.v[:,2].value
    
    vx = s.v[:,0].value
    vy = s.v[:,1].value
    thx = np.arctan2(s.x[:,1].value, s.x[:,0].value)
    thv = np.arctan2(s.v[:,1].value, s.v[:,0].value)
    vr = np.sqrt(vx**2 + vy**2) * np.cos(thx+thv)
    
    vxe = s.verr[:,0]
    vye = s.verr[:,1]
    vze = s.verr[:,2]
    vre = np.sqrt((vx[0]*vxe/vr[0])**2 + (vy[0]*vye/vr[0])**2) * np.cos(thx+thv)**2
    
    # initial parameters
    mur = 10
    muz = 10
    srr = 100
    srz = 10
    szz = 100
    x0 = np.array([mur, muz, srr, szz, srz])
    
    for i in range(Npop):
        #plt.close()
        ##fig, ax = plt.subplots(Nrow,Ncol, figsize=(Ncol*d, Nrow*d), sharex=True, squeeze=False)
        #plt.figure(figsize=(8,6))
    
        psel = logg[i] & s.spectype[teff[i]] & (s.verr[:,2]<20)
        hz, be = np.histogram(s.x[:,2][psel].value, bins=z_bins, weights=s.cf[psel])
        nz, be = np.histogram(s.x[:,2][psel].value, bins=z_bins)
        idx  = np.digitize(s.x[:,2][psel].value, bins=z_bins)
        
        for l in range(Nb):
            if np.sum(idx==l+1)>nmin:
                zsel = idx==l+1
                vz_ = vz[psel][zsel]
                vr_ = vr[psel][zsel]
                
                vze_ = vze[psel][zsel]
                vre_ = vre[psel][zsel]
                
                N = np.size(vre_)
                v = np.array([vr_, vz_]).T
                sig1 = np.array([vre_, vze_]).T
                
                sig = np.empty((N,2,2))
                for i_ in range(N):
                    sig[i_] = np.diag(sig1[i_])
                
                #lnl = lnlike_ellipsoid(x0, v, sig)
                fit_ellipsoid(x0, v, sig, fout='../data/chains/ellipsoid_l{}_t{}_dz{}_l{}'.format(logg_id[i], teff[i], dz, l), nwalkers=100, nburn=500, nstep=500)


def lnlike_ellipsoid(x, v, sig):
    """"""
    # prior
    if (x[2]<0) | (x[3]<0):
        return -np.inf
    
    # likelihood
    else:
        # populate gaussian vectors
        mu = np.array([x[0],x[1]])
        sigma = np.array([[x[2], x[4]],[x[4], x[3]]])
        
        mu_ = v - mu[np.newaxis,:]
        sigma_ = sigma[np.newaxis,:,:] + sig
        
        # covariance inverse + det
        inv_sigma_ = np.linalg.inv(sigma_)
        det = np.linalg.det(sigma_)
        
        #print(inv_sigma_)
        #print(det)
        
        # calculate chi2
        aux = np.einsum('ijk,ik->ij', inv_sigma_, mu_)
        prod = np.einsum('ij,ij->i', aux, mu_)
        
        # sum over all stars
        lj = -0.5*prod - 2*np.pi - 0.5*np.log(np.abs(det))
        lnl = np.sum(lj)
        
        return lnl

def fit_ellipsoid(init, v, sig, fout='', nstep=400, nburn=200, nwalkers=100):
    """"""
    
    # setup
    ndim = np.size(init)
    pos = [init + init*1e-1*np.random.randn(ndim) for i in range(nwalkers)]
    threads = 3
    pool = multiprocessing.Pool(threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike_ellipsoid, pool=pool, args=(v, sig))
    
    # sample
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    
    # save
    np.savez(fout, lnp=sampler.flatlnprobability, chain=sampler.flatchain)
    
    pool.close()

def check_ellipsoid(logg_id=0, teff=2, dz=0.1, l=0):
    """"""
    #l = 0
    #dz = 0.04
    #logg_id = 1
    #teff = 6
    data = np.load('../data/chains/ellipsoid_l{}_t{}_dz{}_l{}.npz'.format(logg_id, teff, dz, l))
    chain = data['chain']
    lnp = data['lnp']
    
    print(l, np.median(chain, axis=0))
    nstep = 400
    step = np.arange(nstep)
    labels = ['vr', 'vz', 'srr', 'szz', 'srz']
    
    plt.close()
    fig, ax = plt.subplots(2,3,figsize=(9,6))
    
    for i in range(5):
        plt.sca(ax[i%2][np.int64(i/2)])
        plt.plot(step, chain[:,i].reshape(nstep,-1))
        
        plt.xlabel('Step')
        plt.ylabel(labels[i])
    
    plt.sca(ax[1][2])
    plt.plot(step, lnp.reshape(nstep,-1))
    plt.xlabel('Step')
    plt.ylabel('ln P')
    
    plt.tight_layout()

#########
# Fitting

class New(object): pass

class Tracer(New):
    def __init__(self, znu, nu, zsig, sigma, nue=None, sige=None, mask_nu=None, mask_sig=None, nuform='exp', rhoform='disk_dm', nuprior=None):
        self.znu = znu.to(u.kpc)
        self.z = self.znu
        self.nu = nu
        if np.any(nue==None):
            self.nue = nu*0.3
        else:
            self.nue = nue
        
        self.zsig = zsig.to(u.kpc)
        self.sigma = sigma.to(u.km/u.s)
        if np.any(sige==None):
            self.sige = np.ones(np.size(sigma))*1*u.km/u.s
        else:
            self.sige = sige
        
        Nnu = np.size(nu)
        if mask_nu is None:
            self.mask_nu = np.ones(Nnu, dtype=bool)
        else:
            self.mask_nu = mask_nu
        
        self.mask = self.mask_nu
        
        Nsig = np.size(sigma)
        if mask_sig is None:
            self.mask_sig = np.ones(Nsig, dtype=bool)
        else:
            self.mask_sig = mask_sig
        
        if nuprior is None:
            self.nuprior = [0, 1e20]
        else:
            self.nuprior = nuprior
        
        self.nuform = nuform
        if self.nuform=='exp':
            self.nuz = self.exp
        elif self.nuform=='lnexp':
            self.nuz = self.lnexp
        elif self.nuform=='sech2':
            self.nuz = self.sech2
        elif self.nuform=='lnsech2':
            self.nuz = self.lnsech2
        
        if rhoform=='disk_dm':
            self.rho = self.disk_dm
            if 'sech2' in self.nuform:
                self.sigfn = self.sig_disk_dm_sech2
                self.sig2fn = self.sig2_disk_dm_sech2
            else:
                self.sig2fn = self.sig2_disk_dm
                self.sigfn = self.sig_disk_dm
        elif rhoform=='disk':
            self.rho = self.disk
            self.sig2fn = self.sig2_disk
            self.sigfn = self.sig_disk

        # for derived quantities
        self.znu_ = np.linspace(self.znu[0], self.znu[-1], 100)
        self.zsig_ = np.linspace(self.zsig[0], self.zsig[-1], 100)
        self.zrho_ = np.linspace(0, 100, 100)
    
    def fit_nu(self, nwalkers=100, nstep=400):
        ndim = 2
        init = np.array([1., 1.])
        pos = [init + init*5e-2*np.random.randn(ndim) for i in range(nwalkers)]

        self.sampler_nu = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob_nu, args=(self.znu.value, self.nu, self.nue, self.nuz, self.lnprior_nu))
        pos_nu, prob_nu, state_nu = self.sampler_nu.run_mcmc(pos, nstep)
    
    def getbest_nu(self):
        idbest = np.argmax(self.sampler_nu.flatlnprobability)
        self.pbest_nu = self.sampler_nu.flatchain[idbest]
        self.nubest = self.nuz(self.pbest_nu, self.znu_.value)
        
    def fit_sigma(self, nwalkers=100, nstep=400):
        ndim = 3
        init = np.array([50, 0.3, 0.008])
        pos = [init + init*1e-10*np.random.randn(ndim) for i in range(nwalkers)]
        
        sig2 = self.sigma**2
        sige2 = self.sige**2

        self.sampler_sig = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob_sig, 
                                                 args=(sig2, sige2, self.sig2fn, self.lnprior_sig, self.zsig, self.pbest_nu[1]))
        pos_sig, prob_sig, state_sig = self.sampler_sig.run_mcmc(pos, nstep)
        
    
    def lnprob_nu(self, p, x, y, yerr, fn, lnprior, *args, **kwargs):
        lp = lnprior(p)
        if ~np.isfinite(lp):
            return -np.inf
        else:
            lnp = self.lnlike_nu(p, x, y, yerr, fn, *args, **kwargs)
            if np.isfinite(lnp):
                return lp + lnp
            else:
                return -np.inf
    
    def lnprior_nu(self, p):
        if (np.all(p>0)):
            lnprior = -1*0.5*((p[-1]-self.nuprior[-1])/self.nuprior[1])**2
            return lnprior
        else:
            return -np.inf
    
    def lnlike_nu(self, p, x, y, yerr, fn, *args, **kwargs):
        model = fn(p, x, *args, **kwargs)
        inv_sigma2 = 1.0*yerr**-2
        return -0.5*(np.nansum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    
    def lnprob_sig(self, p, y, yerr, fn, lnprior, *args, **kwargs):
        lp = lnprior(p)
        if ~np.isfinite(lp):
            return -np.inf
        else:
            lnp = self.lnlike_sig(p, y, yerr, fn, *args, **kwargs)
            if np.isfinite(lnp):
                return lp + lnp
            else:
                return -np.inf
    
    def lnprior_sig(self, p):
        rho = self.rho(p, self.zrho_)
#         if (np.all(p>0)) & np.all(rho>0) & (p[0]<200) & (p[1]<10):
        if (np.all(p>0)) & np.all(rho>0) & (p[0]<200) & (p[1]<0.75):
            return -0.5*((0.5*p[0]/p[1] - 47)/2)**2
            #return 0
        else:
            return -np.inf
    
    def lnlike_sig(self, p, y, yerr, fn, *args, **kwargs):
        model = fn(p, *args, **kwargs)
        inv_sigma2 = 1.0*yerr**-2
        finite = np.isfinite(y) & np.isfinite(yerr)
        return -0.5*(np.sum((y[finite]-model[finite])**2*inv_sigma2[finite] - np.log(inv_sigma2[finite].value)))
    
    def sig2_disk_dm(self, x, z, h, sgas=13.2):
        """Return vz2 for mass distribution following tracer population"""

        sig2 = 4*np.pi*G*h*u.kpc*(0.5*x[0]*u.Msun*u.pc**-2*(1 - x[1]/(h + x[1])*np.exp(-z.to(u.kpc).value/x[1])) + 0.5*sgas*u.Msun*u.pc**-2 + x[2]*u.Msun*u.pc**-3*(h*u.kpc + z))
        
        return sig2.to(u.km**2*u.s**-2)
    
    def sig_disk_dm(self, x, z, h, sgas=13.2):
        """Return sigma_z for mass distribution following tracer population"""

        sig2 = 4*np.pi*G*h*u.kpc*(0.5*x[0]*u.Msun*u.pc**-2*(1 - x[1]/(h + x[1])*np.exp(-z.to(u.kpc).value/x[1]))
                                  + 0.5*sgas*u.Msun*u.pc**-2
                                  + x[2]*u.Msun*u.pc**-3*(h*u.kpc + z))
#         print(x, z, h, sgas, np.sqrt(sig2).to(u.km*u.s**-1))
        return np.sqrt(sig2).to(u.km*u.s**-1)

    def sig2_disk_dm_sech(self, x, z_, h, sgas=13.2):
        """Return sigma_z^2 for disk + dm mass distribution and a sech2 tracer population"""
        z = (z_.to(u.kpc)).value
        h_ = h*u.kpc
        sig2 = 4*np.pi*G*np.cosh(z/(2*h))**2*(4*h_**2*x[2]*u.Msun*u.pc**-3 * (np.log(2*np.cosh(z/(2*h))) - z/(2*h)*np.tanh(z/(2*h)))
                                       + h_*(x[0] + sgas)*u.Msun*u.pc**-2 * (1 - np.tanh(z/(2*h)))
                                       - h_*x[0]*u.Msun*u.pc**-2*(-1 + x[1]/h)**-1 * ((-1+x[1]/h)*gamma(1-h/x[1])*gamma(1+h/x[1]) + gamma(2-h/x[1])*gamma(h/x[1])
                                                                                      - np.exp(-z/x[1])*(np.exp(z/h)*hyp2f1(1,1-h/x[1],2-h/x[1],-np.exp(z/h))
                                                                                                         + (-1+x[1]/h)*(hyp2f1(1,-h/x[1],1-h/x[1],-np.exp(z/h)) + np.tanh(z/(2*h)))
                                                                                                         )
                                                                                      )
                                       )
        return sig2.to(u.km**2*u.s**-2)
    
    def sig_disk_dm_sech(self, x, z_, h, sgas=13.2):
        """Return sigma_z for disk + dm mass distribution and a sech2 tracer population"""
        z = (z_.to(u.kpc)).value
        h_ = h*u.kpc

        sig2 = 4*np.pi*G*np.cosh(z/(2*h))**2*(4*h_**2*x[2]*u.Msun*u.pc**-3 * (np.log(2*np.cosh(z/(2*h))) - z/(2*h)*np.tanh(z/(2*h)))
                                       + h_*(x[0] + sgas)*u.Msun*u.pc**-2 * (1 - np.tanh(z/(2*h)))
                                       - h_*x[0]*u.Msun*u.pc**-2*(-1 + x[1]/h)**-1 * ((-1+x[1]/h)*gamma(1-h/x[1])*gamma(1+h/x[1]) + gamma(2-h/x[1])*gamma(h/x[1])
                                                                                      - np.exp(-z/x[1])*(np.exp(z/h)*hyp2f1(1,1-h/x[1],2-h/x[1],-np.exp(z/h))
                                                                                                         + (-1+x[1]/h)*(hyp2f1(1,-h/x[1],1-h/x[1],-np.exp(z/h)) + np.tanh(z/(2*h)))
                                                                                                         )
                                                                                      )
                                       )
        return np.sqrt(sig2).to(u.km*u.s**-1)
    
    def sig_disk_dm_sech2(self, x, z_, h, sgas=13.2):
        """Return sigma_z for sech2 disk + const dm and a sech2 tracer population"""
        
        z = (z_.to(u.kpc)).value
        h_ = h*u.kpc
        
        vec_integral = np.vectorize(self.integral_sech2tanh)
        integral = vec_integral(x[1], h, z)*u.kpc
        
        sig2 = 4*np.pi*G*np.cosh(z/(2*h))**2*(h_*sgas*u.Msun*u.pc**-2*(1 - np.tanh(z/(2*h))) 
                                             + 4*h_**2*x[2]*u.Msun*u.pc**-3*(np.log(2*np.cosh(z/(2*h))) - z/(2*h)*np.tanh(z/(2*h)))
                                             + x[0]*u.Msun*u.pc**-2*integral)
        
        return np.sqrt(sig2).to(u.km*u.s**-1)
    
    def sig2_disk_dm_sech2(self, x, z_, h, sgas=13.2):
        """Return sigma_z for sech2 disk + const dm and a sech2 tracer population"""
        
        z = (z_.to(u.kpc)).value
        h_ = h*u.kpc
        
        vec_integral = np.vectorize(self.integral_sech2tanh)
        integral = vec_integral(x[1], h, z)*u.kpc
        
        sig2 = 4*np.pi*G*np.cosh(z/(2*h))**2(h_*sgas*u.Msun*u.pc**-2*(1 - np.tanh(z/(2*h))) 
                                             + 4*h_**2*x[2]*u.Msun*u.pc**-3*(np.log(2*np.cosh(z/(2*h))) - z/(2*h)*np.tanh(z/(2*h)))
                                             + x[1]*u.Msun*u.pc**-2*integral)
        
        return sig2.to(u.km**2*u.s**-2)

    def sig_disk_dm_alt(self, x, z, h, sgas=13.2):
        """Return sigma_z for mass distribution following tracer population"""

        sig2 = 4*np.pi*G*h*u.kpc*(0.5*x[0]*u.Msun*u.pc**-2*(1 - h/(h + x[1])*np.exp(-z.to(u.kpc).value/h))
                                  + 0.5*sgas*u.Msun*u.pc**-2
                                  + x[2]*u.Msun*u.pc**-3*(h*u.kpc + z))
        
#         print(x, z, h, sgas, np.sqrt(sig2).to(u.km*u.s**-1))
        return np.sqrt(sig2).to(u.km*u.s**-1)
    
    def sig2_disk(self, x, z, h, sgas=13.2):
        """Return vz2 for mass distribution following tracer population"""

        sig2 = 4*np.pi*G*h*u.kpc*(0.5*x[0]*u.Msun*u.pc**-2*(1 - x[1]/(h + x[1])*np.exp(-z.value/x[1])) + 0.5*sgas*u.Msun*u.pc**-2)
        
        return sig2.to(u.km**2*u.s**-2)
    
    def sig_disk(self, x, z, h, sgas=13.2):
        """Return sigma_z for mass distribution following tracer population"""

        sig2 = 4*np.pi*G*h*u.kpc*(0.5*x[0]*u.Msun*u.pc**-2*(1 - x[1]/(h + x[1])*np.exp(-z.value/x[1])) + 0.5*sgas*u.Msun*u.pc**-2)
        
        return np.sqrt(sig2).to(u.km*u.s**-1)
    
    def exp(self, p, z):
        """normalization free param"""
        return p[0]*np.exp(-z/p[1])
    
    def lnexp(self, p, z):
        """ln nu, normalization free param"""
        return np.log(p[0]*np.exp(-z/p[1]))
    
    def exp2(self, p, z):
        """normalization determined by extremum of chi_nu"""
        nu0 = np.sum(np.exp(-z)*self.nu[self.mask_nu]/self.nue[self.mask_nu])/np.sum(np.exp(-z)*np.exp(-z/p[0])/self.nue[self.mask_nu])
        return nu0*np.exp(-z/p[0])
    
    def lnexp2(self, p, z):
        """ln nu, normalization determined by extremum of chi_nu"""
        nu0 = np.sum(np.exp(-z)*np.exp(self.nu[self.mask_nu])/np.exp(self.nue[self.mask_nu]))/np.sum(np.exp(-z)*np.exp(-z/p[0])/np.exp(self.nue[self.mask_nu]))
        return np.log(nu0*np.exp(-z/p[1]))
    
    def sech2(self, p, z):
        """"""
        return p[0] * (np.cosh(z/(2.*p[1])))**-2
    
    def lnsech2(self, p, z):
        """"""
        return np.log(p[0] * (np.cosh(z/(2.*p[1])))**-2)
    
    def disk_dm(self, p, z):
        return p[0]/p[1]*np.exp(-z/p[1])*1e-3 + p[2]*np.ones(np.size(z))
    
    def disk(self, p, z):
        return p[0]/p[1]*np.exp(-z/p[1])*1e-3
    
    def iexp_iconst(self, z):
        nu0, h = self.pbest_nu[0], self.pbest_nu[1]

        return nu0*h**2*(1 - (1 + z/h)*np.exp(-z/h))
    
    def iexp_iexp(self, z, H=0.3):
        nu0, h = self.pbest_nu[0], self.pbest_nu[1]
        
        return (nu0*h*H**2/(h + H)*(1 - np.exp(-z*(h + H)/(h*H))) - nu0*h*H*(1 - np.exp(-z/h)))

    #def integrand_sech2tanh(t, H, h):
        #return np.tanh(t/(2*H))/np.cosh(t/(2*h))

    def integrand_sech2tanh(self, t, H, h):
        return 2*h*np.tanh(t*h/H)/np.cosh(t)**2

    def integral_sech2tanh(self, H, h, z):
        return quad(self.integrand_sech2tanh, z/(2*h), np.inf, args=(H, h))[0]

import multiprocessing
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult

from types import MethodType
import copy_reg
import types

import pickle

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def lnprob(p, tracer, ncommon=3, nind=2, rhoprior=False):
    if not type(tracer) is list:
        tracer = [tracer]
    
    nt = len(tracer)
    lnp = 0
    
    for i, t in enumerate(tracer):
        psig = p[:ncommon]
        pnu = p[ncommon+i*nind:ncommon+nind + i*nind]
        #print(psig, pnu)
        
        # nu normalization fixed
        lnp += t.lnprob_nu(pnu, t.znu.value[t.mask_nu], t.nu[t.mask_nu], t.nue[t.mask_nu], t.nuz, t.lnprior_nu)
        #print('lnp_nu {:.1f}'.format(lnp))
        lnp += t.lnprob_sig(psig, t.sigma[t.mask_sig], t.sige[t.mask_sig], t.sigfn, t.lnprior_sig, t.zsig[t.mask_sig], pnu[-1])
        #print('lnp_tot {:.1f}'.format(lnp))
        
        # local density prior
        if rhoprior:
            lnp += -0.5 * ((t.rho(psig, np.array([0]))[0] - 0.2)/(0.05))**2 
    
    return lnp

def lnprob_discrete(p, tracer, ncommon=3, nind=1, rhoprior=False):
    if not type(tracer) is list:
        tracer = [tracer]
    
    nt = len(tracer)
    sgas = 13.2*u.Msun*u.pc**-2
    
    if np.any(p<0) | (p[0]>100) | (p[1]>1):
        lnp = -np.inf
    else:
        lnp = 0
        
        for i, t in enumerate(tracer):
            prho = p[:ncommon]
            pi = p[ncommon+i*nind:ncommon+nind + i*nind]
            
            sstar = p[0]*u.Msun*u.pc**-2
            H = p[1]*u.kpc
            rhodm = p[2]*u.Msun*u.pc**-3
            
            y = np.exp(t.nu[t.mask]) * t.sigma[t.mask]**2
            yerr = np.sqrt((np.exp(t.nue[t.mask])*t.sigma[t.mask]**2)**2 + (np.exp(t.nu[t.mask])*2*t.sigma[t.mask]*t.sige[t.mask])**2)
            
            N = np.sum(t.mask)
            integrals = np.zeros(N)*u.km**2*u.s**-2
            integrals[N-1] = pi[0]*u.km**2*u.s**-2
            
            k = (np.exp(t.nu[t.mask][1:]) - np.exp(t.nu[t.mask][:N-1]))/(t.z[t.mask][1:] - t.z[t.mask][:N-1])
            l = np.exp(t.nu[t.mask][:N-1]) - k*t.z[t.mask][:N-1]

            integrals[:N-1] = (np.pi*G * (sstar + sgas) * k * (t.z[t.mask][1:]**2 - t.z[t.mask][:N-1]**2)
                               - 2*np.pi*G * sstar * k * (H*(H+t.z[t.mask][:N-1])*np.exp((-t.z[t.mask][:N-1]/H).decompose()) - H*(H+t.z[t.mask][1:])*np.exp((-t.z[t.mask][1:]/H)).decompose())
                               + 4*np.pi*G * rhodm * k/3 * (t.z[t.mask][1:]**3 - t.z[t.mask][:N-1]**3)
                               + 2*np.pi*G * (sstar + sgas) * l * (t.z[t.mask][1:] - t.z[t.mask][:N-1])
                               - 2*np.pi*G * sstar * l * H*(np.exp((-t.z[t.mask][:N-1]/H).decompose()) - np.exp((-t.z[t.mask][1:]/H).decompose()))
                               + 2*np.pi*G * rhodm * l * (t.z[t.mask][1:]**2 - t.z[t.mask][:N-1]**2)
                               )
            
            model = np.cumsum(integrals[::-1])[::-1]
            lnp += np.nansum(np.log(np.sqrt(2*np.pi)*t.sigma[t.mask].value) - 0.5*((y - model)/yerr)**2)
            
            #print(y-model)
            #print(model)
            #print(t.nu[t.mask])
            
            # local density prior
            if rhoprior:
                lnp += -0.5 * ((t.rho(prho, np.array([0]))[0] - 0.2)/(0.05))**2
        
    return lnp

def lnprob_npar(p, tracer, ncommon=3, nind=1, rhoprior=False):
    """"""
    if not type(tracer) is list:
        tracer = [tracer]
    
    nt = len(tracer)
    sgas = 13.2*u.Msun*u.pc**-2
    
    if np.any(p[-nt:]<0) | np.any(p[:-nt]<-10) | np.any(p[:-nt]>-1.6):
        lnp = -np.inf
    else:
        lnp = 0
        
        for i, t in enumerate(tracer):
            prho = p[:ncommon]*u.Msun*u.pc**-3
            pi = p[ncommon+i*nind:ncommon+nind + i*nind]
            
            y = np.exp(t.nu[t.mask]) * t.sigma[t.mask]**2
            yerr = np.sqrt((np.exp(t.nue[t.mask])*t.sigma[t.mask]**2)**2 + (np.exp(t.nu[t.mask])*2*t.sigma[t.mask]*t.sige[t.mask])**2)
            
            N = np.sum(t.mask)
            integrals = np.zeros(N)*u.km**2*u.s**-2
            integrals[N-1] = pi[0]*u.km**2*u.s**-2
            
            k = (np.exp(t.nu[t.mask][1:]) - np.exp(t.nu[t.mask][:N-1]))/(t.z[t.mask][1:] - t.z[t.mask][:N-1])
            l = np.exp(t.nu[t.mask][:N-1]) - k*t.z[t.mask][:N-1]
            
            rhoelem = np.zeros(N)*u.Msun*u.pc**-3*u.kpc
            if t.mask[0]:
                rhoelem[0] = np.exp(prho[0].value)*t.z[t.mask][0]*prho.unit
            else:
                rhoelem[0] = np.exp(prho[0].value)*(t.z[t.mask][1]-t.z[t.mask][0])*prho.unit
            rhoelem[1:] = np.exp(prho[1:].value)*(t.z[t.mask][1:]-t.z[t.mask][:N-1])*prho.unit
            
            rhoint = 4*np.pi*G * np.cumsum(rhoelem)

            integrals[:N-1] = rhoint[:N-1]*k*0.5*(t.z[t.mask][1:]**2 - t.z[t.mask][:N-1]**2) + rhoint[:N-1]*l*(t.z[t.mask][1:] - t.z[t.mask][:N-1])
            
            model = np.cumsum(integrals[::-1])[::-1]
            lnp += np.nansum(np.log(np.sqrt(2*np.pi)*t.sigma[t.mask].value) - 0.5*((y - model)/yerr)**2)
            
            # local density prior
            if rhoprior:
                lnp += -0.5 * ((np.exp(prho[0].value) - 0.2)/(0.05))**2
        
    return lnp


def trim_chain(chain, nwalkers, nstart, npar):
    """Trim number of usable steps in a chain"""
    
    chain = chain.reshape(nwalkers,-1,npar)
    chain = chain[:,nstart:,:]
    chain = chain.reshape(-1, npar)
    
    return chain

def get_density(z, p):
    return p[2] + p[0]/p[1]*np.exp(-z/p[1])*1e-3


def jeans_fit(test=False, logg_id=[0,0,0], teff=[2,3,4], seed=50, init=True, rhoprior=True, nwalkers=100, nstep=1000, label='_xd'):
    """"""
    
    logg_label = ['dwarfs', 'giants']
    Npop = len(teff)
    spt_label = get_sptlabel(teff)

    gaia = [None] * Npop
    t = [None] * Npop
    
    Ncol = 2
    Nrow = Npop
    d = 3
    colors_dict={2:'royalblue', 3:'orangered', 4:'crimson'}
    colors = [colors_dict[x] for x in teff]
    alpha = 0.3
    
    for i in range(Npop):
        t[i] = Table.read('../data/profile_xd10_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i] = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i].pprint()
        N = len(t[i])
        mask_nu = np.zeros(N, dtype=bool)
        ind = (t[i]['z']>0.2) & (t[i]['z']<0.65)
        if teff[i]==2:
            ind = (t[i]['z']>0.) & (t[i]['z']<0.3)
        else:
            ind = (t[i]['z']>0.35) & (t[i]['z']<1)
        mask_nu[ind] = True

        mask_sig = np.ones(N, dtype=bool)
        if teff[i]==2:
            ind = (t[i]['z']<=0.) | (t[i]['z']>0.3)
        else:
            ind = (t[i]['z']<=0.35) | (t[i]['z']>0.65)
        #ind = (t[i]['n']<10) | ((t[i]['z']>0.43) & (t[i]['z']<0.56))
        mask_sig[ind] = False
        
        nuprior = height_prior(specid=teff[i])
        #nuprior = None

        #gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnsech2', mask_nu=mask_nu, mask_sig=mask_sig, nuprior=nuprior)
        gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnexp', mask_nu=mask_nu, mask_sig=mask_sig, nuprior=nuprior)
    
    init_dict={2:0.05, 3:0.1, 4:0.17}
    #init = np.array([8, 0.09, 0.02] + [init_dict[x] for x in teff])
    #init = np.array([42, 0.2, 0.01] + [init_dict[x] for x in teff])
    pind = np.array([[2e6, init_dict[x]] for x in teff]).flatten()
    init = np.array([42, 0.2, 0.01] + pind.tolist())
    #init = np.array([25, 0.2, 0.0] + pind.tolist())
    sgas = 13.2

    if test:
        print(lnprob(init, gaia, ncommon=3, nind=2, rhoprior=rhoprior))
    else:
        # mcmc fitting
        np.random.seed(seed)
        ndim = np.size(init)
        pos = [init + init*1e-5*np.random.randn(ndim) for i in range(nwalkers)]
        threads = 3
        
        #if cont:
            #info = pickle.load('../data/chains/dwarfs_{}.info'.format(spt_label))
            #state = info['state']
            #positions = np.arange(-nwalkers, 0, dtype=np.int64)
            #extension += '_cont'
            
            #cstep = info['step']
            #cwalker = info['nwalkers']
        #else:
            #positions = np.random.randint(np.int64(0.75*info['nwalkersburn']*info['nburn']), high=info['nwalkersburn']*info['nburn'], size=nwalkersmcmc)
            #prng = np.random.RandomState(seeds[2])
            #genstate = np.random.get_state()
        
        pool = multiprocessing.Pool(threads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=([gaia, 3, 2, rhoprior]))
        pos, prob, state = sampler.run_mcmc(pos, nstep)
        
        np.savez('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label), lnp=sampler.flatlnprobability, chain=sampler.flatchain)
        pool.close()

        info = {'spt': teff, 'logg': logg_id, 'init': init, 'pos': pos, 'prob': prob, 'state': state, 'ndim': ndim, 'nwalkers':nwalkers, 'nstep': np.size(sampler.flatlnprobability)}
        pickle.dump(info, open('../data/chains/dwarfs_{}_xd.info'.format(spt_label), 'wb'))
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d*1.5,Nrow*d), sharex=True, sharey='col', squeeze=False)
    
    for i, t in enumerate(gaia):
        plt.sca(ax[i][0])
        plt.plot(t.znu.to(u.kpc).value, t.nu, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.znu[t.mask_nu].to(u.kpc).value, t.nu[t.mask_nu], 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.znu.to(u.kpc).value, t.nu, yerr=t.nue, ecolor=colors[i], fmt='none', alpha=alpha)
        if t.nuform=='exp':
            plt.gca().set_yscale('log')
        plt.ylim(5,15.5)
        
        plt.ylabel('ln N')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
        
        z = np.linspace(0,2,100)
        #sz_pred = t.sigfn(init[:3], z*u.kpc, init[3+2*i+1], sgas=13.2)
        sz_pred0 = t.sig_disk_dm(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)
        #sz_pred00 = sigmaz(init[3+2*i+1]*u.kpc, z*u.kpc, rhodm=init[2]*u.Msun*u.pc**-3, sigs=init[0]*u.Msun*u.pc**-2, sigg=sgas*u.Msun*u.pc**-2, H=init[1]*u.kpc)
        sz_pred1 = t.sig_disk_dm_sech(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)
        sz_pred2 = t.sig_disk_dm_sech2(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)

        plt.sca(ax[i][1])
        plt.plot(t.zsig.value, t.sigma.value, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.zsig[t.mask_sig].value, t.sigma[t.mask_sig].value, 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.zsig.value, t.sigma.value, yerr=t.sige.value, ecolor=colors[i], fmt='none', alpha=alpha)
        #plt.plot(z, sz_pred00, 'b-', label='exp, exp')
        plt.plot(z, sz_pred0, 'r-', label='exp, exp')
        plt.plot(z, sz_pred1, 'r--', label='exp, sech')
        plt.plot(z, sz_pred2, 'r:', label='sech, sech')
        plt.ylim(0,20)
        plt.xlim(0,1)
        plt.ylabel('$\sigma_Z$ (km s$^{-1}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()

def jeans_fit_fast(logg_id=[0,], teff=[3,], label='_xd'):
    """"""
    
    logg_label = ['dwarfs', 'giants']
    Npop = len(teff)
    spt_label = get_sptlabel(teff)

    gaia = [None] * Npop
    t = [None] * Npop
    
    Ncol = 3
    Nrow = Npop
    d = 3
    colors_dict={2:'royalblue', 3:'orangered', 4:'crimson'}
    colors = [colors_dict[x] for x in teff]
    alpha = 0.3
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d*1.5,Nrow*d), sharex=True, sharey='col', squeeze=False)
    
    for i in range(Npop):
        t[i] = Table.read('../data/profile_xd10_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i] = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i].pprint()
        N = len(t[i])
        mask_nu = np.zeros(N, dtype=bool)
        ind = (t[i]['z']>0.2) & (t[i]['z']<0.65)
        ind = (t[i]['z']>0.) & (t[i]['z']<0.8)
        mask_nu[ind] = True

        mask_sig = np.ones(N, dtype=bool)
        ind = (t[i]['z']>0.65)
        #ind = (t[i]['n']<10) | ((t[i]['z']>0.43) & (t[i]['z']<0.56))
        mask_sig[ind] = False
        
        nuprior = height_prior(specid=teff[i])
        #nuprior = None

        gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnsech2', mask_nu=mask_nu, mask_sig=mask_sig, nuprior=nuprior)
    
    for i, t in enumerate(gaia):
        plt.sca(ax[i][0])
        #plt.plot(t.znu.to(u.kpc).value, t.nu, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.znu[t.mask_nu].to(u.kpc).value, 1/t.nu[t.mask_nu], 'o', color=colors[i], alpha=alpha)
        #plt.errorbar(t.znu.to(u.kpc).value, t.nu, yerr=t.nue, ecolor=colors[i], fmt='none', alpha=alpha)
        #if t.nuform=='exp':
            #plt.gca().set_yscale('log')
        #plt.ylim(5,15.5)
        
        # fit
        ndeg = 4
        ppars1 = np.polyfit(t.znu[t.mask_nu].to(u.kpc).value, 1/t.nu[t.mask_nu], ndeg)
        p1 = np.poly1d(ppars1)
        z = np.linspace(0,1,100)
        plt.plot(z, p1(z), 'k-')
        
        
        plt.ylabel('$\\nu^{-1}$')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
        plt.xlim(0, 1)

        plt.sca(ax[i][1])
        plt.plot(t.zsig.value, t.nu*(t.sigma.value)**2, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.zsig[t.mask_sig].value, t.nu[t.mask_sig]*(t.sigma[t.mask_sig].value)**2, 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.zsig.value, t.sigma.value, yerr=t.sige.value, ecolor=colors[i], fmt='none', alpha=alpha)
        
        plt.ylabel('$\\nu$ $\sigma_Z^2$ (kpc$^{-3}$ km s$^{-1}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
        plt.xlim(0, 1)
        plt.ylim(0,4000)
        
        # fit
        ndeg = 4
        ppars2 = np.polyfit(t.zsig[t.mask_sig].value, t.nu[t.mask_sig]*(t.sigma[t.mask_sig].value)**2, ndeg)
        p2 = np.poly1d(ppars2)
        z = np.linspace(0,1,100)
        plt.plot(z, p2(z), 'k-')
        
        print(ppars2)
        
        # density
        # 4piG rho = -[(1/nu)' * (B) + (1/nu) * B']
        # with B = (nu sig^2)'
        
        A = p1
        B = np.polyder(p2, 1)
        A_ = np.polyder(A, 1)
        B_ = np.polyder(B, 1)
        
        plt.sca(ax[i][2])
        rho = (-(A_(z) * B(z) + A(z) * B_(z)) * u.km**2*u.s**-2*u.kpc**-2 / (4*np.pi*G)).to(u.Msun*u.pc**-3)
        plt.plot(z, rho, 'k-')
        
        plt.axvspan(0.6, 1, color='k', alpha=0.3)
        plt.gca().set_ylim(bottom=0)
        plt.xlim(0, 1)
        
        plt.ylabel('$\\rho$ ($M_\odot$ $pc^{-3}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()
    plt.savefig('../plots/poly_nu_sig_{}.png'.format(spt_label))
    

def discrete_jeans_fit(test=False, logg_id=[0,0,0], teff=[2,3,4], seed=50, init=True, rhoprior=True, nwalkers=100, nstep=1000, label='_xd'):
    """Fit w discrete nu"""
    
    logg_label = ['dwarfs', 'giants']
    Npop = len(teff)
    spt_label = get_sptlabel(teff)

    gaia = [None] * Npop
    t = [None] * Npop
    
    Ncol = 2
    Nrow = Npop
    d = 3
    colors_dict={2:'royalblue', 3:'orangered', 4:'crimson'}
    colors = [colors_dict[x] for x in teff]
    alpha = 0.3
    
    for i in range(Npop):
        t[i] = Table.read('../data/profile_xd10_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i] = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i].pprint()
        N = len(t[i])
        mask_nu = np.zeros(N, dtype=bool)
        ind = (t[i]['z']>0.2) & (t[i]['z']<0.65)
        if teff[i]==2:
            ind = (t[i]['z']>0.) & (t[i]['z']<0.3)
        else:
            ind = (t[i]['z']>0.35) & (t[i]['z']<1)
        mask_nu[ind] = True

        mask_sig = np.ones(N, dtype=bool)
        if teff[i]==2:
            ind = (t[i]['z']<=0.) | (t[i]['z']>0.3)
        else:
            ind = (t[i]['z']<=0.35) | (t[i]['z']>0.65)
        #ind = (t[i]['n']<10) | ((t[i]['z']>0.43) & (t[i]['z']<0.56))
        mask_sig[ind] = False
        
        mask = np.zeros(N, dtype=bool)
        ind = (t[i]['z']<0.65)
        mask[ind] = True
        
        nuprior = height_prior(specid=teff[i])
        #nuprior = None

        gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnexp', mask_nu=mask, mask_sig=mask, nuprior=nuprior)
        #gaia[i] = Tracer(t[i]['z']*u.kpc, t[i]['nueff'], t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=t[i]['nueff']/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='exp', mask_nu=mask_nu, mask_sig=mask_sig, nuprior=nuprior)
    
    init_dict={2:0.05, 3:0.1, 4:0.17}
    #pind = np.array([[2e6, init_dict[x]] for x in teff]).flatten()
    pind = np.array([[2e6] for x in teff]).flatten()
    init = np.array([42, 0.2, 0.01] + pind.tolist())

    if test:
        print(lnprob_discrete(init, gaia, ncommon=3, nind=1, rhoprior=rhoprior))
    else:
        # mcmc fitting
        np.random.seed(seed)
        ndim = np.size(init)
        pos = [init + init*1e-5*np.random.randn(ndim) for i in range(nwalkers)]
        threads = 3
        
        pool = multiprocessing.Pool(threads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_discrete, pool=pool, args=([gaia, 3, 1, rhoprior]))
        pos, prob, state = sampler.run_mcmc(pos, nstep)
        
        np.savez('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label), lnp=sampler.flatlnprobability, chain=sampler.flatchain)
        pool.close()

        info = {'spt': teff, 'logg': logg_id, 'init': init, 'pos': pos, 'prob': prob, 'state': state, 'ndim': ndim, 'nwalkers':nwalkers, 'nstep': np.size(sampler.flatlnprobability)}
        pickle.dump(info, open('../data/chains/dwarfs_{}_xd.info'.format(spt_label), 'wb'))
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d*1.5,Nrow*d), sharex=True, sharey='col', squeeze=False)
    
    for i, t in enumerate(gaia):
        plt.sca(ax[i][0])
        plt.plot(t.znu.to(u.kpc).value, t.nu, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.znu[t.mask].to(u.kpc).value, t.nu[t.mask], 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.znu.to(u.kpc).value, t.nu, yerr=t.nue, ecolor=colors[i], fmt='none', alpha=alpha)
        if t.nuform=='exp':
            plt.gca().set_yscale('log')
        plt.ylim(5,15.5)
        
        plt.ylabel('ln N')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
        
        z = np.linspace(0,2,100)
        ##sz_pred = t.sigfn(init[:3], z*u.kpc, init[3+2*i+1], sgas=13.2)
        #sz_pred0 = t.sig_disk_dm(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)
        ##sz_pred00 = sigmaz(init[3+2*i+1]*u.kpc, z*u.kpc, rhodm=init[2]*u.Msun*u.pc**-3, sigs=init[0]*u.Msun*u.pc**-2, sigg=sgas*u.Msun*u.pc**-2, H=init[1]*u.kpc)
        #sz_pred1 = t.sig_disk_dm_sech(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)
        #sz_pred2 = t.sig_disk_dm_sech2(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)

        plt.sca(ax[i][1])
        plt.plot(t.zsig.value, t.sigma.value, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.zsig[t.mask].value, t.sigma[t.mask].value, 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.zsig.value, t.sigma.value, yerr=t.sige.value, ecolor=colors[i], fmt='none', alpha=alpha)
        ##plt.plot(z, sz_pred00, 'b-', label='exp, exp')
        #plt.plot(z, sz_pred0, 'r-', label='exp, exp')
        #plt.plot(z, sz_pred1, 'r--', label='exp, sech')
        #plt.plot(z, sz_pred2, 'r:', label='sech, sech')
        plt.ylim(0,20)
        plt.xlim(0,1)
        plt.ylabel('$\sigma_Z$ (km s$^{-1}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()

def npar_jeans_fit(test=False, logg_id=[0,0,0], teff=[2,3,4], seed=50, init=True, rhoprior=True, nwalkers=100, nstep=1000, label='_xd'):
    """fit nonparametric nu and rho"""
    logg_label = ['dwarfs', 'giants']
    Npop = len(teff)
    spt_label = get_sptlabel(teff)

    gaia = [None] * Npop
    t = [None] * Npop
    
    Ncol = 2
    Nrow = Npop
    d = 3
    colors_dict={2:'royalblue', 3:'orangered', 4:'crimson'}
    colors = [colors_dict[x] for x in teff]
    alpha = 0.3
    
    for i in range(Npop):
        t[i] = Table.read('../data/profile_xd10_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i] = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
        #t[i].pprint()
        N = len(t[i])
        
        mask = np.zeros(N, dtype=bool)
        ind = (t[i]['z']<0.7)
        mask[ind] = True
        M = np.sum(mask)
        
        nuprior = height_prior(specid=teff[i])
        #nuprior = None

        gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnexp', mask_nu=mask, mask_sig=mask, nuprior=nuprior)
        #gaia[i] = Tracer(t[i]['z']*u.kpc, t[i]['nueff'], t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=t[i]['nueff']/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='exp', mask_nu=mask_nu, mask_sig=mask_sig, nuprior=nuprior)
    
    #init_dict={2:2e5, 3:5e6, 4:1e8}
    init_dict={2:2e5, 3:1e7, 4:1e8}
    #pind = np.array([[2e6, init_dict[x]] for x in teff]).flatten()
    pind = np.array([[init_dict[x]] for x in teff]).flatten()
    init = np.array([-3 for x in range(M)] + pind.tolist())

    if test:
        print(lnprob_npar(init, gaia, ncommon=M, nind=1, rhoprior=rhoprior))
    else:
        # mcmc fitting
        np.random.seed(seed)
        ndim = np.size(init)
        pos = [init + init*1e-5*np.random.randn(ndim) for i in range(nwalkers)]
        threads = 3
        
        pool = multiprocessing.Pool(threads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_npar, pool=pool, args=([gaia, M, 1, rhoprior]))
        pos, prob, state = sampler.run_mcmc(pos, nstep)
        
        np.savez('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label), lnp=sampler.flatlnprobability, chain=sampler.flatchain)
        pool.close()

        info = {'spt': teff, 'logg': logg_id, 'init': init, 'pos': pos, 'prob': prob, 'state': state, 'ndim': ndim, 'nwalkers':nwalkers, 'nstep': np.size(sampler.flatlnprobability)}
        pickle.dump(info, open('../data/chains/dwarfs_{}_xd.info'.format(spt_label), 'wb'))
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d*1.5,Nrow*d), sharex=True, sharey='col', squeeze=False)
    
    for i, t in enumerate(gaia):
        plt.sca(ax[i][0])
        plt.plot(t.znu.to(u.kpc).value, t.nu, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.znu[t.mask].to(u.kpc).value, t.nu[t.mask], 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.znu.to(u.kpc).value, t.nu, yerr=t.nue, ecolor=colors[i], fmt='none', alpha=alpha)
        if t.nuform=='exp':
            plt.gca().set_yscale('log')
        plt.ylim(5,15.5)
        
        plt.ylabel('ln N')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
        
        z = np.linspace(0,2,100)
        ##sz_pred = t.sigfn(init[:3], z*u.kpc, init[3+2*i+1], sgas=13.2)
        #sz_pred0 = t.sig_disk_dm(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)
        ##sz_pred00 = sigmaz(init[3+2*i+1]*u.kpc, z*u.kpc, rhodm=init[2]*u.Msun*u.pc**-3, sigs=init[0]*u.Msun*u.pc**-2, sigg=sgas*u.Msun*u.pc**-2, H=init[1]*u.kpc)
        #sz_pred1 = t.sig_disk_dm_sech(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)
        #sz_pred2 = t.sig_disk_dm_sech2(init[:3], z*u.kpc, init[3+2*i+1], sgas=sgas)

        plt.sca(ax[i][1])
        plt.plot(t.zsig.value, t.sigma.value, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.zsig[t.mask].value, t.sigma[t.mask].value, 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.zsig.value, t.sigma.value, yerr=t.sige.value, ecolor=colors[i], fmt='none', alpha=alpha)
        ##plt.plot(z, sz_pred00, 'b-', label='exp, exp')
        #plt.plot(z, sz_pred0, 'r-', label='exp, exp')
        #plt.plot(z, sz_pred1, 'r--', label='exp, sech')
        #plt.plot(z, sz_pred2, 'r:', label='sech, sech')
        plt.ylim(0,20)
        plt.xlim(0,1)
        plt.ylabel('$\sigma_Z$ (km s$^{-1}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()


def get_sptlabel(teff):
    """"""
    spt_label = ['{:d}'.format(x) for x in teff]
    spt_label = '_'.join(spt_label)
    
    return spt_label

def analyze_fit(teff=[2,3,4], logg_id=[0,0,0], label=''):
    """Analyze chains of Jeans modeling"""
    
    spt_label = get_sptlabel(teff)
    d = np.load('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label))
    chain = d['chain']
    lnp = d['lnp']
    
    nwalkers = 100
    nstep, ndim = np.shape(chain)
    nstep = int(nstep/nwalkers)
    
    nx = 2
    ny = int((ndim+2)/2)
    dx = 15
    dy = dx*nx/ny
    
    plt.close()
    fig, ax = plt.subplots(nx, ny, figsize=(dx, dy))

    for i in range(ndim):
        plt.sca(ax[int(i/ny)][i%ny])
        plt.plot(np.arange(nstep), chain[:,i].reshape(nwalkers,nstep).T, '-');
        plt.xlabel('Step')

    plt.sca(ax[nx-1][ny-1])
    plt.plot(np.arange(nstep), lnp.reshape(nwalkers,nstep).T, '-');
    plt.xlabel('Step')
    plt.ylabel('ln(p)')

    plt.tight_layout()

def pdf(nstart=0, teff=[2,3,4], logg_id=[0,0,0], label='', params='all'):
    """"""
    spt_label = get_sptlabel(teff)
    d = np.load('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label))
    Npop = len(teff)
    if params=='mw':
        chain = d['chain'][:,:3]
    else:
        chain = d['chain'][:,:3+Npop*2]
    lnp = d['lnp']
    
    nwalkers = 100
    nstep, ndim = np.shape(chain)
    nstep = int(nstep/nwalkers)
    
    samples = trim_chain(chain, nwalkers, nstart, ndim)
    labels = ['$\Sigma_\star$', 'H', '$\\rho_{dm}$', '$\\nu_1$', '$h_1$', '$\\nu_2$', '$h_2$', '$\\nu_3$', '$h_3$']
    labels = labels[:ndim]
    
    plt.close()
    fig = triangle.corner(samples, labels=labels, cmap='gray', quantiles=[0.16,0.50,0.84], angle=0)
    plt.savefig('../plots/jeans_pdf_{:s}_{}{}.png'.format(params, spt_label, label))

def best_fit(teff=[2,3,4], logg_id=[0,0,0], label=''):
    """"""
    logg_label = ['dwarfs', 'giants']
    Npop = len(teff)
    spt_label = get_sptlabel(teff)
    
    gaia = [None] * Npop
    t = [None] * Npop
    
    Ncol = 2
    Nrow = Npop
    d = 3
    colors=['royalblue', 'orangered', 'crimson']
    alpha = 0.3
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d*1.5,Nrow*d), sharex=True, sharey='col', squeeze=False)
    
    for i in range(Npop):
        if label=='':
            t[i] = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
            N = len(t[i])
            mask_nu = np.zeros(N, dtype=bool)
            ind = (t[i]['z']>0.2) & (t[i]['z']<0.65)
            mask_nu[ind] = True

            mask_sig = np.ones(N, dtype=bool)
            ind = ((t[i]['z']>0.43) & (t[i]['z']<0.56)) | (t[i]['n']<10)
            mask_sig[ind] = False

            gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sza']*u.km/u.s, nue=2*np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['szae']*u.km/u.s, nuform='lnexp', mask_nu=mask_nu, mask_sig=mask_sig)
        else:
            t[i] = Table.read('../data/profile_xd10_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
            N = len(t[i])
            mask_nu = np.zeros(N, dtype=bool)
            ind = (t[i]['z']>0.) & (t[i]['z']<0.8)
            mask_nu[ind] = True

            mask_sig = np.ones(N, dtype=bool)
            ind = (t[i]['z']<=0.) | (t[i]['z']>0.65)
            mask_sig[ind] = False

            gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=2*np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnexp', mask_nu=mask_nu, mask_sig=mask_sig)
    
    # best fit
    d = np.load('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label))
    chain = d['chain']
    lnp = d['lnp']
    idbest = np.argmax(lnp)
    p = chain[idbest]
    z = np.linspace(0,2,100)
    print(p)
    
    for i, t in enumerate(gaia):
        plt.sca(ax[i][0])
        plt.plot(t.znu.to(u.kpc).value, t.nu, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.znu[t.mask_nu].to(u.kpc).value, t.nu[t.mask_nu], 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.znu.to(u.kpc).value, t.nu, yerr=t.nue, ecolor=colors[i], fmt='none', alpha=alpha)
        
        #plt.plot(t.znu[t.mask_nu].to(u.kpc).value, t.nuz([p[3+2*i+1]], t.znu[t.mask_nu].to(u.kpc).value), ls='-', lw=2, color='r')
        #print(p[3+2*i:3+2*(i+1)])
        plt.plot(t.znu_.to(u.kpc).value, t.nuz(p[3+2*i:3+2*(i+1)], t.znu_.to(u.kpc).value), ls='-', lw=2, color='r')
        
        if t.nuform=='exp':
            plt.gca().set_yscale('log')
        #plt.gca().set_ylim(bottom=0)
        
        plt.ylabel('ln N')
        plt.ylim(5,20)
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')

        plt.sca(ax[i][1])
        plt.plot(t.zsig.value, t.sigma.value, 'wo', mec=colors[i], alpha=alpha)
        plt.plot(t.zsig[t.mask_sig].value, t.sigma[t.mask_sig].value, 'o', color=colors[i], alpha=alpha)
        plt.errorbar(t.zsig.value, t.sigma.value, yerr=t.sige.value, ecolor=colors[i], fmt='none', alpha=alpha)
        
        # predicted velocity dispersion
        if 'sech2' not in label:
            sz_pred = sigmaz(p[3+2*i+1]*u.kpc, z*u.kpc, rhodm=p[2]*u.Msun*u.pc**-3, sigs=p[0]*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=p[1]*u.kpc)
        else:
            sz_pred = t.sigfn(p[:3], z*u.kpc, p[3+2*i+1], sgas=13.2)
        #sigmaz(p[3+2*i+1]*u.kpc, z*u.kpc, rhodm=p[2]*u.Msun*u.pc**-3, sigs=p[0]*u.Msun*u.pc**-2, sigg=13*u.Msun*u.pc**-2, H=p[1]*u.kpc)
        plt.plot(z, sz_pred, '-', color='r', label='Fiducial $\\rho_{DM}$')
        
        if i==0:
            plt.text(0.4, 0.2, '$\\rho_{dm}$'+' = {:.3f}'.format(p[2])+' $M_\odot$pc$^{-3}$', ha='left', transform=plt.gca().transAxes, fontsize='x-small')
            plt.text(0.4, 0.1, '$\\rho_{dm}$'+' = {:.3f}'.format((p[2]*u.Msun*u.pc**-3).to(u.GeV*u.cm**-3, equivalencies=u.mass_energy()).value)+' GeV cm$^{-3}$', ha='left', transform=plt.gca().transAxes, fontsize='x-small')
        
        plt.ylim(0,20)
        plt.xlim(0,1)
        plt.ylabel('$\sigma_Z$ (km s$^{-1}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()
    plt.savefig('../plots/jeans_bestfit_{}{}.png'.format(spt_label, label))

def best_fit_discrete(teff=[2,3,4], logg_id=[0,0,0], label=''):
    """"""
    logg_label = ['dwarfs', 'giants']
    Npop = len(teff)
    spt_label = get_sptlabel(teff)
    
    gaia = [None] * Npop
    t = [None] * Npop
    
    Ncol = 1
    Nrow = Npop
    d = 5
    colors=['royalblue', 'orangered', 'crimson']
    alpha = 0.3
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d,0.7*Nrow*d), sharex=True, squeeze=False)
    
    for i in range(Npop):
        if label=='':
            t[i] = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
            N = len(t[i])
            mask_nu = np.zeros(N, dtype=bool)
            ind = (t[i]['z']>0.2) & (t[i]['z']<0.65)
            mask_nu[ind] = True

            mask_sig = np.ones(N, dtype=bool)
            ind = ((t[i]['z']>0.43) & (t[i]['z']<0.56)) | (t[i]['n']<10)
            mask_sig[ind] = False

            gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sza']*u.km/u.s, nue=2*np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['szae']*u.km/u.s, nuform='lnexp', mask_nu=mask_nu, mask_sig=mask_sig)
        else:
            t[i] = Table.read('../data/profile_xd10_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
            N = len(t[i])
            
            mask = np.zeros(N, dtype=bool)
            ind = (t[i]['z']<0.65)
            mask[ind] = True
            
            nuprior = height_prior(specid=teff[i])
            #nuprior = None

            gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnexp', mask_nu=mask, mask_sig=mask, nuprior=nuprior)
    
    # best fit
    d = np.load('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label))
    chain = d['chain']
    lnp = d['lnp']
    idbest = np.argmax(lnp)
    p = chain[idbest]
    z = np.linspace(0,2,100)
    print(p)
    ncommon, nind = 3, 1
    
    for i, t in enumerate(gaia):
        plt.sca(ax[i][0])
        
        prho = p[:ncommon]
        pi = p[ncommon+i*nind:ncommon+nind + i*nind]
        
        sgas = 13.2*u.Msun*u.pc**-2
        sstar = p[0]*u.Msun*u.pc**-2
        H = p[1]*u.kpc
        rhodm = p[2]*u.Msun*u.pc**-3
        
        y = np.exp(t.nu[t.mask]) * t.sigma[t.mask]**2
        yerr = np.sqrt((np.exp(t.nue[t.mask])*t.sigma[t.mask]**2)**2 + (np.exp(t.nu[t.mask])*2*t.sigma[t.mask]*t.sige[t.mask])**2)
        
        N = np.sum(t.mask)
        integrals = np.zeros(N)*u.km**2*u.s**-2
        integrals[N-1] = pi[0]*u.km**2*u.s**-2
        
        k = (np.exp(t.nu[t.mask][1:]) - np.exp(t.nu[t.mask][:N-1]))/(t.z[t.mask][1:] - t.z[t.mask][:N-1])
        l = np.exp(t.nu[t.mask][:N-1]) - k*t.z[t.mask][:N-1]

        integrals[:N-1] = (np.pi*G * (sstar + sgas) * k * (t.z[t.mask][1:]**2 - t.z[t.mask][:N-1]**2)
                            - 2*np.pi*G * sstar * k * (H*(H+t.z[t.mask][:N-1])*np.exp((-t.z[t.mask][:N-1]/H).decompose()) - H*(H+t.z[t.mask][1:])*np.exp((-t.z[t.mask][1:]/H)).decompose())
                            + 4*np.pi*G * rhodm * k/3 * (t.z[t.mask][1:]**3 - t.z[t.mask][:N-1]**3)
                            + 2*np.pi*G * (sstar + sgas) * l * (t.z[t.mask][1:] - t.z[t.mask][:N-1])
                            - 2*np.pi*G * sstar * l * H*(np.exp((-t.z[t.mask][:N-1]/H).decompose()) - np.exp((-t.z[t.mask][1:]/H).decompose()))
                            + 2*np.pi*G * rhodm * l * (t.z[t.mask][1:]**2 - t.z[t.mask][:N-1]**2)
                            )
        
        model = np.cumsum(integrals[::-1])[::-1]
        
        plt.plot(t.z[t.mask], y, 'ko')
        plt.errorbar(t.z[t.mask].value, y.value, yerr=yerr.value, color='k', fmt='none')
        plt.plot(t.z[t.mask], model, 'r-')
        
        #if i==0:
            #plt.text(0.4, 0.2, '$\\rho_{dm}$'+' = {:.3f}'.format(p[2])+' $M_\odot$pc$^{-3}$', ha='left', transform=plt.gca().transAxes, fontsize='x-small')
            #plt.text(0.4, 0.1, '$\\rho_{dm}$'+' = {:.3f}'.format((p[2]*u.Msun*u.pc**-3).to(u.GeV*u.cm**-3, equivalencies=u.mass_energy()).value)+' GeV cm$^{-3}$', ha='left', transform=plt.gca().transAxes, fontsize='x-small')
        
        plt.gca().set_yscale('log')
        #plt.ylim(0,20)
        plt.xlim(0,0.8)
        plt.ylabel('$\\nu$ $\sigma_Z^2$ (km$^2$ s$^{-2}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()
    plt.savefig('../plots/jeans_bestfit_{}{}.png'.format(spt_label, label))
    
def best_fit_npar(teff=[2,3,4], logg_id=[0,0,0], label='', nstart=0):
    """"""
    logg_label = ['dwarfs', 'giants']
    Npop = len(teff)
    spt_label = get_sptlabel(teff)
    
    gaia = [None] * Npop
    t = [None] * Npop
    
    Ncol = 1
    Nrow = Npop+1
    d = 5
    colors=['royalblue', 'orangered', 'crimson']
    alpha = 0.3
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol,figsize=(Ncol*d,0.7*Nrow*d), sharex=True, squeeze=False)
    
    for i in range(Npop):
        if label=='':
            t[i] = Table.read('../data/profile_analytic_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
            N = len(t[i])
            mask_nu = np.zeros(N, dtype=bool)
            ind = (t[i]['z']>0.2) & (t[i]['z']<0.65)
            mask_nu[ind] = True

            mask_sig = np.ones(N, dtype=bool)
            ind = ((t[i]['z']>0.43) & (t[i]['z']<0.56)) | (t[i]['n']<10)
            mask_sig[ind] = False

            gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sza']*u.km/u.s, nue=2*np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['szae']*u.km/u.s, nuform='lnexp', mask_nu=mask_nu, mask_sig=mask_sig)
        else:
            t[i] = Table.read('../data/profile_xd10_logg{}_teff{}.fits'.format(logg_id[i], teff[i]))
            N = len(t[i])
            
            mask = np.zeros(N, dtype=bool)
            ind = (t[i]['z']<0.65)
            mask[ind] = True
            
            nuprior = height_prior(specid=teff[i])
            #nuprior = None

            gaia[i] = Tracer(t[i]['z']*u.kpc, np.log(t[i]['nueff']), t[i]['z']*u.kpc, t[i]['sz']*u.km/u.s, nue=np.log(t[i]['nueff'])/np.sqrt(t[i]['n']), sige=t[i]['sze']*u.km/u.s, nuform='lnexp', mask_nu=mask, mask_sig=mask, nuprior=nuprior)
    
    # best fit
    d = np.load('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label))
    chain = d['chain']
    lnp = d['lnp']
    idbest = np.argmax(lnp)
    p = chain[idbest]
    z = np.linspace(0,2,100)
    
    nwalkers = 100
    nstep, ndim = np.shape(chain)
    nstep = int(nstep/nwalkers)
    samples = trim_chain(chain, nwalkers, nstart, ndim)
    
    percentiles = np.percentile(samples, [16,50,84], axis=0)
    
    ncommon, nind = np.sum(mask), 1
    
    plt.sca(ax[0][0])
    plt.plot(gaia[0].z[gaia[0].mask], p[:ncommon], 'ko')
    plt.plot(gaia[0].z[gaia[0].mask], percentiles[1,:ncommon], 'k-')
    plt.fill_between(gaia[0].z[gaia[0].mask].value, percentiles[0,:ncommon], percentiles[2,:ncommon], color='k', alpha=0.2)
    
    print(p)
    print(np.exp(p[:ncommon]))
    p = percentiles[1]
    
    #plt.xlabel('|Z| kpc')
    plt.ylabel('ln $\\rho$ ($M_\odot$ $pc^{-3}$)')
    #plt.gca().set_yscale('log')
    #plt.ylim(1e-4,0.5)
    
    
    for i, t in enumerate(gaia):
        plt.sca(ax[i+1][0])
        
        prho = p[:ncommon]*u.Msun*u.pc**-3
        pi = p[ncommon+i*nind:ncommon+nind + i*nind]
        
        y = np.exp(t.nu[t.mask]) * t.sigma[t.mask]**2
        yerr = np.sqrt((np.exp(t.nue[t.mask])*t.sigma[t.mask]**2)**2 + (np.exp(t.nu[t.mask])*2*t.sigma[t.mask]*t.sige[t.mask])**2)
        
        N = np.sum(t.mask)
        integrals = np.zeros(N)*u.km**2*u.s**-2
        integrals[N-1] = pi[0]*u.km**2*u.s**-2
        
        k = (np.exp(t.nu[t.mask][1:]) - np.exp(t.nu[t.mask][:N-1]))/(t.z[t.mask][1:] - t.z[t.mask][:N-1])
        l = np.exp(t.nu[t.mask][:N-1]) - k*t.z[t.mask][:N-1]
        
        rhoelem = np.zeros(N)*u.Msun*u.pc**-3*u.kpc
        if t.mask[0]:
            rhoelem[0] = np.exp(prho[0].value)*t.z[t.mask][0]*prho.unit
        else:
            rhoelem[0] = np.exp(prho[0].value)*(t.z[t.mask][1]-t.z[t.mask][0])*prho.unit
        rhoelem[1:] = np.exp(prho[1:].value)*(t.z[t.mask][1:]-t.z[t.mask][:N-1])*prho.unit
        
        rhoint = 4*np.pi*G * np.cumsum(rhoelem)

        integrals[:N-1] = rhoint[:N-1]*k*0.5*(t.z[t.mask][1:]**2 - t.z[t.mask][:N-1]**2) + rhoint[:N-1]*l*(t.z[t.mask][1:] - t.z[t.mask][:N-1])
        
        model = np.cumsum(integrals[::-1])[::-1]
        
        plt.plot(t.z[t.mask], y, 'ko')
        plt.errorbar(t.z[t.mask].value, y.value, yerr=yerr.value, color='k', fmt='none')
        plt.plot(t.z[t.mask], model, 'r-')

        plt.gca().set_yscale('log')
        #plt.ylim(0,20)
        plt.xlim(0,0.8)
        plt.ylabel('$\\nu$ $\sigma_Z^2$ (km$^2$ s$^{-2}$)')
        if i==Npop-1:
            plt.xlabel('|Z| (kpc)')
    
    plt.tight_layout()
    plt.savefig('../plots/jeans_bestfit_{}{}.png'.format(spt_label, label))


def dm0(nstart=0, teff=[2,3,4], logg_id=[0,0,0], label=''):
    """Print constraints on local dark matter density"""
    
    spt_label = get_sptlabel(teff)
    d = np.load('../data/chains/dwarfs_{}{}.npz'.format(spt_label, label))
    chain = d['chain']
    
    nwalkers = 100
    nstep, ndim = np.shape(chain)
    nstep = int(nstep/nwalkers)
    samples = trim_chain(chain, nwalkers, nstart, ndim)
    
    percentiles = np.percentile(samples, [16,50,84], axis=0)
    
    dm_ = percentiles[:,2]
    dm = np.array([dm_[1], dm_[1]-dm_[0], dm_[2]-dm_[1]])*u.Msun*u.pc**-3
    print(dm)
    print(dm.to(u.GeV*u.cm**-3, equivalencies=u.mass_energy()))

    percentiles = np.percentile(samples, [1,5,10,16,50,84,90,95], axis=0)
    dm = percentiles[:,2]*u.Msun*u.pc**-3
    print(dm)
    print(dm.to(u.GeV*u.cm**-3, equivalencies=u.mass_energy()))
    

def height_prior(spt=None, specid=3, graph=False):
    """Return average scale height of TGAS dwarfs by spectral type
    Uses Bovy 2017 measurements"""
    
    if spt is None:
        s = Sample()
        selection = s.dwarf & s.spectype[specid] & (s.cf>0)
        spt = s.spt[selection]

    t = Table.read('../data/dwarfs.txt', format='ascii.commented_header', fill_values=['--', np.nan])
    
    ind = myutils.wherein(t['spectype'], spt)
    zd = (t['zd'][ind]*u.pc).to(u.kpc).value
    finite = np.isfinite(zd)
    
    if graph:
        plt.close()
        plt.hist(zd[finite], bins=10)
        
        plt.xlabel('Scale height (kpc)')
        plt.ylabel('Density (kpc$^{-1}$)')
        plt.tight_layout()
    
    return (np.nanmean(zd), np.nanstd(zd))


def sech_exp():
    """"""
    
    H = 0.2
    h = 0.1
    z = np.linspace(0,1,100)
    
    y1 = np.exp(-z/h)
    y2 = np.cosh(z/(2*h))**-2
    y3 = np.tanh(z/(2*H))
    
    plt.close()
    
    #plt.subplot(211)
    plt.plot(z, y1, '-', label='exp (-z/h)')
    #plt.plot(z, 1/y2, '-', label='cosh (-z/(2h))')
    plt.plot(z, y2, '-', label='sech2 (z/(2h))')
    
    plt.xlabel('z')
    plt.ylabel('y(z)')
    
    plt.gca().set_yscale('log')
    plt.legend(fontsize='small')
    
    #plt.subplot(212)
    #plt.plot(z, y2*y3, '-')
    #plt.gca().set_yscale('log')
    
    #plt.xlabel('z')
    #plt.ylabel('sech2 z/(2h) tanh (z/(2H))')

    plt.tight_layout()


#################
# stability test

def test_jeans(logg=0, teff=2, l=98, Nboot=2):
    """Check if the Jeans equation is satisfied for a fiducial Galactic model"""
    
    t = Table.read('../data/profile_xd10_logg{}_teff{}_z{}.fits'.format(logg, teff, l))
    N = len(t)
    
    z = 0.5*(t['z'][:-1] + t['z'][1:])
    z = np.linspace(z[0], 2, 100)
    z_ = z * u.kpc
    
    zmaxnu = {0: 1, 1: 3}
    
    finite = np.isfinite(t['sz'])
    zmax = {0: np.max(t[finite]['z'])-0.05, 1: 2}
    kall = {0: 3, 1: 5}
    k = kall[logg]

    # model
    rhodm = 0.0065*u.Msun*u.pc**-3
    sigs = 42*u.Msun*u.pc**-2
    H = 0.2*u.kpc
    sigg = 13.2*u.Msun*u.pc**-2
    y_model = (4*np.pi*G * (rhodm*z_ + 0.5*sigs*(1 - np.exp(-(z_.value/H.to(z_.unit).value))) + 0.5*sigg)).to(u.km**2 * u.s**-2 * u.kpc**-1)
    
    plt.close()
    fig, ax = plt.subplots(2,3, figsize=(15,10), sharex=True)
    
    if Nboot==1:
        dn = np.zeros(N)[np.newaxis,:]
        dsz = np.zeros(N)[np.newaxis,:]
        dvrz = np.zeros(N)[np.newaxis,:]
    else:
        np.random.seed(59)
        dn = np.random.randn(Nboot,N) * t['nueff']/np.sqrt(t['n'])
        dn[:,t['nueff']==0] = np.zeros((Nboot,np.sum(t['nueff']==0)))
        dsz = np.random.randn(Nboot,N) * t['sze']
        dsz[~np.isfinite(dsz)] = 0
        dvrz = np.random.randn(Nboot,N) * t['vrze']
        dvrz[~np.isfinite(dvrz)] = 0
    
    # data bootstrap
    for i in range(Nboot):
        if i==0:
            labels = ['BSpline fit', 'BSpline fit', 'BSpline prediction']
        else:
            labels = ['' for i_ in range(3)]
        
        # bspline nu
        finite = np.isfinite(t['nueff'])
        tfin = t[finite]
        isort = np.argsort(tfin['z'])
        zaux = np.linspace(z[0], zmaxnu[logg], 5)
        t_ = np.r_[(tfin['z'][isort][0],)*(k+1), zaux, (tfin['z'][isort][-1],)*(k+1)]
        
        fit_nu = scipy.interpolate.make_lsq_spline(tfin['z'][isort], tfin['nueff'][isort] + dn[i][finite][isort], t_, k=k)
        
        # bspline nu sig^2
        finite2 = np.isfinite(t['nueff']) & np.isfinite(t['sz'])
        tfin2 = t[finite2]
        isort2 = np.argsort(tfin2['z'])
        zaux = np.linspace(z[0], zmax[logg], 3)
        t2_ = np.r_[(tfin2['z'][isort2][0],)*(k+1), zaux, (tfin2['z'][isort2][-1],)*(k+1)]
        
        fit_nusig = scipy.interpolate.make_lsq_spline(tfin2['z'][isort2], (tfin2['nueff'][isort2] + dn[i][finite][isort2]) * (tfin2['sz'][isort2] + dsz[i][finite][isort2])**2, t2_, k=k)
        fit_nusig_der = fit_nusig.derivative()
        
        # bspline nu
        finite = np.isfinite(t['vrz'])
        tfin3 = t[finite]
        isort3 = np.argsort(tfin3['z'])
        zaux = np.linspace(z[0], zmax[logg], 3)
        t3_ = np.r_[(tfin3['z'][isort3][0],)*(k+1), zaux, (tfin3['z'][isort3][-1],)*(k+1)]
        
        fit_vrz = scipy.interpolate.make_lsq_spline(tfin['z'][isort3], tfin3['vrz'][isort3] + dvrz[i][finite][isort3], t3_, k=k)
        
        y_data = (-fit_nusig_der(z) / fit_nu(z) - 2*fit_vrz(z)/8.3) * y_model.unit
        
        plt.sca(ax[0][0])
        plt.plot(z, fit_nu(z), 'b-', zorder=0, lw=0.5, label=labels[0])
        
        plt.sca(ax[0][1])
        plt.plot(z, fit_nusig(z), 'b-', zorder=0, lw=0.5, label=labels[1])
        
        plt.sca(ax[0][2])
        plt.plot(z, fit_vrz(z), 'b-', zorder=0, lw=0.5, label=labels[1])
        
        plt.sca(ax[1][0])
        plt.plot(z, y_data, 'b-', zorder=0, lw=0.5, label=labels[2])
        
        plt.sca(ax[1][1])
        plt.plot(z, (y_model - y_data).to(u.cm*u.s**-2), 'b-', zorder=0, lw=0.5)
        
        plt.sca(ax[1][2])
        plt.plot(z, 1 - y_data/y_model, 'b-', zorder=0, lw=0.5)
        
    
    plt.sca(ax[0][0])
    plt.plot(t['z'], t['nueff'], 'ko', alpha=0.3, label='Gaia')
    plt.axvspan(zmax[logg], 4, color='k', alpha=0.3)
    
    plt.gca().set_yscale('log')
    plt.ylabel('Z (kpc)')
    plt.ylabel('N')
    plt.legend(fontsize='small')
    
    plt.sca(ax[0][1])
    plt.plot(t['z'], t['nueff']*t['sz']**2, 'ko', alpha=0.3, label='Gaia + RAVE')
    plt.axvspan(zmax[logg],2, color='k', alpha=0.3)

    plt.gca().set_yscale('log')
    plt.xlabel('Z (kpc)')
    plt.ylabel('N $\sigma_z^2$ (km$^2$ s$^{-2}$)')
    plt.legend(fontsize='small')
    
    plt.sca(ax[0][2])
    plt.plot(t['z'], t['vrz'], 'ko', alpha=0.3, label='Gaia + RAVE')
    plt.axvspan(zmax[logg],2, color='k', alpha=0.3)
    
    plt.ylim(-2000, 2000)
    plt.xlabel('Z (kpc)')
    plt.ylabel('$V_{Rz}$ (km$^2$ s$^{-2}$)')
    plt.legend(fontsize='small')
    
    plt.sca(ax[1][0])
    plt.plot(z, y_model, 'r-', label='Fiducial model')
    plt.axvspan(zmax[logg],2, color='k', alpha=0.3)
    
    plt.ylim(0, 3000)
    plt.xlim(0, 2)
    plt.xlabel('Z (kpc)')
    plt.ylabel('Acceleration (km$^2$ s$^{-2}$ kpc$^{-1}$)')
    plt.legend(fontsize='small')
    
    plt.sca(ax[1][1])
    plt.axhline(0, color='r')
    
    #plt.ylim(-2000, 2000)
    plt.xlabel('Z (kpc)')
    plt.ylabel('$1/\\nu$ $\partial(\\nu v_{z})$ / $\partial t$ (cm s$^{-2}$)')
    
    plt.sca(ax[1][2])
    plt.axhline(0, color='r')
    
    plt.ylim(-2, 2)
    plt.xlabel('Z (kpc)')
    plt.ylabel('Residual (1 - D/M)')

    plt.tight_layout()
    plt.savefig('../plots/jeans_test_logg{}_teff{}_z{}_boot{}.png'.format(logg, teff, l, Nboot))


def test_jeans_2side(logg=0, teff=2, l=79, Nboot=1, alpha=1):
    """Check whether Gaia+RAVE observations on both sides of the disk plane are consistent with fiducial distribution of matter"""
    
    t = Table.read('../data/profile_xd10_logg{}_teff{}_z{}_s1.fits'.format(logg, teff, l))
    idn = t['z']>0
    tn = t[idn]
    
    ids = t['z']<0
    ts = t[ids]
    ts['z'] = -ts['z']
    #ts['vrz'] = -ts['vrz']
    
    labelpn = ['S', 'N']
    colors = ['b', 'g']
    sign = [1, -1]
    data = [[None, None] for x in range(Nboot)]
    
    tx = Table.read('../data/rvrz_xd10_logg{}_teff{}_z{}_s1.fits'.format(logg, teff, l))
    finite = np.isfinite(tx['dvrz'])
    tx = tx[finite]
    Nx = len(tx)
    
    plt.close()
    fig, ax = plt.subplots(2,4, figsize=(15,6), sharex='col')
    
    for e, t in enumerate([ts, tn]):
        N = len(t)
        z = 0.5*(t['z'][:-1] + t['z'][1:])
        z = np.linspace(np.min(z), 2, 100)
        z_ = z * u.kpc
        
        zmaxnu = {0: 1, 1: 2}
        
        finite = np.isfinite(t['sz'])
        zmax = {0: np.max(t[finite]['z'])-0.05, 1: 2.}
        #zmax = {0: 0.6, 1: 2}
        kall = {0: 2, 1: 8}
        krzall = {0: 2, 1: 4}
        k = kall[logg]
        krz = krzall[logg]

        # model
        rhodm = 0.0065*u.Msun*u.pc**-3
        sigs = 42*u.Msun*u.pc**-2
        H = 0.2*u.kpc
        sigg = 13.2*u.Msun*u.pc**-2
        y_model = (4*np.pi*G * (rhodm*z_ + 0.5*sigs*(1 - np.exp(-(z_.value/H.to(z_.unit).value))) + 0.5*sigg)).to(u.km**2 * u.s**-2 * u.kpc**-1)
        
        Mlmc = 1e11*u.Msun
        xlmc = np.array([-0.8133237, -41.00658987, -26.96279919])*u.kpc
        xsun = np.stack([-8.3*np.ones_like(z), np.zeros_like(z), -sign[e]*z])*u.kpc
        glmc = (G*Mlmc*(xlmc[:,np.newaxis]-xsun)**-2).to(u.cm*u.s**-2)
        
        ne = t['nueff']/np.sqrt(t['n'])
        err = np.sqrt((2*t['sze']*t['sz']*t['nueff'])**2 + (t['sz']**2*t['nueff']/np.sqrt(t['n']))**2)
        
        if Nboot==1:
            dn = np.zeros(N)[np.newaxis,:]
            dsz = np.zeros(N)[np.newaxis,:]
            dvrz = np.zeros(N)[np.newaxis,:]
            drvrz = np.zeros(Nx)[np.newaxis,:]
        else:
            np.random.seed(59)
            dn = np.random.randn(Nboot,N) * t['nueff']/np.sqrt(t['n'])
            dn[:,t['nueff']==0] = np.zeros((Nboot,np.sum(t['nueff']==0)))
            dsz = np.random.randn(Nboot,N) * err
            dsz[~np.isfinite(dsz)] = 0
            dvrz = np.random.randn(Nboot,N) * t['vrze']
            dvrz[~np.isfinite(dvrz)] = 0
            drvrz = np.random.randn(Nboot,Nx) * tx['dvrze']
        
        # data bootstrap
        for i in range(Nboot):
            if i==0:
                labels = ['BSpline fit {}'.format(labelpn[e]), 'BSpline fit', 'BSpline prediction']
            else:
                labels = ['' for i_ in range(3)]
            
            # bspline nu
            finite = np.isfinite(t['nueff'])
            tfin = t[finite]
            isort = np.argsort(tfin['z'])
            zaux = np.linspace(np.min(z), zmaxnu[logg], 5)
            t_ = np.r_[(tfin['z'][isort][0],)*(k+1), zaux, (tfin['z'][isort][-1],)*(k+1)]
            
            fit_nu = scipy.interpolate.make_lsq_spline(tfin['z'][isort], tfin['nueff'][isort] + dn[i][finite][isort], t_, k=k)
            p_nu = np.polyfit(tfin['z'][isort], tfin['nueff'][isort] + dn[i][finite][isort], k)
            
            # bspline nu sig^2
            if ((teff==4) | (teff==3)): # & (e==1):
                k = 1
                krz = 1
            finite2 = np.isfinite(t['nueff']) & np.isfinite(t['sz'])
            tfin2 = t[finite2]
            isort2 = np.argsort(tfin2['z'])
            zaux = np.linspace(np.min(z), tfin2['z'][isort2][-2], 3)
            t2_ = np.r_[(tfin2['z'][isort2][0],)*(k+1), zaux, (tfin2['z'][isort2][-1],)*(k+1)]
            
            #fit_nusig = scipy.interpolate.make_lsq_spline(tfin2['z'][isort2], (tfin2['nueff'][isort2] + dn[i][finite][isort2]) * (tfin2['sz'][isort2] + dsz[i][finite][isort2])**2, t2_, k=k)
            fit_nusig = scipy.interpolate.make_lsq_spline(tfin2['z'][isort2], (tfin2['nueff'][isort2]*tfin2['sz'][isort2]**2 + dsz[i][finite][isort2]), t2_, k=k)
            fit_nusig_der = fit_nusig.derivative()
            p_nusig = np.polyfit(tfin2['z'][isort2], (tfin2['nueff'][isort2]*tfin2['sz'][isort2]**2 + dsz[i][finite][isort2]), k)
            p_nusig_der = np.polyder(p_nusig)
            
            # bspline vrvz
            finite = np.isfinite(t['vrz'])
            tfin3 = t[finite]
            isort3 = np.argsort(tfin3['z'])
            zaux = np.linspace(np.min(z), tfin3['z'][isort3][-2], 3)
            #krz = k
            t3_ = np.r_[(tfin3['z'][isort3][0],)*(krz+1), zaux, (tfin3['z'][isort3][-1],)*(krz+1)]
            
            fit_vrz = scipy.interpolate.make_lsq_spline(tfin3['z'][isort3], tfin3['vrz'][isort3] + dvrz[i][finite][isort3], t3_, k=krz)
            p_vrz = np.polyfit(tfin3['z'][isort3], tfin3['vrz'][isort3] + dvrz[i][finite][isort3], krz)
            
            k = kall[logg]
            krz = krzall[logg]

            # polyfit vrvz vs r
            p = np.polyfit(tx['R'], tx['dvrz']+drvrz[i], 1)
            #print(p)

            y_data = (-sign[e]*fit_nusig_der(z) / fit_nu(z) - (1+alpha)*fit_vrz(z)/8.3 - (-sign[e]*600)) * y_model.unit
            #y_data = (-sign[e]*fit_nusig_der(z) / fit_nu(z) - (1+alpha)*fit_vrz(z)/8.3 - (-sign[e]*324)) * y_model.unit
            data[i][e] = y_data.to(u.cm*u.s**-2)
            
            #poly_nusig_der = np.poly1d(p_nusig_der)
            #poly_nu = np.poly1d(p_nu)
            #poly_vrz = np.poly1d(p_vrz)
            #y_data = (-sign[e]*poly_nusig_der(z) / poly_nu(z) - (1+alpha)*poly_vrz(z)/8.3) * y_model.unit
            #data[i][e] = y_data.to(u.cm*u.s**-2)
            
            plt.sca(ax[0][0])
            plt.plot(z, fit_nu(z), '-', color=colors[e], zorder=0, lw=0.5, label=labels[0])
            
            plt.sca(ax[0][1])
            plt.plot(z, fit_nusig(z), '-', color=colors[e], zorder=0, lw=0.5, label=labels[1])
            
            plt.sca(ax[0][2])
            plt.plot(z, fit_vrz(z), '-', color=colors[e], zorder=0, lw=0.5, label=labels[1])
            
            plt.sca(ax[1][0])
            plt.plot(z, y_data, '-', color=colors[e], zorder=0, lw=0.5, label=labels[2])
            
            #plt.sca(ax[1][1])
            #plt.plot(z, sign[e]*(sign[e]*y_model - y_data).to(u.cm*u.s**-2), '-', color=colors[e], zorder=0, lw=0.5)
            
            #plt.sca(ax[1][2])
            #plt.plot(z, 1 - y_data/(sign[e]*y_model), '-', color=colors[e], zorder=0, lw=0.5)
        
    
        plt.sca(ax[0][0])
        plt.plot(t['z'], t['nueff'], 'o', color=colors[e], mec='k', alpha=0.3, label='Gaia {}'.format(labelpn[e]))
        plt.fill_between(t['z'], t['nueff']-ne, y2=t['nueff']+ne, color=colors[e], alpha=0.3, label='')
        plt.axvspan(zmax[logg], 4, color='k', alpha=0.3)
        
        plt.gca().set_yscale('log')
        plt.ylabel('Z (kpc)')
        plt.ylabel('N')
        plt.legend(fontsize='x-small')
        
        plt.sca(ax[0][1])
        plt.plot(t['z'], t['nueff']*t['sz']**2, 'o', color=colors[e], mec='k', alpha=0.3, label='Gaia + RAVE')
        err = np.sqrt((2*t['sz']*t['sze']*t['nueff'])**2 + (t['nueff']/np.sqrt(t['n'])*t['sz']**2)**2)
        plt.fill_between(t['z'], t['nueff']*t['sz']**2+err, y2=t['nueff']*t['sz']**2-err, color=colors[e], alpha=0.3, label='')
        plt.axvspan(zmax[logg],2, color='k', alpha=0.3)

        plt.gca().set_yscale('log')
        #plt.xlabel('Z (kpc)')
        plt.ylabel('N $\sigma_z^2$ (km$^2$ s$^{-2}$)')
        #plt.legend(fontsize='small')
        
        plt.sca(ax[0][2])
        #plt.plot(t['z'], t['sz'], 'o', color=colors[e], mec='k', alpha=0.3, label='Gaia + RAVE')
        plt.plot(t['z'], t['vrz'], 'o', color=colors[e], mec='k', alpha=0.3, label='Gaia + RAVE')
        plt.fill_between(t['z'], t['vrz']-t['vrze'], y2=t['vrz']+t['vrze'], color=colors[e], alpha=0.3, label='')
        plt.axvspan(zmax[logg],2, color='k', alpha=0.3)
        
        plt.ylim(-2000, 2000)
        #plt.xlabel('Z (kpc)')
        plt.ylabel('$V_{Rz}$ (km$^2$ s$^{-2}$)')
        #plt.legend(fontsize='small')
        
        plt.sca(ax[1][0])
        if e==1:
            plt.plot(z, sign[e]*y_model, 'r-', label='Fiducial model')
        else:
            plt.plot(z, sign[e]*y_model, 'r-', label='')
        plt.axvspan(zmax[logg],2, color='k', alpha=0.3)
        
        plt.ylim(-3000, 3000)
        plt.xlim(0, 2)
        plt.xlabel('|Z| (kpc)')
        plt.ylabel('Acceleration (km$^2$ s$^{-2}$ kpc$^{-1}$)')
        #plt.legend(fontsize='small')
        
        plt.sca(ax[1][1])
        plt.axhline(0, color='r')
        #plt.plot(z, glmc[2].value, '--', color=colors[e], lw=3)
        plt.plot(z, y_model.to(u.cm*u.s**-2), 'r-', label='Fiducial model')
        plt.axvspan(zmax[logg], 2, color='k', alpha=0.3)
        
        if e==1:
            for j in range(Nboot):
                plt.plot(z, 0.5*(data[j][0] - data[j][1]), 'k-', zorder=0)
        #plt.ylim(-2000, 2000)
        plt.ylim(-0e-8, 1e-8)
        plt.xlabel('|Z| (kpc)')
        plt.ylabel('|$a_N$ - $a_S$| (cm s$^{-2}$)')
        #plt.ylabel('$1/\\nu$ $\partial(\\nu v_{z})$ / $\partial t$ (cm s$^{-2}$)')
        
        plt.sca(ax[1][2])
        plt.axhline(0, color='r')
        plt.axvspan(zmax[logg],2, color='k', alpha=0.3)
        if e==1:
            for j in range(Nboot):
                plt.plot(z, data[j][1] + data[j][0], 'k-')
            
        plt.ylim(-1e-8, 1e-8)
        plt.xlabel('|Z| (kpc)')
        plt.ylabel('$a_N$ + $a_S$ (cm s$^{-2}$)')

    plt.sca(ax[0][3])
    plt.plot(tx['R'], tx['dvrz'], 'ko', alpha=0.3)
    plt.fill_between(tx['R'], tx['dvrz']-tx['dvrze'], tx['dvrz']+tx['dvrze'], color='k', alpha=0.3)
    
    plt.xlabel('R (kpc)')
    plt.ylabel('$V_{Rz}$ (km$^2$ s$^{-2}$)')
    plt.xlim(7.3, 9.3)
    
    plt.sca(ax[1][3])
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('../plots/jeans_test_logg{}_teff{}_z{}_s1_boot{}.png'.format(logg, teff, l, Nboot))

# north sky
import gaia_tools.xmatch

def match_lamost():
    """"""
    tgas = gaia_tools.load.tgas()
    lamost_cat = gaia_tools.load.lamost(cat='star')
    m1, m2, sep = gaia_tools.xmatch.xmatch(lamost_cat, tgas, colRA1='ra', colDec1='dec', colRA2='ra', colDec2='dec', epoch1=2000., epoch2=2015., swap=True)
    lamost_cat = lamost_cat[m1]
    tgas = tgas[m2]
    print(len(lamost_cat))

def match_apogee():
    """"""
    # load catalogs
    tgas = gaia_tools.load.tgas()
    apogee_cat = gaia_tools.load.apogee()
    
    # make sure all angles are valid
    nan = (apogee_cat['DEC']<-90) | (apogee_cat['RA']<0)
    apogee_cat['RA'][nan] = np.nan
    apogee_cat['DEC'][nan] = np.nan

    if not os.path.isfile('../data/xmatch_tgas_apogee.npz'):
        m1, m2, sep = gaia_tools.xmatch.xmatch(apogee_cat, tgas, colRA2='ra', colDec2='dec', epoch1=2000., epoch2=2015., swap=True)
        np.savez('../data/xmatch_tgas_apogee', mapo=m1, mtgas=m2, sep=sep.value)
    
    xm = np.load('../data/xmatch_tgas_apogee.npz')
    
    apogee_cat = apogee_cat[xm['mapo']]
    tgas = tgas[xm['mtgas']]
    #print(len(apogee_cat))
    
    MJ = apogee_cat['J'] - 5*np.log10(1000/tgas['parallax']) + 5
    JH = apogee_cat['J'] - apogee_cat['H']
    
    plt.close()
    plt.figure()
    
    plt.plot(JH, MJ, 'ko', ms=1, alpha=0.2)
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(8, -3)

def apogee_info():
    """"""
    
    apogee = Table.read('/home/ana/data/gaia_tools/apogee/DR13/allStar-l30e.2.fits')
    #apogee = Table.read('/home/ana/data/gaia_tools/apogee/DR14/allStar-l31c.2.fits')
    print(apogee.colnames)
    apogee.pprint()
    
    print(np.min(apogee['DEC']), np.max(apogee['DEC']))
    print(np.min(apogee['RA']), np.max(apogee['RA']))

def hrd_apogee():
    """Plot the 2MASS HRD for TGAS--RAVE cross-match"""
    
    tgas = gaia_tools.load.tgas()
    t = gaia_tools.load.apogee()
    
    # match
    xm = np.load('../data/xmatch_tgas_apogee.npz')
    t = t[xm['mapo']]
    tgas = tgas[xm['mtgas']]

    MJ = t['J'] - 5*np.log10(1000/tgas['parallax']) + 5
    
    # bins
    jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 0.79])
    jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 1])
    speclabels = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    x = np.array([0.437, 0.359])
    y = np.array([2.6, 1.7])
    pf = np.polyfit(x, y, 1)
    poly = np.poly1d(pf)
    dwarf = (MJ>poly(t['J'] - t['H'])) | (MJ>2.5)
    giant = ~dwarf
    
    plt.close()
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.plot(t['J'] - t['H'], MJ, 'ko', ms=1, alpha=0.01, rasterized=True)
    plt.plot(x, y, 'r-')
    for c in jh_bins:
        plt.axvline(c, color='k', lw=2, alpha=0.2)
    
    plt.xlabel('J - H')
    plt.ylabel('$M_J$')
    plt.xlim(-0.5, 1.5)
    plt.ylim(8, -5)

    plt.sca(ax[1])
    plt.plot(t['J'][giant] - t['H'][giant], MJ[giant], 'ro', ms=1, rasterized=True, label='Giants')
    plt.plot(t['J'][dwarf] - t['H'][dwarf], MJ[dwarf], 'bo', ms=1, rasterized=True, label='Dwarfs')
    for i, c in enumerate(jh_bins):
        plt.axvline(c, color='k', lw=2, alpha=0.2)
        if i>0:
            plt.text(c-0.02, -4, speclabels[i-1], ha='right', fontsize='x-small')
    
    plt.legend(fontsize='small', markerscale=4, handlelength=0.4)
    plt.xlabel('J - H')
    plt.xlim(-0.5, 1.5)
    plt.ylim(8, -5)
    
    plt.tight_layout()
    plt.savefig('../plots/hrd_apogee.png')

def spatial_layout(logg=1, teff=5):
    """"""
    
    tgas = gaia_tools.load.tgas()
    t = gaia_tools.load.apogee()
    s = Sample()
    
    selection = s.giant & s.spectype[teff]
    s.x = s.x[selection]
    s.v = s.v[selection]
    
    # match
    xm = np.load('../data/xmatch_tgas_apogee.npz')
    t = t[xm['mapo']]
    tgas = tgas[xm['mtgas']]
    
    MJ = t['J'] - 5*np.log10(1000/tgas['parallax']) + 5
    JH = t['J'] - t['H']
    
    # bins
    jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 0.79])
    jh_bins = np.array([-0.2, -0.159, -0.032, 0.098, 0.262, 0.387, 0.622, 1])
    speclabels = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    spectype = np.empty((len(jh_bins)-1, len(t)), dtype=bool)
    for i in range(np.size(jh_bins)-1):
        spectype[i] = (JH>=jh_bins[i]) & (JH<jh_bins[i+1])

    x = np.array([0.437, 0.359])
    y = np.array([2.6, 1.7])
    pf = np.polyfit(x, y, 1)
    poly = np.poly1d(pf)
    dwarf = (MJ>poly(t['J'] - t['H'])) | (MJ>2.5)
    giant = ~dwarf
    
    logg_id = {0: dwarf, 1: giant}
    ind = logg_id[logg] & spectype[teff]
    print(np.sum(ind), np.sum(selection))
    
    t = t[ind]
    tgas = tgas[ind]
    
    # Galactocentric coordinates
    c = coord.SkyCoord(ra=np.array(t['RA'])*u.deg, dec=np.array(t['DEC'])*u.deg, distance=1/tgas['parallax']*u.kpc)
    cgal = c.transform_to(coord.Galactocentric)

    x = (np.transpose([cgal.x, cgal.y, cgal.z])*u.kpc).to(u.kpc)
    v = np.transpose(gc.vhel_to_gal(c.icrs, rv=t['VHELIO_AVG']*u.km/u.s, pm=[np.array(tgas['pmra']), np.array(tgas['pmdec'])]*u.mas/u.yr)).to(u.km/u.s)
    
    dz = 0.2
    z_bins = np.arange(-4, 4+dz, dz)
    z = myutils.bincen(z_bins)
    Nb = np.size(z)
    ncomp = 10
    
    idx_n = np.digitize(x[:,2].value, bins=z_bins)
    mu_n = np.ones(Nb) * np.nan
    var_n = np.ones(Nb) * np.nan

    idx_s = np.digitize(s.x[:,2].value, bins=z_bins)
    mu_s = np.ones(Nb) * np.nan
    var_s = np.ones(Nb) * np.nan
    
    for l in range(Nb):
        ind = idx_n==l+1
        mu_n[l] = np.mean(v[:,2][ind].value)
        var_n[l] = np.std(v[:,2][ind].value)
        
        ind = idx_s==l+1
        mu_s[l] = np.mean(s.v[:,2][ind].value)
        var_s[l] = np.std(s.v[:,2][ind].value)
    
    plt.close()
    fig, ax = plt.subplots(1,4,figsize=(16,4), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(x[:,2], bins=z_bins, color='b', histtype='step', normed=True, lw=2, label='Gaia+APOGEE')
    plt.hist(s.x[:,2], bins=z_bins, color='r', histtype='step', normed=True, lw=2, label='Gaia+RAVE')
    
    plt.legend(fontsize='small')
    plt.xlabel('Z (kpc)')
    plt.ylabel('Density (kpc$^{-1}$)')
    
    plt.sca(ax[1])
    plt.plot(z, mu_n, 'bo')
    plt.plot(z, mu_s, 'ro')
    
    plt.xlabel('Z (kpc)')
    plt.ylabel('$V_z$ (km s$^{-1}$)')
    
    plt.sca(ax[2])
    plt.plot(z, var_n, 'bo')
    plt.plot(z, var_s, 'ro')
    
    plt.xlabel('Z (kpc)')
    plt.ylabel('$\sigma_z$ (km s$^{-1}$)')
    
    plt.sca(ax[3])
    plt.plot(z, var_n - var_n[::-1], 'bo', alpha=0.4)
    plt.plot(z, var_s - var_s[::-1], 'ro', alpha=0.4)
    
    plt.xlabel('Z (kpc)')
    plt.ylabel('$\sigma_z$(Z) - $\sigma_z$(-Z)')
    
    plt.xlim(-2,2)
    
    plt.tight_layout()




def full_jeans(logg=1, teff=5, l=39, optimize=True):
    """"""
    
    h = 0.3*u.kpc
    nu0 = 7e5*u.kpc**-3
    sz0 = 15*u.km/u.s
    srz0 = 25*u.km/u.s
    
    if teff==6:
        nu0 = 1e5*u.kpc**-3
        h = 0.25*u.kpc
        srz0 = 25*u.km/u.s
    
    A = 15.3*u.km*u.s**-1*u.kpc**-1
    B = -11.9*u.km*u.s**-1*u.kpc**-1
    C = (10*u.km*u.s**-1)**2 * nu0
    rhodm = 0.008*u.Msun*u.pc**-3
    H = 0.2*u.kpc
    sigs = 42*u.Msun*u.pc**-2
    sigg = 13*u.Msun*u.pc**-2
    Rsun = 8.3*u.kpc
    R0 = 1*u.kpc
    D = 1500*u.km**2*u.s**-2
    n = 2
    z0 = 1*u.kpc
    
    t = Table.read('../data/profile_xd10_logg{}_teff{}_z{}_s0.fits'.format(logg, teff, l))
    #print(t.colnames)
    
    z = np.linspace(0,2,100)*u.kpc
    nuz = nu0*np.exp(-z/h)
    
    vrz = D*(z/z0)**n + (55*u.km/u.s)**2
    
    sz = full_sz(z=z, nu0=nu0, h=h, sz0=sz0, D=D, n=n)
    szfid = full_sz(z=z, nu0=nu0, h=h, sz0=sz0, D=-0*u.km**2*u.s**-2, B=0*u.km*u.s**-1*u.kpc**-1, A=0*u.km*u.s**-1*u.kpc**-1)
    
    x = [sigs.value, H.value, rhodm.value, D.value, n, R0.value, nu0.value, h.value, sz0.value, srz0.value]
    
    mask = (t['z']>0.2) & (t['z']<1.2)
    tm = t[mask]
    nue = tm['nueff']/np.sqrt(tm['n'])
    
    p = lnlike(x, tm['zeff'], tm['nueff'], nue, tm['z'], tm['sz'], tm['sze'], tm['srz'], tm['srze'])
    print(p)
    
    if optimize:
        res = scipy.optimize.minimize(chi_fn, x, args=(tm['zeff'], tm['nueff'], nue, tm['z'], tm['sz'], tm['sze'], tm['srz'], tm['srze']))
        print(res.x)
        print(res.success)
        print(res.status)
        print(res.message)
        np.save('../data/minchi_logg{}_teff{}_z{}_s0'.format(logg, teff, l), res.x)
        x = res.x
    else:
        x = np.load('../data/minchi_logg{}_teff{}_z{}_s0.npy'.format(logg, teff, l))
    print(x)
    
    nuzbest = x[6]*np.exp(-z.value/x[7])
    szbest = full_sz(z=z, sigs=x[0]*u.Msun*u.pc**-2, H=x[1]*u.kpc, rhodm=x[2]*u.Msun*u.pc**-3, D=x[3]*u.km**2*u.s**-2, n=x[4], R0=x[5]*u.kpc, nu0=x[6]*u.kpc**-3, h=x[7]*u.kpc, sz0=x[8]*u.km*u.s**-1)
    srzbest = np.sqrt(x[3]*(z/z0)**x[4] + x[9]**2)
    
    nuzbest_ = x[6]*np.exp(-tm['zeff']/x[7])
    szbest_ = full_sz(z=tm['z']*u.kpc, sigs=x[0]*u.Msun*u.pc**-2, H=x[1]*u.kpc, rhodm=x[2]*u.Msun*u.pc**-3, D=x[3]*u.km**2*u.s**-2, n=x[4], R0=x[5]*u.kpc, nu0=x[6]*u.kpc**-3, h=x[7]*u.kpc, sz0=x[8]*u.km*u.s**-1).value
    srzbest_ = np.sqrt(x[3]*(tm['z']*u.kpc/z0)**x[4] + x[9]**2)
    
    a = 0.2
    
    plt.close()
    fig, ax = plt.subplots(2,3, figsize=(15,7), gridspec_kw = {'height_ratios':[5,2]}, sharex='col', squeeze=False)
    
    plt.sca(ax[0][0])
    plt.plot(t['zeff'], t['nueff'], 'ko', alpha=a)
    plt.errorbar(t['zeff'], t['nueff'], yerr=t['nueff']/np.sqrt(t['n']), fmt='none', color='k', alpha=a)
    plt.plot(tm['zeff'], tm['nueff'], 'ko')
    plt.errorbar(tm['zeff'], tm['nueff'], yerr=tm['nueff']/np.sqrt(tm['n']), fmt='none', color='k')
    plt.plot(z, nuzbest)
    
    plt.gca().set_yscale('log')
    plt.ylabel('$\\nu$ (kpc$^{-3}$)')

    plt.sca(ax[1][0])
    plt.axhline(0, color='r')
    plt.plot(tm['zeff'], tm['nueff']-nuzbest_, 'ko')
    plt.errorbar(tm['zeff'], tm['nueff']-nuzbest_, yerr=tm['nueff']/np.sqrt(tm['n']), fmt='none', color='k')
    plt.xlabel('Z (kpc)')
    plt.ylabel('$\Delta$ $\\nu$')
    
    plt.sca(ax[0][1])
    plt.plot(t['z'], t['sz'], 'ko', alpha=a)
    plt.errorbar(t['z'], t['sz'], yerr=t['sze'], fmt='none', color='k', alpha=a)
    plt.plot(tm['z'], tm['sz'], 'ko')
    plt.errorbar(tm['z'], tm['sz'], yerr=tm['sze'], fmt='none', color='k')
    plt.plot(z, szbest)
    
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
    
    plt.sca(ax[1][2])
    plt.axhline(0, color='r')
    plt.plot(tm['zeff'], tm['srz']-srzbest_, 'ko')
    plt.errorbar(tm['z'], tm['srz']-srzbest_, yerr=tm['srze'], fmt='none', color='k')
    plt.xlabel('Z (kpc)')
    plt.ylabel('$\Delta$ $\sigma_{Rz}$')
    
    
    plt.tight_layout()
    

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

def lnlike(x, znu, nu, nue, z, sz, sze, srz, srze):
    """"""
    if (x[0]<0) | (x[1]<0) | (x[2]<0):
        lnp = -np.inf
    else:
        chi = chi_fn(x, znu, nu, nue, z, sz, sze, srz, srze)
        lnp = -0.5*chi
    
    return lnp

def chi_fn(x, znu, nu, nue, z, sz, sze, srz, srze):
    """"""
    chi = 0.
    chi += chi_nu(x, znu, nu, nue)
    chi += chi_sigz(x, z, sz, sze)
    chi += chi_sigrz(x, z, srz, srze)
    
    return chi

def chi_nu(x, z, nu, nue):
    """"""
    nu_mod = x[6] * np.exp(-z/x[7])
    err = nue/nu
    chi2 = np.nansum((np.log(nu_mod) - np.log(nu))**2/err**2)
    
    return chi2

def chi_sigz(x, z, sz, sze):
    """"""
    sz_mod = full_sz(z=z*u.kpc, sigs=x[0]*u.Msun*u.pc**-2, H=x[1]*u.kpc, rhodm=x[2]*u.Msun*u.pc**-3, D=x[3]*u.km**2*u.s**-2, n=x[4], R0=x[5]*u.kpc, nu0=x[6]*u.kpc**-3, h=x[7]*u.kpc, sz0=x[8]*u.km*u.s**-1).value
    chi2 = np.nansum((sz_mod - sz)**2/sze**2)
    
    return chi2

def chi_sigrz(x, z, srz, srze):
    """"""
    srz_mod = np.sqrt(x[3]*(z)**x[4] + x[9]**2)
    chi2 = np.nansum((srz_mod - srz)**2/srze**2)
    
    return chi2

