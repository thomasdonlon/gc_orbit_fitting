import greatCircle as gc
import numpy as np
import scipy as sc
from astropy import units as u
import scipy.optimize as scopt
from galpy.potential import HernquistPotential
from galpy.potential import LogarithmicHaloPotential
from galpy.potential import MiyamotoNagaiPotential
import matplotlib.pyplot as plt
from galpy.orbit import Orbit

t_length = 0.25 #Gyr
resolution = 1000
ts = np.linspace(0, t_length, num=resolution)*u.Gyr

#==============================
#Galactic Potential Parameters (Newberg et al. 2009, Orphan Stream fit #4)
#==============================
m_bulge = 3.4e10*u.solMass #solar masses
m_disk = 1.0e11*u.solMass
v_halo = 74.61*u.km/u.s #km/s

G = 6.67e-11*u.m**3/(u.kg*u.s**2)

pot_bulge = HernquistPotential(amp=2*m_bulge, a=0.7*u.kpc, ro=8., vo=220.)
pot_disk = MiyamotoNagaiPotential(amp=G*m_disk, a=6.5*u.kpc, b=0.26*u.kpc, ro=8., vo=220.)
pot_halo = LogarithmicHaloPotential(amp=2*v_halo**2, q=1., core=12.0*u.kpc, ro=8., vo=220.)
pot = [pot_bulge, pot_disk, pot_halo]
#==============================

interp = 5 #number of times the getModelFromOrbit process iterates to find a close timestep for each Lambda
interp_res = 5 #number of sections the getModelFromOrbit process splits up the number line into

o = Orbit(vxvv=[300, 67.92, 12.45, 68.76, 102.12 - 220, 215.31], uvw=True, lb=True, ro=8, vo=220., solarmotion=[0,0,0]) #generate the orbit
o.integrate(ts, pot) #integrate the orbit
o_rev = o.flip() #generate the orbit
o_rev.integrate(ts, pot) #integrate the orbit

#o2 = Orbit(vxvv=[230.17, 63.23, 11.96, 285.04, -20.78 - 220, 64.46], uvw=True, lb=True, ro=8, vo=220., solarmotion=[0,0,0]) #generate the orbit
#o2 = Orbit(vxvv=[192.52, 29.44, 16.36, 256.6, -40 - 220, 91], uvw=True, lb=True, ro=8, vo=220., solarmotion=[0,0,0]) #generate the orbit
o2 = Orbit(vxvv=[291.5, 60.4, 14.9, 338, 67 - 220, -17], uvw=True, lb=True, ro=8, vo=220., solarmotion=[0,0,0]) #generate the orbit
o2.integrate(ts, pot) #integrate the orbit
o2_rev = o2.flip() #generate the orbit
o2_rev.integrate(ts, pot) #integrate the orbit

fig = plt.figure(figsize=(12,12))

fig.add_subplot(111)
o.plot(d1='ra', d2='dec', overplot=True)
o_rev.plot(d1='ra', d2='dec', overplot=True)
o2.plot(d1='ra', d2='dec', overplot=True)
o2_rev.plot(d1='ra', d2='dec', overplot=True)
plt.xlim(360, 0)
plt.ylim(-90, 90)

plt.show()
