#============================
#Tom Donlon, 2018
#============================

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
from astropy.coordinates import SkyCoord
import galpy

'''
==============================
PARAMETERS FOR OPTIMIZATION
==============================
'''
verbose = False #whether or not to print warnings

t_length = 0.3 #Gyr
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
'''
============================
'''

class Data():
    #all of these parameters can be np.arrays of floats
    def __init__(self, l, b, d, U, V, W, l_err, b_err, d_err, vel_err):
        self.l = l
        self.b = b
        self.d = d #heliocentric distance
        self.U = U
        self.V = V
        self.W = W

        self.l_err = l_err
        self.b_err = b_err
        self.d_err = d_err
        self.vel_err = vel_err

        self.x = self.d * np.cos(self.l*np.pi/180) * np.cos(self.b*np.pi/180) - 8
        self.y = self.d * np.sin(self.l*np.pi/180) * np.cos(self.b*np.pi/180)
        self.z = self.d * np.sin(self.b*np.pi/180)

    def LamBet(self, normal, point):
        #normal: the normal vector to the plane of the Great Circle we are estimating for the orbit
        #point: parameter for the axis generation of the Great Circle coordinates
        self.L, self.B = gc.gal2LamBet(self.x, self.y, self.z, normal=normal, point=point)
        self.D = (self.x**2 + self.y**2 + self.z**2)**0.5
        self.L_err = self.l_err #should be error propogated (Newby et al. 2013) but is not yet
        self.B_err = self.b_err #should be error propogated (Newby et al. 2013) but is not yet
        self.D_err = self.d_err #should be error propogated but is not yet
        return self

#getSliceIndices: np.array([]) -> list(int)
#outputs a list of indices of where to slice to get segments (slices along where the segments wrap)
def getSliceIndices(Lam):
    #Lam: the array of Lambda values that you are searching through

    #split at every wrap:
    #warning, indexing gets pretty dicey here so hold on

    #get the list of indices right before the split
    #i.e. [... 350, 355, 5, 10... ] grab the index of 355
    split_ind = [-1] #-1 because it makes things simpler in the next section: think of it as a 0
    i = 0
    while i < len(Lam)-2:
        #if it is discontinuous, mark it as the end of a section
        if abs(Lam[i] - Lam[i + 1]) >= 300: #not 360 because it's not precisely 0 and 360, but 300 should be inclusive and exclusive enough of a buffer
            split_ind.append(i)
        i += 1
    split_ind.append(len(Lam)-2) #-2 because it makes things simpler in the next section: think of it as a -1

    return split_ind

#getBoundsSegmentWithVal: float, Lam, list(int) -> int int
#provides the indix bounds for the segment inclusive of val
def getBoundsSegmentWithVal(val, Lam, split_ind):
    #val: the Lambda value that you want to find the closest index to
    #Lam: the array of Lambda values that you are searching through
    #split_ind: the list if integers describing where to split segments

    #separate the segments according to split_ind
    segments = []
    i = 0
    while i < len(split_ind) - 1:
        segment = Lam[split_ind[i]+1:split_ind[i+1]+1] #+1 on both sides
        #this causes us to never repeat a number (+1 on the left)
        #and also not truncate on the end of the section (+1 on the right)
        if min(segment) <= val <= max(segment): #check if val is in this segment
            #could make this so it throws an error if things are double-valued but that's a future project if it becomes a problem
            return split_ind[i]+1, split_ind[i+1]+1
        i += 1

    if verbose:
        print('WARNING: Could not find a segment including a data point')
        print('(error: getBoundsSegmentWithVal)')
    return split_ind[0], split_ind[len(split_ind) - 1]

#getSegmentBounds: float, np.array([]) -> int int
#provide the lower and upper indices for a segment of Lam that does not wrap and has val inclusive
def getSegmentBounds(val, Lam):
    #val: the Lambda value that you want to find the closest index to
    #Lam: the array of Lambda values that you are searching through

    split_ind = getSliceIndices(Lam)
    ni, nf = getBoundsSegmentWithVal(val, Lam, split_ind)

    return ni, nf

#getClosestIndex: float, np.array([]), int, int -> int
#given a segment, finds the index that provides the closest value in Lam to the input val
def getClosestIndex(val, Lam, ni, nf):
    #val: the Lambda value that you want to find the closest timestep to
    #Lam: the array of Lambda values that you are searching through
    #ni: lower limit of time to check (int)
    #nf: upper limit of time to check (int)

    new_ni = ni
    new_nf = ni #these are failsafes to keep the program running without errors in the event that the fitter tests a bad orbit that doesn't pass over the data

    if (nf - ni) <= 1:#determines if you have found the closest two indices to the correct answer
        if abs(Lam[ni] - val) < abs(Lam[nf] - val): #just chooses the better of the two
            return ni
        else:
            return nf

    else: #haven't found the closest two indices, so lets scan the current section of the 'number line'
        for i in range(0, interp_res):
            bottom_split_time = int((nf - ni) / interp_res * i + ni) #chooses bounds of the section of the 'number line'
            top_split_time = int((nf - ni) / interp_res * (i + 1) + ni)

            #if val is within this section, run it again with these as our bounds, goes both directions (increasing and decreasing)
            if (Lam[bottom_split_time] <= val <= Lam[top_split_time]) or (Lam[bottom_split_time] >= val >= Lam[top_split_time]):
                new_ni = bottom_split_time
                new_nf = top_split_time
                break #don't need to scan anymore if we found it

        return getClosestIndex(val, Lam, new_ni, new_nf) #now that we've narrowed down our bounds; wash, rinse, repeat

#getPointList: np.array([]), np.array([]), int, int -> np.array([])
#given the full list of Lambdas, outputs the indices of the points within that list closest to our data's Lambdas
#(while keeping in mind that it may wrap from 0 to 360 degrees and vice versa)
def getPointList(vals, Lam):
    #vals: the Lambda values that you want to find the closest indices to
    #Lam: the array of Lambda values that you are searching through

    point_list = []

    for val in vals:
        #split segments according to wrapping
        #find the segment with val in it, return that
        ni, nf = getSegmentBounds(val, Lam)

        #within that segment, find the index which produces the value closest to val
        point = getClosestIndex(val, Lam, ni, nf)

        #toss it in the list
        point_list.append(point)

    return point_list

#getModelFromOrbit: data, orbit, vector, vector -> list(int) x3
#take in data, orbit, and plane info: Output model data corresponding to each data point
def getModelFromOrbit(data, o, normal, point):
    #data: the data that the orbit is being fit to
    #o: the test orbit that we are calculating the goodness-of-fit of
    #normal: the normal vector to the plane of the Great Circle we are estimating for the orbit
    #point: parameter for the axis generation of the Great Circle coordinates

    #initialize the orbit we are fitting --
    #we flip it around so that we are fitting both the forwards and the backwards orbit
    #except we are just integrating the orbit from the backwards end point twice as far in time.
    #This is because if you create back orbit and forwards orbit seperately they are not continuous
    #because galpy sucks and can't integrate backwards
    ts = np.linspace(0, t_length, num=resolution)*u.Gyr
    o_rev = o.flip()
    o_rev.integrate(ts, pot)

    #sign swap on vx because galpy is left-handed, and we are inputting data in a right-handed coordinate system
    data_orbit = Data(np.array(o.ll(ts)), np.array(o.bb(ts)), np.array(o.dist(ts)), np.array(o.vx(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array(o.vy(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array(o.vz(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array([]), np.array([]), np.array([]), np.array([]))
    data_orbit = data_orbit.LamBet(normal, point) #...
    data_orbit_rev = Data(np.array(o_rev.ll(ts)), np.array(o_rev.bb(ts)), np.array(o_rev.dist(ts)), np.array(o_rev.vx(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array(o_rev.vy(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array(o_rev.vz(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array([]), np.array([]), np.array([]), np.array([]))
    data_orbit_rev = data_orbit_rev.LamBet(normal, point) #...

    #grab full lists so that we can select the closest points once we get a list
    Lam = np.append(np.flip(data_orbit_rev.L), data_orbit.L)
    Bet = np.append(np.flip(data_orbit_rev.B), data_orbit.B)
    D = np.append(np.flip(data_orbit_rev.D), data_orbit.D)
    U = np.append(np.flip(data_orbit_rev.U), data_orbit.U)
    V = np.append(np.flip(data_orbit_rev.V), data_orbit.V)
    W = np.append(np.flip(data_orbit_rev.W), data_orbit.W)

    #get the list of points closest to each data point in Lambda -- harder than it sounds
    #only because galpy sucks and wasn't really built for this
    point_list = getPointList(data.L, Lam)

    #grab the model points from the point list we grabbed
    B_model = [Bet[p] for p in point_list]
    D_model = [D[p] for p in point_list]
    U_model = [U[p] for p in point_list]
    V_model = [V[p] for p in point_list]
    W_model = [W[p] for p in point_list]

    return B_model, D_model, U_model, V_model, W_model

#chi_squared: data, galpy.Orbit() --> float
#takes in the observed data and a test orbit and calculates the goodness-of-fit using a chi-squared method
def chiSquared(params, data=[], normal=(0, 0, 0), point=(1, 0, 0)):
    #data: the data that the orbit is being fit to
    #o: the test orbit that we are calculating the goodness-of-fit of
    #normal: the normal vector to the plane of the Great Circle we are estimating for the orbit
    #point: parameter for the axis generation of the Great Circle coordinates

    o = Orbit(vxvv=[params[0], params[1], params[2], params[3], params[4] - 220, params[5]], uvw=True, lb=True, ro=8., vo=220.) #generate the orbit
    o.integrate(ts, pot) #integrate the orbit

    B_model, D_model, U_model, V_model, W_model = getModelFromOrbit(data, o, normal, point) #get model data from orbit

    x2_B = sum(((B_model - data.B)/data.B_err)**2)
    x2_D = sum(((D_model - data.D)/data.D_err)**2)
    x2_U = sum(((U_model - data.U)/data.vel_err)**2)
    x2_V = sum(((V_model - data.V)/data.vel_err)**2)
    x2_W = sum(((W_model - data.W)/data.vel_err)**2)

    #get normalization factor
    N = len(data.L) #number of data points
    n = 5 #number of parameters
    eta = N - n - 1 #normalizing parameter
    if eta <= 0:
        eta = 1 #if you use fewer data points than needed to constrain the problem, then this will still work but it won't be normalized correctly

    x2 = (1/eta) * (x2_B + x2_D + x2_U + x2_V + x2_W) #Willett et al. 2009

    print('=================================')
    print('Optimizing:')
    print('Chi Squared: ' + str(x2))
    print('=================================')

    return x2

#testGC: data -> (float, float, float), (float, float, float), plot
#takes in data and plots it along the Great Circle it fit to the data
#This is so that you can look at the Great Circle before you optimize to ensure it is where you want it
def testGC(data):
    normal = gc.getPlaneNormal(gc.plane_OLS(data.x, data.y, data.z)) #get normal vector for fit Great Circle
    point = (1, 0, 0) #not currently fitting this, but this can be changed or fit at a later date
    #this way it makes 0 deg. in the Great Circle ~0 deg. in galactic longitude

    data = data.LamBet(normal, point)

    #-220 in vy puts galactic frame into solar frame
    #o = Orbit(vxvv=[300, 67.92, 12.45, 68.76, 102.12 - 220, 215.31], uvw=True, lb=True, ro=8, vo=220.) #generate the orbit (Cocytos)
    o = Orbit(vxvv=[294, 59.98, 14.57, -44.1, -63.3 - 220, 145], uvw=True, lb=True, ro=8, vo=220.) #generate the orbit #VRS
    o.integrate(ts, pot) #integrate the orbit
    o_rev = o.flip() #generate the orbit
    o_rev.integrate(ts, pot) #integrate the orbit

    x2 = chiSquared([300, 67.92, 12.45, 68.76, 102.12, 215.31], data=data, normal=normal, point=point)
    print('x2 = ' + str(x2))

    #turn the orbit into Lambda/Beta Frame
    #sign swap on vx because galpy is left-handed, and we are inputting data in a right-handed coordinate system
    data_orbit = Data(np.array(o.ll(ts)), np.array(o.bb(ts)), np.array(o.dist(ts)), np.array(o.vx(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array(o.vy(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array(o.vz(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array([]), np.array([]), np.array([]), np.array([]))
    data_orbit = data_orbit.LamBet(normal, point) #...
    data_orbit_rev = Data(np.array(o_rev.ll(ts)), np.array(o_rev.bb(ts)), np.array(o_rev.dist(ts)), np.array(o_rev.vx(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array(o_rev.vy(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array(o_rev.vz(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array([]), np.array([]), np.array([]), np.array([]))
    data_orbit_rev = data_orbit_rev.LamBet(normal, point) #...

    fig = plt.figure(figsize=(18, 12))
    fig.add_subplot(311)

    plt.scatter(data_opt.L, data_opt.B, s=100, c='b')
    plt.plot([0,360], [0, 0], 'k') #shows great circle on plot for reference
    plt.scatter(data_orbit.L, data_orbit.B, s=25, c='r')
    plt.scatter(data_orbit_rev.L, data_orbit_rev.B, s=25, c='g')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    plt.xlabel('$\Lambda$')
    plt.ylabel('$\\beta$')

    fig.add_subplot(312)

    plt.scatter(data_opt.l, data_opt.b, s=100, c='b')
    plt.plot([0,360], [0, 0], 'k') #shows great circle on plot for reference
    plt.scatter(data_orbit.l, data_orbit.b, s=25, c='r')
    plt.scatter(data_orbit_rev.l, data_orbit_rev.b, s=25, c='g')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    plt.xlabel('l')
    plt.ylabel('b')

    fig.add_subplot(313)

    plt.scatter(data_opt.l, data_opt.d, s=100, c='b')
    plt.plot([0,360], [0, 0], 'k') #shows great circle on plot for reference
    plt.scatter(data_orbit.l, data_orbit.d, s=25, c='r')
    plt.scatter(data_orbit_rev.l, data_orbit_rev.d, s=25, c='g')
    plt.xlim(0, 360)
    plt.ylim(0, 50)
    plt.xlabel('l')
    plt.ylabel('dist (helio)')

    plt.show()

    return normal, point

'''
==============================
INPUTS
==============================
'''
#guess_params = [275, 65, 6, 0, 0, 0] #l, b, dist (helio), U, V, W, group d
guess_params = [293.98, 59.98, 14.57, -44.06, -63.30, 144.96] #l, b, dist (helio), U, V, W, perp stream
#guess_params = [300, 67.92, 12.45, 75, 100, 200] #cocytos

#=============================
#GET DATA
#=============================
f = '/home/thomas/Desktop/duffau_data/all_duffau_data.csv'
f = open(f, 'r')
header = f.readline()
header = header.strip().split(',')

list_of_stars = {}
for line in f:
    star = {}
    line = line.strip().split(',')
    i = 0
    star[header[i]] = int(line[i]) #has to be an int otherwise it doens't match properly with the id_list. Shame it's hardcoded but
    i += 1
    while i < len(header) - 1:
        star[header[i]] = float(line[i])
        i += 1
    star[header[i]] = line[i] #unfortunate that it had to be hard coded but only the last value is not a float

    #we don't want the MSTO stars
    if star['source'] != 'sdss_lb_300_55' and star['source'] != 'sdss_lb_288_62':
        list_of_stars[line[0]] = star

    star['x'] = star['dist'] * np.cos(star['l'] * np.pi/180) * np.cos(star['b'] * np.pi/180) - 8
    star['y'] = star['dist'] * np.sin(star['l'] * np.pi/180) * np.cos(star['b'] * np.pi/180)
    star['z'] = star['dist'] * np.sin(star['b'] * np.pi/180)

    star['drdt'] = (star['x']*star['U'] + star['y']*star['V'] + star['z']*star['W'])/((star['x']**2 + star['y']**2 + star['z']**2)**0.5)
    star['rot_spd'] = (star['x']*star['V'] - star['y']*star['U'])/(star['x']**2 + star['y']**2)**0.5
    star['r_vel'] = (star['U']**2 + star['V']**2 + star['W']**2)**0.5

    star['lx'] = star['y'] * star['W'] - star['z'] * star['V']
    star['ly'] = star['x'] * star['W'] - star['z'] * star['U']
    star['lz'] = star['x'] * star['V'] - star['y'] * star['U']
    star['lperp'] = (star['lx']**2 + star['ly']**2)**0.5
    star['L'] = (star['lx']**2 + star['ly']**2 + star['lz']**2)**0.5

    PE = galpy.potential.evaluatePotentials(pot, (star['x']**2 + star['y']**2)**0.5 * u.kpc, star['z']*u.kpc, ro=8.*u.kpc, vo=220.*u.km/u.s)
    KE = 0.5*(star['U']**2 + star['V']**2 + star['W']**2)
    star['energy'] = PE + KE

#need list of id's that are in accepted groups (these are all the stars in groups a, b, d, f, and h that we determined were significant)
id_list = [3600853752939394816, 3698509244686084608, 3694689339428967552, 3696636437082881280, 3693851167971369600, 3696347712201213568, 3696397774340083840, 3697194061277159168, 3695996796192922752,
3601648424967929344, 3598360094928086528, 3597997015572148736, 3698057418423030656, 3702875134777825408, 3689603410955212544, 3694242181793912832, 3693284507166557952, 3688698130633240832, 3711384358344137600,
3799382220989581696, 3597170492066667776, 3681313986635168512, 3689637358376778752, 3697745260199152512, 3682043993636579712, 3702672648544372352, 3688562413961811456,
3797093488752748032, 3813285962135198336, 3891787828184023168, 3695934570706802944]

n_test = 0

#data_temp = [list_of_stars[x] for x in list_of_stars if (list_of_stars[x]['source'] == 'duffau_group_a' or list_of_stars[x]['source'] == 'duffau_group_b') and list_of_stars[x]['id'] in id_list]
data_temp = [list_of_stars[x] for x in list_of_stars if (list_of_stars[x]['source'] == 'duffau_group_f' or list_of_stars[x]['source'] == 'duffau_group_h' or list_of_stars[x]['source'] == 'duffau_group_a' or list_of_stars[x]['source'] == 'duffau_group_b') and list_of_stars[x]['id'] in id_list and list_of_stars[x]['lx'] > -1000]

l = []
b = []
d = []
U = []
V = []
W = []
l_err = []
b_err = []
d_err = []
vel_err = []
for x in data_temp:
    l.append(x['l'])
    b.append(x['b'])
    d.append(x['dist'])
    U.append(x['U'])
    V.append(x['V'])
    W.append(x['W'])
    l_err.append(2)
    b_err.append(2)
    d_err.append(0.25)
    vel_err.append(15)

data_opt = Data(np.array(l), np.array(b), np.array(d), np.array(U), np.array(V), np.array(W), np.array(l_err), np.array(b_err), np.array(d_err), np.array(vel_err))
#=============================

#optimize: data -> [float, float, float, float, float], (float, float, float), (float, float, float)
#takes in data, then fits a Great Circle to that data and minimizes the chi_squared to fit an orbit to the data
def optimize(data_opt):
    normal = gc.getPlaneNormal(gc.plane_OLS(data_opt.x, data_opt.y, data_opt.z)) #get normal vector for fit Great Circle
    point = (1, 0, 0) #not currently fitting this, but this can be changed or fit at a later date
    #this way it makes 0 deg. in the Great Circle ~0 deg. in galactic longitude

    data_opt = data_opt.LamBet(normal, point)

    '''
    ===================================================
    TOGGLE MINIMIZATION OR DIFFERENTIAL EVOLUTION BELOW
    ===================================================
    '''
    ''' <-- toggle switch, add or remove # before block quotes
    #this uses "gradient-descent-like" algorithms, faster but prone to bugs involving bumpy likelihood surfaces
    params = scopt.minimize(chiSquared, guess_params, args=(data_opt, normal, point), bounds=[(0, 360), (-90, 90), (0, 50), (-1000, 1000), (-1000, 1000), (-1000, 1000)]).x
    '''
    #this uses differential evolution, much slower but better with bumpy likelihood surfaces
    #parameters for differential optimization
    max_it = 20
    pop_size = 50 #10 times number of parameters
    diff_scaling_factor = 0.8
    crossover_rate = 0.9
    bounds=[(280, 310), (55, 70), (0, 50), (-1000, 1000), (-1000, 1000), (-1000, 1000)]

    params = scopt.differential_evolution(chiSquared, bounds, args=(data_opt, normal, point), strategy='rand1bin', maxiter=max_it, popsize=pop_size, mutation=diff_scaling_factor, recombination=crossover_rate).x
    #'''

    x2 = chiSquared(params, data_opt, normal, point)

    #plotting, because plotting is good for you
    #==============================================

    #temp for plotting for poster
    t_length = 0.3 #Gyr
    resolution = 5000
    ts = np.linspace(0, t_length, num=resolution)*u.Gyr

    #orbit manipulation junk
    o = Orbit(vxvv=[params[0], params[1], params[2], params[3], params[4] - 220, params[5]], uvw=True, lb=True, ro=8, vo=220.) #generate the orbit
    o.integrate(ts, pot) #integrate the orbit
    o_rev = o.flip() #generate the orbit
    o_rev.integrate(ts, pot) #integrate the orbit

    #turn the orbit into Lambda/Beta Frame
    #sign swap on vx because galpy is left-handed, and we are inputting data in a right-handed coordinate system
    data_orbit = Data(np.array(o.ll(ts)), np.array(o.bb(ts)), np.array(o.dist(ts)), np.array(o.vx(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array(o.vy(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array(o.vz(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array([]), np.array([]), np.array([]), np.array([]))
    data_orbit = data_orbit.LamBet(normal, point) #...
    data_orbit_rev = Data(np.array(o_rev.ll(ts)), np.array(o_rev.bb(ts)), np.array(o_rev.dist(ts)), np.array(o_rev.vx(ts, obs=[8., 0., 0., 0., 0., 0.]))*-1, np.array(o_rev.vy(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array(o_rev.vz(ts, obs=[8., 0., 0., 0., 0., 0.])), np.array([]), np.array([]), np.array([]), np.array([]))
    data_orbit_rev = data_orbit_rev.LamBet(normal, point) #...

    fig = plt.figure(figsize=(16, 4))
    fig.add_subplot(121)

    plt.scatter(data_opt.L, data_opt.B, s=100, c='b')
    plt.plot([0,360], [0, 0], 'k') #shows great circle on plot for reference
    plt.scatter(data_orbit.L, data_orbit.B, s=25, c='r')
    plt.scatter(data_orbit_rev.L, data_orbit_rev.B, s=25, c='g')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    plt.xlabel('$\Lambda$', fontsize=20)
    plt.ylabel('$\\beta$', fontsize=20)
    plt.scatter([1000], [1000], c='b', label='Data Points')
    plt.plot([1000, 2000], [1000, 2000], c='k', markersize=20, label='Orbit Fit')
    plt.legend(loc='upper left')

    fig.add_subplot(122)

    plt.scatter(data_opt.l, data_opt.b, s=100, c='b')
    plt.plot([0,360], [0, 0], 'k') #shows great circle on plot for reference
    plt.scatter(data_orbit.l, data_orbit.b, s=25, c='r')
    plt.scatter(data_orbit_rev.l, data_orbit_rev.b, s=25, c='g')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    plt.xlabel('l', fontsize=20)
    plt.ylabel('b', fontsize=20)



    plt.tight_layout()
    plt.savefig('/home/thomas/Desktop/python/python_figures/lambdabeta.eps')

    fig = plt.figure(figsize=(17, 17))
    fig.add_subplot(221)

    plt.scatter(data_opt.l, data_opt.b, s=100, c='b')
    plt.scatter(data_orbit.l, data_orbit.b, s=25, c='r')
    plt.scatter(data_orbit_rev.l, data_orbit_rev.b, s=25, c='g')
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    plt.xlabel('l')
    plt.ylabel('b')


    fig.add_subplot(222)

    plt.scatter(data_opt.l, data_opt.d, s=100, c='b')
    plt.scatter(data_orbit.l, data_orbit.d, s=25, c='r')
    plt.scatter(data_orbit_rev.l, data_orbit_rev.d, s=25, c='g')
    plt.xlim(0, 360)
    plt.ylim(0, 50)
    plt.xlabel('l')
    plt.ylabel('dist (helio)')


    fig.add_subplot(223)

    plt.scatter(data_opt.U, data_opt.V, s=100, c='b')
    plt.scatter(data_orbit.U, data_orbit.V, s=25, c='r')
    plt.scatter(data_orbit_rev.U*-1, data_orbit_rev.V*-1, s=25, c='g')
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.xlabel('U')
    plt.ylabel('V')


    fig.add_subplot(224)

    plt.scatter(data_opt.U, data_opt.W, s=100, c='b')
    plt.scatter(data_orbit.U, data_orbit.W, s=25, c='r')
    plt.scatter(data_orbit_rev.U*-1, data_orbit_rev.W*-1, s=25, c='g')
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.xlabel('U')
    plt.ylabel('W')


    plt.tight_layout()

    plt.savefig('/home/thomas/Desktop/python/python_figures/orbit_fit' + str(n_test) + '.eps')


    c_data = SkyCoord(l=data_opt.l, b=data_opt.b, frame='galactic', unit='deg')
    c_data_icrs = c_data.transform_to('icrs')
    c_orbit = SkyCoord(l=data_orbit.l, b=data_orbit.b, frame='galactic', unit='deg')
    c_orbit_icrs = c_orbit.transform_to('icrs')
    c_orbit_rev = SkyCoord(l=data_orbit.l, b=data_orbit.b, frame='galactic', unit='deg')
    c_orbit_rev_icrs = c_orbit_rev.transform_to('icrs')


    fig = fig = plt.figure(figsize=(17, 8))
    fig.add_subplot(211)

    plt.scatter(c_data_icrs.ra.degree, c_data_icrs.dec.degree, s=100, c='b')
    plt.scatter(c_orbit_icrs.ra.degree, c_orbit_icrs.dec.degree, s=25, c='r')
    plt.scatter(c_orbit_rev_icrs.ra.degree, c_orbit_rev_icrs.dec.degree, s=25, c='g')
    plt.xlim(360, 0)
    plt.ylim(-90, 90)
    plt.xlabel('ra')
    plt.ylabel('dec')


    fig.add_subplot(212)

    plt.scatter(c_data_icrs.ra.degree, data_opt.d, s=100, c='b')
    plt.scatter(c_orbit_icrs.ra.degree, data_orbit.d, s=25, c='r')
    plt.scatter(c_orbit_rev_icrs.ra.degree, data_orbit_rev.d, s=25, c='g')
    plt.xlim(360, 0)
    plt.ylim(0, 50)
    plt.xlabel('l')
    plt.ylabel('dist (helio)')

    plt.savefig('/home/thomas/Desktop/python/python_figures/orbit_fit_icrs' + str(n_test) + '.eps')

    #===============================================

    return params, normal, point, x2

'''
==============================
'''
'''
#testing
normal, point = testGC(data_opt)

print('===================================')
print('Normal Vector:')
print(normal)
print()
print('Point Vector')
print(point)
print('===================================')

'''
#optimization
params, normal, point, x2 = optimize(data_opt)

print('===================================')
print('Params: l, b, d, U, V, W')
print(params)
print()
print('Normal Vector:')
print(normal)
print()
print('Point Vector:')
print(point)
print()
print('Chi Squared:')
print(x2)
print('===================================')
