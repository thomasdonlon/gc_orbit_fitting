import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

def getXYZ(dist, l, b):

    l_rad = l*(np.pi/180)
    b_rad = b*(np.pi/180)
    #'''
    #right-handed toward galactic center
    x = dist*np.cos(l_rad)*np.cos(b_rad) - 8
    y = dist*np.sin(l_rad)*np.cos(b_rad)
    z = dist * np.sin(b_rad)
    #'''
    '''
    #'right'-handed away from galactic center
    x = 8 - dist*np.cos(l_rad)*np.cos(b_rad)
    y = - dist*np.sin(l_rad)*np.cos(b_rad)
    z = dist*np.sin(b_rad)
    #'''

    return x, y, z

#input: distance [kpc], l [deg], b [deg] (spherical-ish coordinates)
#output: Cylindrical coordinates (R, [kpc], Z [kpc], Phi [deg])
#inputs should be numpy arrays
def getRZPHI(x, y, z):

    R = (x**2 + y**2)**0.5

    P_rad = R

    if x == 0 and y ==0:
        P_rad = 0
    elif x >= 0:
        P_rad = np.arcsin(y/R)
    elif x > 0:
        P_rad = np.arctan(y/x)
    else:
        P_rad = -np.arcsin(y/R) + np.pi

    Phi = P_rad * (180/np.pi)

    return R, z, Phi

def getRZPHI_list(x, y, z):

    R = (x**2 + y**2)**0.5

    P_rad = R.copy()

    i = 0
    while i < len(P_rad):
        if x[i] == 0 and y[i] ==0:
            P_rad[i] = 0
        elif x[i] >= 0:
            P_rad[i] = np.arcsin(y[i]/R[i])
        elif x[i] > 0:
            P_rad[i] = np.arctan(y[i]/x[i])
        else:
            P_rad[i] = -np.arcsin(y[i]/R[i]) + np.pi
        i += 1

    Phi = P_rad * (180/np.pi)

    return R, z, Phi
#takes in solar-centered cylindrical coordinates and outputs galactic-centered cylindrical coordinates
#assumes the sun is ~8kpc from the galactic center
#ra of galactic center = 17h 45m 40.04s (266.417deg), dec = -29deg 00' 28.1" (-29.0078deg)
def SoltoGalRZPhi(Rsol,Zsol,Psol):
    #solar cylindrical -> solar x,y,z
    #solar x,y,z -> galactic x,y,z
    #galactic x,y,z -> galactic cylindrical

    Psol_rad = Psol*(np.pi/180)

    Xsol = Rsol*np.cos(Psol_rad)
    Ysol = Rsol*np.sin(Psol_rad)
    #note: geocentric
    #need to make x,y,z heliocentric

    '''
    #should center the galactic coordinates correctly
    Xgal = Xsol + 8*np.cos(266.417*np.pi/180)#check units -- is in kpc
    Ygal = Ysol + 8*np.sin(266.417*np.pi/180)
    Zgal = Zsol - 8*np.tan(29.0078*np.pi/180)'''

    Rot = np.array([[-0.05465441196, .9980453361, 0.03030515501],
                    [-.8728442916, -0.06249405705, .4839807182],
                    [.4849285906, 0, .8745537502]], dtype=np.float32)

    sol_cart = np.array([Xsol,Ysol,Zsol])
    Xgal, Ygal, Zgal = np.dot(Rot, sol_cart)

    #now adjust for origin (still geocentric currently)
    Xgal += 8

    Rgal = (Xgal**2+Ygal**2)**0.5
    Pgal_rad = Psol_rad
    i = 0
    while i < len(Pgal_rad):
        x = Xgal[i]
        y = Ygal[i]
        if x == 0 and y ==0:
            Pgal_rad[i] = 0
        elif x >= 0:
            Pgal_rad[i] = np.arcsin(y/Rgal[i])
        elif x > 0:
            Pgal_rad[i] = np.arctan(y/x)
        else:
            Pgal_rad[i] = -np.arcsin(y/Rgal[i]) + np.pi
        i += 1

    Pgal = Pgal_rad * (180/np.pi)

    return Rgal, Zgal, Pgal

# Input: distance [kpc], radial velocity [km/s], RA/DEC [degrees], and pmRA/pmDEC [mas/yr]
# Returns: Galactic U,V,W velocities [km/s]
# NOTE: pmRA = d/dt(RA) * cos(DEC)
# Arguments should be numpy arrays for most efficient usage
def getUVW(dist, rv, ra, dec, pmra, pmde):

    # Conversion from Equatorial (J2000) Cartesian to Galactic Cartesian
    EQ2GC = np.array( [[-0.05487572,-0.87343729, -0.48383453],
                      [ 0.49410871, -0.44482923,  0.7469821 ],
                      [-0.86766654, -0.19807649,  0.45598456]], dtype=np.float32 )

    ra_rad = ra * (np.pi/180.0)
    dec_rad = dec * (np.pi/180.0)

    sina = np.sin(ra_rad)
    cosa = np.cos(ra_rad)
    sind = np.sin(dec_rad)
    cosd = np.cos(dec_rad)

    vra =  4.741067035842384 * pmra * dist # / cosd
    vdec = 4.741067035842384 * pmde * dist

    vx_Eq = rv * cosd * cosa   -   vdec * sind * cosa   -   vra * sina # * cosd
    vy_Eq = rv * cosd * sina   -   vdec * sind * sina   +   vra * cosa # * cosd
    vz_Eq = rv * sind          +   vdec * cosd

    vel_Eq = np.array([ vx_Eq, vy_Eq, vz_Eq ])
    vx, vy, vz = np.dot(EQ2GC, vel_Eq)

    # Sun's velocity is (10.1, 224.0, 6.7)_GSR
    vx += 10.1
    vy += 224.0
    vz += 6.7

    return vx, vy, vz

def RZPHI_to_LB(r, z, phi):
    l = phi
    b = np.arcsin(z/r)
    b = b * (180/np.pi)
    return l, b

def approx_UVW_from_2lbr(l1,b1,r1,l2,b2,r2,vel_mag):
    x1, y1, z1 = getXYZ(r1, l1, b1)
    x2, y2, z2 = getXYZ(r2, l2, b2)
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1

    #normalize the velocities to obtain the direction
    # & multiply by arbitrary velocity
    un_norm_mag = (vx**2 + vy**2 + vz**2)**0.5

    vx *= vel_mag/un_norm_mag
    vy *= vel_mag/un_norm_mag
    vz *= vel_mag/un_norm_mag

    return vx, vy, vz

def approx_lsr(x, y):
    v_lsr = 220
    vx = v_lsr / (1 + (x/y)**2)**0.5
    vy = v_lsr / (1 + (y/x)**2)**0.5
    if x > 0:
        vy = -1 * vy
    if y < 0:
        vx = -1 * vx
    return vx, vy

#does not work for arrays!
def getrvpm(ra, dec, dist, U, V, W):
    k = 4.74057

    ra_rad = ra * np.pi/180
    dec_rad = dec * np.pi/180

    T = np.array([[-0.05487572,-0.87343729, -0.48383453], [0.49410871, -0.44482923,  0.7469821], [-0.86766654, -0.19807649,  0.45598456]])
    A = np.array([[np.cos(ra_rad)*np.cos(dec_rad), -1*np.sin(ra_rad), -1*np.cos(ra_rad)*np.sin(dec_rad)], [np.sin(ra_rad)*np.cos(dec_rad), np.cos(ra_rad), -1*np.sin(ra_rad)*np.sin(dec_rad)], [np.sin(dec_rad), 0, np.cos(dec_rad)]])
    B = np.matmul(T,A)
    B_inv = np.linalg.inv(B)

    uvw = np.array([[U], [V], [W]])
    rvpm = np.matmul(B_inv, uvw)

    rv = rvpm[0][0]
    pmra = rvpm[1][0]/dist/k
    pmdec = rvpm[2][0]/dist/k

    return rv, pmra, pmdec

#does not work for arrays!
def removeSolarMotionFromPM(ra, dec, dist, pmra, pmdec):
    rv, mura, mudec = getrvpm(ra, dec, dist, -10, -224, -7)

    pmra -= mura
    pmdec -= mudec

    return pmra, pmdec

#does not work for arrays!
def getUVWerrors(dist, ra, dec, pmra, pmdec, err_pmra, err_pmdec, err_rv, err_dist):
    #distance in pc
    k = 4.74057

    ra_rad = ra * np.pi/180
    dec_rad = dec * np.pi/180

    T = np.array([[-0.05487572,-0.87343729, -0.48383453], [0.49410871, -0.44482923,  0.7469821], [-0.86766654, -0.19807649,  0.45598456]])
    A = np.array([[np.cos(ra_rad)*np.cos(dec_rad), -1*np.sin(ra_rad), -1*np.cos(ra_rad)*np.sin(dec_rad)], [np.sin(ra_rad)*np.cos(dec_rad), np.cos(ra_rad), -1*np.sin(ra_rad)*np.sin(dec_rad)], [np.sin(dec_rad), 0, np.cos(dec_rad)]])
    B = np.matmul(T,A)
    C = B**2

    M = np.array([[err_rv**2],
                  [(k*dist)**2 * (err_pmra**2 + (pmra*err_dist/dist)**2)],
                  [(k*dist)**2 * (err_pmdec**2 + (pmdec*err_dist/dist)**2)]])
    N = 2*pmra*pmdec*k**2*err_dist**2*np.array([[B[0][1]*B[0][2]],
                                               [B[1][1]*B[1][2]],
                                               [B[2][1]*B[2][2]]])

    uvw_var = np.matmul(C,M) + N

    err_u = (uvw_var[0][0])**0.5
    err_v = (uvw_var[1][0])**0.5
    err_w = (uvw_var[2][0])**0.5

    return err_u, err_v, err_w
