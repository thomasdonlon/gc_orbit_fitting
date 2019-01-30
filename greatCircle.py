#============================
#Tom Donlon, 2018
#============================

import math as ma
import numpy as np
from numpy import linalg
import scipy as sc

#Below is antiquated version of plane fitter
'''
#adopted from NewbyTools
def plane_OLS(x,y,z):
    """ Solves for the best-fit plane to a set of x,y,z data using ordinary
        least-squares.  Equation is of form z = Ax + By + C.
        Output is normalized a,b,c,d of a plane of the form ax+by+cz+d=0"""
    sum_x, sum_y, sum_z = np.sum(x), np.sum(y), np.sum(z)
    sum_xx, sum_yy, sum_n = np.sum(x*x), np.sum(y*y), len(x)
    sum_xy, sum_xz, sum_yz = np.sum(x*y), np.sum(x*z), np.sum(y*z)
    #  Will solve Ax=B, x is solution parameters (called p below)
    A = np.matrix([ [sum_xx, sum_xy, sum_x],
                    [sum_xy, sum_yy, sum_y],
                    [sum_x,  sum_y,  sum_n] ])
    B = np.matrix([sum_xz, sum_yz, sum_z])
    #print LA.inv(A)
    #print B.T
    p = linalg.inv(A)*B.T
    params = [-float(p[0]), -float(p[1]), 1.0, -float(p[2])]  #c=1.0 by default, minuses due to OLS definition
    bottom = np.sqrt(params[0]*params[0] + params[1]*params[1] + params[2]*params[2])
    for i in range(len(params)):  params[i] = params[i]/bottom
    print("# - Normalized best-fit plane parameters: {0}".format(params))
    return params
'''

#x, y, z should be numpy arrays
#takes in galactocentric x, y, z data and outputs parameters for the best fit plane to those points incident the galactic center
def plane_OLS(x,y,z):
    """ Solves for the best-fit plane to a set of x,y,z data using ordinaryleast-squares.
        Equation is of form z = Ax + By.
        DIFFERENT FROM NEWBY TOOLS plane_OLS() IN THAT WE CONSTRAIN THE PLANE THROUGH GALACTIC CENTER
        Output is normalized a,b,c of a plane of the form ax+by+cz=0"""
    A = np.array([x, y]).T
    B = z.T

    #solve Ax=B
    p = np.matmul(np.matmul(linalg.inv(np.matmul(A.T, A)), A.T), B.T) #uses left pseudo-inverse {(A^T * A)^-1 * A^T} due to system being overconstrained (A doesn't have a true inverse)
    params = [-float(p[0]), -float(p[1]), 1.0]  #c=1.0 by default
    bottom = np.sqrt(params[0]*params[0] + params[1]*params[1] + params[2]*params[2])
    for i in range(len(params)):  params[i] = params[i]/bottom
    print("# - Normalized best-fit plane parameters: {0}".format(params))
    for i in range(len(x)):
        print(plane_dist(x[i], y[i], z[i], params))
    return params

#borrowed from NewbyTools
def plane_dist(x,y,z, params):
    a,b,c = params
    return (a*x + b*y + c*z)

#getPlaneNormal: [float, float, float, float] --> np.array([float, float, float])
#takes in parameters that define a plane in 3D and returns a normalized normal vector to that plane
def getPlaneNormal(params):
    #params ([a, b, c, d]) corresponding to the equation for a plane ax + by + cz + d = 0
    #comes from the plan fitting method above

    #definition of a normal vector, given equation of a plane
    normal = np.array([params[0], params[1], params[2]])

    #normalize the normal vector
    len_normal = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
    normal = (normal[0]/len_normal, normal[1]/len_normal, normal[2]/len_normal)

    return normal

#gal2plane: np.array(floats), np.array(floats), np.array(floats), (float, float, float), (float, float, float) --> np.array(floats), np.array(floats), np.array(floats)
#takes in galactic coordinates for a star(s) and returns their x,y,z coordinates with respect to a rotated plane with the normal vector provided
#Newby 2013 et al, appendix
def gal2plane(x,y,z, normal=(0,0,0), point=(1,0,0)):
    #x, y, z: galactocentric x, y, z coordinates
    #normal: 3-vector providing the orientation of the plane to rotate into
    #point: 3-vector 'suggesting' the direction of the new x-axis

    #ensure that normal and point are normalized
    len_normal = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
    if len_normal != 1.0:
        normal = (normal[0]/len_normal, normal[1]/len_normal, normal[2]/len_normal)
    len_point = (point[0]**2 + point[1]**2 + point[2]**2)**0.5
    if len_point != 1.0:
        point = (point[0]/len_point, point[1]/len_point, point[2]/len_point)

    #define new axes along the plane
    z_plane = np.array(normal)
    y_plane = np.cross(z_plane, np.array(point))
    x_plane = np.cross(y_plane, z_plane)

    #get new x, y, z through change of basis
    xyz = np.array([x, y, z])

    COB = linalg.inv(np.array([x_plane, y_plane, z_plane]).T)
    new_xyz = np.matmul(COB, xyz)

    return new_xyz[0], new_xyz[1], new_xyz[2]

#gal2plane: np.array(floats), np.array(floats), np.array(floats), (float, float, float), (float, float, float) --> np.array(floats), np.array(floats)
#takes in galactic coordinates for a star(s) and returns their Lamba, Beta coordinates with respect to a rotated plane with the normal vector provided
#Newby 2013 et al, appendix
def gal2LamBet(x,y,z, normal=(0,0,0), point=(1,0,0)):
    #x, y, z: galactocentric x, y, z coordinates
    #normal: 3-vector providing the orientation of the plane to rotate into
    #point: 3-vector 'suggesting' the direction of the new x-axis

    x_prime, y_prime, z_prime = gal2plane(x,y,z, normal=normal, point=point)
    Lam = np.arctan2(y_prime, x_prime)*180/np.pi #convert to degrees
    #correct Lam to be between 0 and 360 instead of -180 to 180
    i = 0
    while i < len(Lam):
        if Lam[i] < 0:
            Lam[i] += 360
        i += 1

    Bet = np.arcsin(z_prime/(x_prime**2 + y_prime**2 + z_prime**2)**0.5)*180/np.pi #convert to degrees
    return Lam, Bet
