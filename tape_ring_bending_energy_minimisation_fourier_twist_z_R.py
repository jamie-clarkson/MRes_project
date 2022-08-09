# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:59:07 2021

@author: Jamie

This looks at tape ring bending, where rotations are applied to the boundaries. 
The tape ring does not fold, but the sides twist and the middle forms a v shape

This version is fully numeric, so no manual differentiation required
"""

#importing required libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch


#finite element results for plotting in comparison
phi_1_FE = [0 ,-0.0505 ,-0.084	,-0.103	,-0.114	,-0.122	,-0.128	,-0.135	,-0.142	,-0.151	,-0.161	,-0.175	,-0.191	,-0.207	,-0.227	,-0.248	,-0.267	,-0.285	,-0.305	,-0.322	,-0.34	]
z_1_FE = [0,1.63,2.85,3.69,4.27,4.71,5.01,5.3,5.58,5.92,6.28,6.75,7.28,7.85,8.51,9.21,9.91,10.61,11.31,11.91,12.61]

#defining some 3D arrows for plotting if desired
from matplotlib.text import Annotation
class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)


#defining geometric and material parameters
N = 100

r = 6e-3
alpha = np.pi/2
b = np.pi*r/2
R0 = 51.5e-3
t = 0.15e-3
E = 120e9
v = 0.34
E_y = 120e9
D = E_y*t**3/(12*(1-v**2))


def curvature_roots(E,F,G,L,M,N):
    """
    Function for calculating principal curvatures from the coefficients of the fundamental forms

    Parameters
    ----------
    E : Float
        Coefficient of first fundemental form.
    F : Float
        Coefficient of first fundemental form.
    G : Float
        Coefficient of first fundemental form.
    L : Float
        Coefficient of second fundemental form.
    M : Float
        Coefficient of second fundemental form.
    N : Float
        Coefficient of second fundemental form.

    Returns
    -------
    k1 : Float
        Principal curvature 1.
    k2 : Float
        Principal curvature 2.
    psi_1 : Float
        principal curvature 1 direction [dtheta,dpsi] if dtheta=1.
    psi_2 : Float
        principal curvature 2 direction [dtheta,dpsi] if dtheta=1.

    """
    
    a = E*G-F**2
    b = 2*M*F-L*G-N*E
    c = L*N-M**2
    k1 = (-b+(b**2-4*a*c)**0.5)/(2*a)
    k2 = (-b-(b**2-4*a*c)**0.5)/(2*a)
    psi_1 = -(L-k1*E)/(M-k1*F)
    psi_2 = -(L-k2*E)/(M-k2*F)
    return k1,k2,psi_1,psi_2

 
def curvature_original_directions(E,F,G,L,M,N):
    """
    Function for calculating curvature in the directions of the original principal curvatures 

    Parameters
    ----------
    E : Float
        Coefficient of first fundemental form.
    F : Float
        Coefficient of first fundemental form.
    G : Float
        Coefficient of first fundemental form.
    L : Float
        Coefficient of second fundemental form.
    M : Float
        Coefficient of second fundemental form.
    N : Float
        Coefficient of second fundemental form.

    Returns
    -------
    kx : Float
        Curvature in direction of original principal curvature 1.
    ky : Float
        Curvature in direction of original principal curvature 2.

    """
    kx = L/E  #theta direction
    ky = N/G  #psi direction
    return kx,ky


#global varible to define where the tape ring surface twists about
global prop_down
prop_down = 1


def length(params,phi_0):
    """
    Function for calculating length of theta = 0 line

    Parameters
    ----------
    params : Array of floats
        The amplitudes of the different components of phi, z , R.
    phi_0 : Float
        Applied tape ring rotation.

    Returns
    -------
    dif : Float
        Difference in length from original length.

    """
    fourier_amps_twist = [params[0],params[1],params[2],params[3]]
    fourier_amps_z = [params[4],params[5],params[6],params[7]]
    fourier_amps_R =[params[8],params[9],params[10],params[11]]
    
    alpha = np.pi/2
    theta = 0
    psi  = np.linspace(0,np.pi/2,1000)
    
    
    
    twist = twist_fourier(fourier_amps_twist, psi, phi_0)
    
    z_offset = z_fourier(fourier_amps_z,psi)
    R = R_fourier(fourier_amps_R,psi,R0)
    
    """ New method where section twists around middle of arc"""
    x = (R -r*prop_down*np.sin(twist)+ r*np.sin(theta+twist))*np.cos(psi)
    y = (R -r*prop_down*np.sin(twist) + r*np.sin(theta+twist))*np.sin(psi)
    z = r*np.cos(theta+twist) -r*prop_down*np.cos(twist) + z_offset
    
    dx = np.gradient(x,edge_order =2)
    dy = np.gradient(y,edge_order =2)
    dz = np.gradient(z,edge_order =2)
    dist_long = (dx**2 +dy**2 +dz**2)**0.5   #nodal distances along the psi direction
    dist = np.sum(dist_long)
    dif = np.abs(dist - (R0+r*np.sin(theta))*np.pi/2)
    return dif


def length_in(params,phi_0):
    """
    Function for calculating length of theta = -alpha/2 line

    Parameters
    ----------
    params : Array of floats
        The amplitudes of the different components of phi, z , R.
    phi_0 : Float
        Applied tape ring rotation.

    Returns
    -------
    dif : Float
        Difference in length from original length.

    """    
    fourier_amps_twist = [params[0],params[1],params[2],params[3]]
    fourier_amps_z = [params[4],params[5],params[6],params[7]]
    fourier_amps_R =[params[8],params[9],params[10],params[11]]
    
    alpha = np.pi/2
    theta = -alpha/2
    psi  = np.linspace(0,np.pi/2,1000)
    
    
    twist = twist_fourier(fourier_amps_twist, psi, phi_0)
    
    z_offset = z_fourier(fourier_amps_z,psi)
    R = R_fourier(fourier_amps_R,psi,R0)
    
    """ New method where section twists around middle of arc"""
    
    x = (R -r*prop_down*np.sin(twist)+ r*np.sin(theta+twist))*np.cos(psi)
    y = (R -r*prop_down*np.sin(twist) + r*np.sin(theta+twist))*np.sin(psi)
    z = r*np.cos(theta+twist) -r*prop_down*np.cos(twist) + z_offset
    
    global dx,dy,dz
    dx = np.gradient(x,edge_order =2)
    dy = np.gradient(y,edge_order =2)
    dz = np.gradient(z,edge_order =2)
    
    dist_long = (dx**2 +dy**2 +dz**2)**0.5   #nodal distances along the psi direction
    dist = np.sum(dist_long)
    dif = np.abs(dist - (R0+r*np.sin(theta))*np.pi/2)
    return dif



def length_out(params,phi_0):
    """
    Function for calculating length of theta = +alpha/2 line

    Parameters
    ----------
    params : Array of floats
        The amplitudes of the different components of phi, z , R.
    phi_0 : Float
        Applied tape ring rotation.

    Returns
    -------
    dif : Float
        Difference in length from original length.

    """    
    fourier_amps_twist = [params[0],params[1],params[2],params[3]]
    fourier_amps_z = [params[4],params[5],params[6],params[7]]
    fourier_amps_R =[params[8],params[9],params[10],params[11]]
    
    alpha = np.pi/2
    theta = alpha/2
    psi  = np.linspace(0,np.pi/2,1000)
    
    
    
    twist = twist_fourier(fourier_amps_twist, psi, phi_0)
    
    z_offset = z_fourier(fourier_amps_z,psi)
    R = R_fourier(fourier_amps_R,psi,R0)
    
    """ New method where section twists around middle of arc"""
    x = (R -r*prop_down*np.sin(twist)+ r*np.sin(theta+twist))*np.cos(psi)
    y = (R -r*prop_down*np.sin(twist) + r*np.sin(theta+twist))*np.sin(psi)
    z = r*np.cos(theta+twist) -r*prop_down*np.cos(twist) + z_offset
    
    dx = np.gradient(x,edge_order =2)
    dy = np.gradient(y,edge_order =2)
    dz = np.gradient(z,edge_order =2)
    dist_long = (dx**2 +dy**2 +dz**2)**0.5   #nodal distances along the psi direction
    dist = np.sum(dist_long)
    dif = np.abs(dist - (R0+r*np.sin(theta))*np.pi/2)
    return dif


def twist_fourier(fourier_amps,psi,phi_0):
    """
    Gives twist as a function of the psi and mode amplitudes

    Parameters
    ----------
    fourier_amps : Array of floats
        The amplitudes of the different components of phi.
    psi : Float
        Psi value on tape ring.
    phi_0 : Float
        Applied tape ring rotation.

    Returns
    -------
    twist : Float
        Twist phi for given psi location.

    """
    twist = phi_0
    for n in range(0,len(fourier_amps)):
        a_n = fourier_amps[n]
        twist = twist + a_n*(np.cos(n*2*psi)-1)
    twist = phi_0 + fourier_amps[0]*np.sin(psi)**2 + fourier_amps[1]*(np.cos(psi)**2-1)+ fourier_amps[2]*np.sin(2*psi)**2 + fourier_amps[3]*(np.cos(2*psi)**2-1)
    return twist


def z_fourier(fourier_amps,psi):
    """
    Gives Z as a function of the psi and mode amplitudes  

    Parameters
    ----------
    fourier_amps : Array of floats
        The amplitudes of the different components of z.
    psi : Float
        Psi value on tape ring.

    Returns
    -------
    z : Float
        Vertical offset z for given psi location.

    """    
    z = fourier_amps[0]*np.sin(psi)**2 + fourier_amps[1]*(np.cos(psi)**2-1) + fourier_amps[2]*np.sin(2*psi)**2 + fourier_amps[3]*(np.cos(2*psi)**2-1)
    return z


def R_fourier(fourier_amps,psi,R0):
    """
    Gives R as a function of the psi and mode amplitudes

    Parameters
    ----------
    fourier_amps : Array of floats
        The amplitudes of the different components of R.
    psi : Float
        Psi value on tape ring.
    R0 : Float
        Original tape spring major radius R.

    Returns
    -------
    R : Float
        Major radius R for given psi location.

    """
    R = R0 - fourier_amps[0]*np.cos(psi)**4  #+ fourier_amps[1]*np.cos(psi)**4 #+ 0*fourier_amps[1]*np.sin(psi)**2 + fourier_amps[1]*np.cos(2*psi)**2 #+ fourier_amps[3]*np.sin(2*psi)**2 
    return R


def bending_energy_opt(params,phi_0):
    """
    Function for calculating the strain energy in the deformed tape ring, which is then minimised

    Parameters
    ----------
    params : Array of floats
        The amplitudes of the different components of phi, z , R.
    phi_0 : Float
        Applied tape ring rotation.

    Returns
    -------
    total_energy : Float
        Total strain energy in tape ring for given amplitude parameters.

    """

    fourier_amps_twist = [params[0],params[1],params[2],params[3]]
    fourier_amps_z = [params[4],params[5],params[6],params[7]]
    fourier_amps_R =[params[8],params[9],params[10],params[11]]
    
    alpha = np.pi/2
    theta = np.linspace(-alpha/2,alpha/2,100)
    psi  = np.linspace(0,np.pi/2,1000)
    psi,theta = np.meshgrid(psi,theta)
    
    
    twist = twist_fourier(fourier_amps_twist, psi, phi_0)
    
    z_offset = z_fourier(fourier_amps_z,psi)
    R = R_fourier(fourier_amps_R,psi,R0)
    
     
    x0 = (R + r*np.sin(theta))*np.cos(psi)
    y0 = (R + r*np.sin(theta))*np.sin(psi)
    z0 = r*np.cos(theta) 
    
    dx0 = np.gradient(x0,axis=1,edge_order =2)
    dy0 = np.gradient(y0,axis=1,edge_order =2)
    dz0 = np.gradient(z0,axis=1,edge_order =2)
    dist0_long = (dx0**2 +dy0**2 +dz0**2)**0.5
    
    dx0 = np.gradient(x0,axis=0,edge_order =2)
    dy0 = np.gradient(y0,axis=0,edge_order =2)
    dz0 = np.gradient(z0,axis=0,edge_order =2)
    dist0_trans = (dx0**2 +dy0**2 +dz0**2)**0.5
    
    """ Old method where section twists around middle of circle """
    # x = (R + r*np.sin(theta+twist))*np.cos(psi)
    # y = (R + r*np.sin(theta+twist))*np.sin(psi)
    # z = r*np.cos(theta+twist) + z_offset
    
    """ New method where section twists around middle of arc"""
    x = (R -r*prop_down*np.sin(twist)+ r*np.sin(theta+twist))*np.cos(psi)
    y = (R -r*prop_down*np.sin(twist) + r*np.sin(theta+twist))*np.sin(psi)
    z = r*np.cos(theta+twist) -r*prop_down*np.cos(twist) + z_offset
    
    
    dx = np.gradient(x,axis=1,edge_order =2)
    dy = np.gradient(y,axis=1,edge_order =2)
    dz = np.gradient(z,axis=1,edge_order =2)
    dist_long = (dx**2 +dy**2 +dz**2)**0.5   #nodal distances along the psi direction
    
    dx = np.gradient(x,axis=0,edge_order =2)
    dy = np.gradient(y,axis=0,edge_order =2)
    dz = np.gradient(z,axis=0,edge_order =2)
    dist_trans = (dx**2 +dy**2 +dz**2)**0.5   #nodal distances along the theta direction
    
    r_theta_u = np.gradient(x,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_v = np.gradient(y,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_w = np.gradient(z,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
      
    r_psi_u = np.gradient(x,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_v = np.gradient(y,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_w = np.gradient(z,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    
    n_u = r_theta_v*r_psi_w-r_theta_w*r_psi_v
    n_v = - (r_theta_u*r_psi_w-r_theta_w*r_psi_u)
    n_w = r_theta_u*r_psi_v-r_psi_u*r_theta_v
    
    mag = (n_u**2+n_v**2+n_w**2)**0.5
    n_u = n_u/mag
    n_v = n_v/mag
    n_w = n_w/mag
    
    r_theta_theta_u = np.gradient(r_theta_u,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_theta_v = np.gradient(r_theta_v,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_theta_w = np.gradient(r_theta_w,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    
    r_theta_psi_u = np.gradient(r_theta_u,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_theta_psi_v = np.gradient(r_theta_v,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_theta_psi_w = np.gradient(r_theta_w,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    
    r_psi_psi_u = np.gradient(r_psi_u,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_psi_v = np.gradient(r_psi_v,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_psi_w = np.gradient(r_psi_w,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    
    E = r_theta_u**2 + r_theta_v**2 + r_theta_w**2
    F = r_theta_u*r_psi_u + r_theta_v*r_psi_v + r_theta_w*r_psi_w
    G = r_psi_u**2 + r_psi_v**2 + r_psi_w**2

    L = r_theta_theta_u*n_u + r_theta_theta_v*n_v +r_theta_theta_w*n_w
    M = r_theta_psi_u*n_u + r_theta_psi_v*n_v +r_theta_psi_w*n_w
    N = r_psi_psi_u*n_u + r_psi_psi_v*n_v +r_psi_psi_w*n_w

    k1,k2,psi_1,psi_2 = curvature_roots(E, F, G, L, M, N)   
    kx,ky = curvature_original_directions(E, F, G, L, M, N)   #kx is as expeceted
    
    #vec 1 is in psi direction, vec 2 in theta direction
    k10 = -np.sin(theta)/(R+r*np.sin(theta))
    k20 = -1/r
    K0 = k10*k20

    mohrs_radius = np.abs(k1-k2)/2
    C = (k1+k2)/2
    kxy = (mohrs_radius**2-(kx-C)**2)**0.5
    kxy = np.nan_to_num(kxy)   #this is looking better

    dk1 = ky-k10
    dk2 = kx-k20
    dk12 = kxy
    dK = k1*k2-K0

    int_1 = np.cumsum(dK, axis=0)*r*alpha/100
    int_1 = int_1 - np.mean(int_1,axis=0)
    
    int_2 = np.cumsum(int_1, axis=0)*r*alpha/100  #this should be the strains
    int_2 = int_2 - np.mean(int_2,axis=0)  #making sure force is zero
    
    y = theta*r
    force = np.sum(int_2,axis=0)*E*t*(b/100)
    moment = np.sum(int_2*y,axis=0)*E*t*(b/100)

    D3 = (0-force)/(E*t*b)
    C3 = (12/(E*t*b**3))*(0-moment)
    int_2 = int_2 + C3*y +D3
    
    global strains
    strains_long  = (dist_long-dist0_long)/dist0_long + int_2
    strains_trans = (dist_trans-dist0_trans)/dist0_trans
    
    strains_long = int_2
    #print(np.sum(np.abs(strains_trans)))
    strains_trans = strains_trans*0
    
    energy_bending = 0.5*D*(dk1**2 + 2*v*dk1*dk2 +dk2**2 +2*(1-v)*dk12**2)  #per unit area
    #energy_bending = 0.5*D*((k1-k10)**2 + 2*v*(k1-k10)*(k2-k20) +(k2-k20)**2)  #per unit area
    
    energy_stretching = (1/(2*(1-v**2)))*E_y*t*strains_long**2 #+ 0.5*E_y*t*strains_trans**2  #per unit area    #this appears to be 5000 times more dominant!
    energy_stretching = energy_stretching
    #print(np.sum(energy_stretching)/np.sum(energy_bending))
    total_energy = np.sum(energy_bending*dist_long*dist_trans) + np.sum(energy_stretching*dist_long*dist_trans)
    #total_energy = total_energy*alpha*r*(np.pi/2)*R
    
    return total_energy

def bending_energy_plot(params,phi_0,it):
    """
    Function for plotting the bent tape ring

    Parameters
    ----------
    params : Array of floats
        The amplitudes of the different components of phi, z , R.
    phi_0 : Float
        Applied tape ring rotation.
    it : Int
        Counter for naming saved figures.

    Returns
    -------
    None.

    """

    fourier_amps_twist = [params[0],params[1],params[2],params[3]]
    fourier_amps_z = [params[4],params[5],params[6],params[7]]
    fourier_amps_R =[params[8],params[9],params[10],params[11]]
    
    alpha = np.pi/2
    theta = np.linspace(-alpha/2,alpha/2,30)
    psi  = np.linspace(0,np.pi*2,1020)
    psi,theta = np.meshgrid(psi,theta)
    
    
    
    twist = twist_fourier(fourier_amps_twist, psi, phi_0)
    
    z_offset = z_fourier(fourier_amps_z,psi)
    R = R_fourier(fourier_amps_R,psi,R0)
    
        
    """ New method where section twists around middle of arc"""
    x = (R -r*prop_down*np.sin(twist)+ r*np.sin(theta+twist))*np.cos(psi)
    y = (R -r*prop_down*np.sin(twist) + r*np.sin(theta+twist))*np.sin(psi)
    z = r*np.cos(theta+twist) -r*prop_down*np.cos(twist) + z_offset
    
    
    dx = np.gradient(x,axis=1,edge_order =2)
    dy = np.gradient(y,axis=1,edge_order =2)
    dz = np.gradient(z,axis=1,edge_order =2)
    dist_long = (dx**2 +dy**2 +dz**2)**0.5   #nodal distances along the psi direction
    
    dx = np.gradient(x,axis=0,edge_order =2)
    dy = np.gradient(y,axis=0,edge_order =2)
    dz = np.gradient(z,axis=0,edge_order =2)
    dist_trans = (dx**2 +dy**2 +dz**2)**0.5   #nodal distances along the theta direction
    
    r_theta_u = np.gradient(x,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_v = np.gradient(y,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_w = np.gradient(z,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
      
    r_psi_u = np.gradient(x,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_v = np.gradient(y,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_w = np.gradient(z,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    
    n_u = r_theta_v*r_psi_w-r_theta_w*r_psi_v
    n_v = - (r_theta_u*r_psi_w-r_theta_w*r_psi_u)
    n_w = r_theta_u*r_psi_v-r_psi_u*r_theta_v
    
    mag = (n_u**2+n_v**2+n_w**2)**0.5
    n_u = n_u/mag
    n_v = n_v/mag
    n_w = n_w/mag
    
    r_theta_theta_u = np.gradient(r_theta_u,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_theta_v = np.gradient(r_theta_v,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    r_theta_theta_w = np.gradient(r_theta_w,axis=0,edge_order=2)/np.gradient(theta,axis=0,edge_order=2)
    
    r_theta_psi_u = np.gradient(r_theta_u,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_theta_psi_v = np.gradient(r_theta_v,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_theta_psi_w = np.gradient(r_theta_w,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    
    r_psi_psi_u = np.gradient(r_psi_u,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_psi_v = np.gradient(r_psi_v,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    r_psi_psi_w = np.gradient(r_psi_w,axis=1,edge_order=2)/np.gradient(psi,axis=1,edge_order=2)
    
    E = r_theta_u**2 + r_theta_v**2 + r_theta_w**2
    F = r_theta_u*r_psi_u + r_theta_v*r_psi_v + r_theta_w*r_psi_w
    G = r_psi_u**2 + r_psi_v**2 + r_psi_w**2

    L = r_theta_theta_u*n_u + r_theta_theta_v*n_v +r_theta_theta_w*n_w
    M = r_theta_psi_u*n_u + r_theta_psi_v*n_v +r_theta_psi_w*n_w
    N = r_psi_psi_u*n_u + r_psi_psi_v*n_v +r_psi_psi_w*n_w

    k1,k2,psi_1,psi_2 = curvature_roots(E, F, G, L, M, N)   
    kx,ky = curvature_original_directions(E, F, G, L, M, N)   #kx is as expeceted
    
    #vec 1 is in psi direction, vec 2 in theta direction
    k10 = -np.sin(theta)/(R+r*np.sin(theta))
    k20 = -1/r
    K0 = k10*k20

    mohrs_radius = np.abs(k1-k2)/2
    C = (k1+k2)/2
    kxy = (mohrs_radius**2-(kx-C)**2)**0.5
    kxy = np.nan_to_num(kxy)   #this is looking better

    dk1 = ky-k10
    dk2 = kx-k20
    
    
    dk12 = kxy
    dK = k1*k2-K0

    int_1 = np.cumsum(dK, axis=0)*r*alpha/30
    int_1 = int_1 - np.mean(int_1,axis=0)
    
    int_2 = np.cumsum(int_1, axis=0)*r*alpha/30  #this should be the strains
    int_2 = int_2 - np.mean(int_2,axis=0)  #making sure force is zero
    
    
    #setting constants for zero force and moment
    yb = theta*r
    force = np.sum(int_2,axis=0)*E_y*t*(b/30)
    moment = np.sum(int_2*yb,axis=0)*E_y*t*(b/30)

    D3 = (0-force)/(E_y*t*b)
    C3 = (12/(E_y*t*b**3))*(0-moment)
    int_2 = int_2 + C3*yb +D3
    #need to make M = F = 0
    
    strains_long = int_2
    
    energy_bending = 0.5*D*(dk1**2 + 2*v*dk1*dk2 +dk2**2 +2*(1-v)*dk12**2)  #per unit area

    energy_stretching = (1/(2*(1-v**2)))*E_y*t*strains_long**2 #+ 0.5*E_y*t*strains_trans**2  #per unit area    #this appears to be 5000 times more dominant!
  
    total_E = energy_stretching + energy_bending
    
    val = total_E

    #max_val = np.max(np.abs(val))
    true_min = np.min(val)
    true_max = np.max(val)

    #vmin = -max_val 
    #vmax = max_val
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    #white_val = (0-true_min)/(true_max-true_min)
    stress = strains_long*E_y
    print(np.max(stress)/1e6)
    #max E = 2773
    #print(np.max(stress))
    print(np.mean(total_E))
    
    alph = (dK-0)/4720
    #colors = cmap2(alph)
    colors = cm.jet(alph)
    
    print('max dK',np.max(dK))
    fig = plt.figure(figsize=(15,10))
    ax = plt.axes(projection='3d',proj_type = 'ortho')
    ax.view_init(elev=30, azim=-45)
    ax.view_init(elev=0, azim=0)
    
    ax.plot_surface(x, y, z,facecolors=colors,edgecolors='none',linewidth = 0.2,shade=False,rstride=1,cstride=1)
    #ax.plot_surface(x, y, z,cmap=cm.jet,linewidth = 0.2,shade=False,rstride=5,cstride=5)
    
    #ax.plot_wireframe(x, y, z, rcount=1, ccount=0,color = 'k',linewidth =1)
    #ax.plot_wireframe(x, y, z, rcount=4, ccount=8,color = 'k',linewidth =0.2)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax._axis3don = False
    plt.title(r'$\phi_0=$' + str(round(phi_0,2))+r', $\phi_1=$' + str(round(phi_1,2)) + r', $z_1=$' + str(round(z_1*1000,1)) +'mm',y=0.8,size=15)
    plt.savefig('min/fourier_z_R' + str(it)+ '.png',dpi=300,bbox_inches="tight")
    plt.show()
    
    fig = plt.figure(figsize=(15,10))
    ax = plt.axes(projection='3d',proj_type = 'ortho')
    ax.view_init(elev=90, azim=0)
    
    ax.plot_surface(x, y, z,facecolors=colors,edgecolors='none',linewidth = 0.2,shade=False,rstride=1,cstride=1)
    #ax.plot_wireframe(x, y, z, rcount=1, ccount=0,color = 'k',linewidth =1)
    #ax.plot_wireframe(x, y, z, rcount=4, ccount=8,color = 'k',linewidth =0.2)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax._axis3don = False
    plt.title(r'$\phi_0=$' + str(round(phi_0,2))+r', $\phi_1=$' + str(round(phi_1,2)) + r', $z_1=$' + str(round(z_1*1000,1)) +'mm',y=1.0,size=15)
    plt.savefig('min/fourier_z_R_plan' + str(it)+ '.png',dpi=300,bbox_inches="tight")
    plt.show()
  



import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint

#defining constraints
dist_tolerance = 0 
constraint_distance =  NonlinearConstraint(lambda params: length(params,phi_0), -0, 0)
constraint_distance2 =  NonlinearConstraint(lambda params: length_in(params,phi_0), -dist_tolerance, dist_tolerance)
constraint_distance3 =  NonlinearConstraint(lambda params: length_out(params,phi_0), -dist_tolerance, dist_tolerance)

all_constraints = [constraint_distance,constraint_distance2,constraint_distance3]
all_constraints = [constraint_distance]
all_constraints = []

all_constraints = [constraint_distance2,constraint_distance3]
#g = [1,1,1,1,1,1,1,1,1,1,1,1]
g = [0,0,0,0,0,0,0,0,0,0,0,0]
phi_0 = np.pi/6

p0 = np.linspace(0,1.6,41)
phi_1s = []
z_1s = []
R_ds = []
energies = []
test = True

#running through applied rotations phi_0, and performing energy minimisation to find the deformed shape
if test == True:
    it = 0
    for phi_0 in p0:
        it+=1
        result = optimize.minimize(bending_energy_opt,g, args = (phi_0),constraints=all_constraints) 
        params = result.x
        
        fourier_amps_twist = [params[0],params[1],params[2],params[3]]
        fourier_amps_z = [params[4],params[5],params[6],params[7]]
        fourier_amps_R =[params[8],params[9],params[10],params[11]]
        
        phi_1 = twist_fourier(fourier_amps_twist, np.pi/2, phi_0)
        z_1 = z_fourier(fourier_amps_z,np.pi/2) - z_fourier(fourier_amps_z,0)
     
        phi_1s.append(phi_1)
        z_1s.append(z_1)
        energies.append(bending_energy_opt(params,phi_0))
        
        bending_energy_plot(params,phi_0,it)
        
     
        g = params
            
     
    phi_1s = np.array(phi_1s)
    z_1s = np.array(z_1s)*1000
    
    plt.plot(phi_1s,z_1s,color='k',label='Mode shape energy minimisation')
    plt.xlabel(r'$\phi_1$')
    plt.ylabel(r'$z_1$ (m)')
    
    
    #plt.quiver(phi_1s[:-1], z_1s[:-1], phi_1s[1:]-phi_1s[:-1], z_1s[1:]-z_1s[:-1], scale_units='xy', angles='xy', scale=1)
    dy_dx = np.gradient(z_1s,edge_order=2)/np.gradient(phi_1s,edge_order=2)
    ax = plt.gca()
    
    # for i in range(0,len(z_1s)):
    #     if i%4 == 0:
    #         ax.annotate('', xytext=(phi_1s[i], z_1s[i]),  xy=(phi_1s[i]+0.01, z_1s[i]+0.01*dy_dx[i]), arrowprops=dict(arrowstyle="<|-", color='k'),   size=15)


    plt.plot(phi_1_FE,z_1_FE,color='r',label='FE result')
    plt.title('Tape ring bending, fourier mode shape')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=2,frameon=False)
    plt.savefig('minimisation_fourier_z_R.png',dpi=300,bbox_inches="tight")
    plt.show()
    
    np.save('z_1_mode_3', z_1s)
    np.save('phi_1_mode_3', phi_1s)
    np.save('energies_mode_3', energies)
    
    z_1s = z_1s/1000
    
    plt.plot(p0,phi_1s)
    plt.xlabel(r'$\phi_0$')
    plt.ylabel(r'$\phi_1$')
    plt.show()
    
    plt.plot(p0,energies)
    plt.xlabel(r'$\phi_0$')
    plt.ylabel(r'$E$ (J)')
    plt.show()
    
    T = np.gradient(energies,edge_order=2)/np.gradient(p0,edge_order=2)

    plt.plot(p0,T)
    plt.xlabel(r'$\phi_0$')
    plt.ylabel(r'$T$ (Nm)')
    plt.show()



