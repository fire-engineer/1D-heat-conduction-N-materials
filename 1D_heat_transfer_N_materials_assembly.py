# -*- coding: utf-8 -*-
"""
#INSTRUCTIONS : INPUT ANY NEW MATERIALS, MAKE ASSEMBLY, AND THEN DEFINE ANY OTHER SIMULATION PARAMETERS
@author: Benoît.LeBlanc
"""
import matplotlib.pyplot as plt  
import numpy as np
import sys
from statistics import mean
from dataclasses import dataclass

@dataclass
class material:
    full_name: str
    color:     str
    thickness: float
    cp:     float
    rho :   float
    t_cond: float
    def fourier(self, t):
        return self.t_cond / (self.cp * self.rho) * t / self.thickness**2 
    def numerical_fourier(self, dx, dt):
        return self.t_cond / (self.cp * self.rho) * dt / dx**2 
    def biot(self, h_conv, L):
        return h_conv * L / self.t_cond 


###############################################################################
###############################################################################
###############################################################################
############################## Input Values ###################################
###############################################################################
###############################################################################
###############################################################################


################################################################
################# Materials & Properties #######################
################################################################

#list of materials defined. To add another definition simply add another line. 

# NAME                   FULL NAME                     color     hickness    cp       rho     t_cond
#                                                                 [m]  [J /kg K]  [kg/m3]  [W/m K]
ICFconc   = material(   'ICF Concrete',               'grey',    0.2,     750,     2400,    0.72)
ICFinsu   = material(   'ICF Insulation',             'cyan',    0.092,  1670,     22.4,    0.04)
densglass = material(   'DensGlass',                'yellow',    0.016,  1085,      755,    0.135)
claybrick = material(   'Clay Brick',                 'pink',    0.025,   960,     1920,    1.06)
cementboard = material( 'Cement Board',              'black',    0.025,  1000,      918,    1.27)
stucco    = material(   'Low. Dens. Cement Mortar',  'black',    0.025,  1000,      918,    1.27)


################
### IF R VALUE IS GIVEN
# Rval = ... #  [m2 K / W]
# thickness = ... [m] 
# t_cond = thickness / Rval

### IF THERMAL DIFFUSIVITY IS GIVEN
#k = ... # [m2/s]
#t_cond = k * rho * cp


# once materials are defined, assembly can be created

##################################################################
################# ASSEMBLE MATERIALS     #########################
##################################################################
# here input the assembly to use in the simulation. 
# Format is [ material_1, material_2, ... , material_N ]

assembly = [ICFconc,ICFinsu,claybrick]




##################################################################
################# Simulation Parameters ##########################
##################################################################

# Temperature initial conditions and Boundary conditions
Theta_initx = 20.0  # [deg C] initial temperature throughout
Theta_init1 = 20.0 # [deg C] initial temp on left boundary
Theta_init2 = 600.0 # [deg C] initial temp on right boundary


# Boundary Condition types for
BC1 = 'fixed' # BC1 is left boundary
BC2 = 'fixed' # BC2 is right boundary

#BC = 'symmetry'    #symmetry BC (same as adiabatic)
#BC = 'fixed'       #fixed temperature BC
#BC = 'convection'  #convection BC
#h_conv = 10 # [W/m2 K] convection coefficient
#Tinf = 600 # [deg C]

# Total simulation time
t_final = 1800.0  # [s] Total runtime (1800s = 30m)

# Mesh and time Resolution
dt      = 0.1     # time step size    [s]
dx      = 0.001   # discrete element size  [m]



###############################################################################
###############################################################################
###############################################################################
############################## Calculations ###################################
###############################################################################
###############################################################################
###############################################################################


##################################################################
############ calculate non-dimensional numbers  ##################
##################################################################

# Calculate integral fourier number for last material to the right
integral_Fo = assembly[-1].fourier(t_final)
print('The integral Fourier number for the external cladding layer is: '+ str(integral_Fo))

# Calculate numerical Fourier numbers in material(s)
for m,material in enumerate(assembly): 
    Fo = material.numerical_fourier(dx, dt)
    if Fo>0.5:
        print('Fourier number for material '+ material.full_name +' out of bounds ( > 0.5 ) :'+ str(Fo))
        sys.exit()

# Calculate Biot number if convection
# if 'convection' in BC:  Bi= assembly[-1].biot(h_conv,dx)   
# if 'convection' in BC:      print('stability criterion Fo(1+Bi)>0.5  : '+ str(Fo3*(1+Bi)))


#################################################################
############## initialize arrays ################################ 
#################################################################
N = len(assembly)-1  # last index of assembly

### Calculate x and indice positions where materials start and end
x_start = np.zeros(len(assembly)+1)          #start and end positions
i_start = np.zeros(len(assembly), dtype=int) #start and end indice

for m,material in enumerate(assembly):
    x_start[m+1:] +=      material.thickness
i_start = x_start/dx
i_start = i_start.astype(int)

x_total = x_start[N+1]
i_total = i_start[N+1]

### initialize physics fields
X = np.arange(0,x_total,dx)
Time = np.arange(0,t_final,dt)

Theta     = np.zeros(len(X))  # temperature array
Theta_old = np.zeros(len(X))  # temperature at previous timestep

#initialize temperatures
Theta[:]  = Theta_initx
Theta[0]  = Theta_init1
Theta[-1] = Theta_init2

# initialize property arrays with material 1 properties
CP        = np.zeros(len(X)) ; CP[:]     = assembly[0].cp
RHO       = np.zeros(len(X)) ; RHO[:]    = assembly[0].rho
T_COND    = np.zeros(len(X)) ; T_COND[:] = assembly[0].t_cond

# fill out properties of other materials
for i,x in enumerate(X):    
    for m,material in enumerate(assembly):
        if i > i_start[m] and i <= i_start[m+1]:
            CP[i]     = assembly[m].cp
            RHO[i]    = assembly[m].rho
            T_COND[i] = assembly[m].t_cond    

# Calculate Fourier Number throughout entire domain
FO = T_COND/(RHO*CP) * dt / dx**2  

#get average values for Fourier number on both sides East and West of each element, these are directly used as coefficients in the equation
FO_W = np.zeros(len(X))  
FO_E = np.zeros(len(X))  
for i in range(1, len(Theta)-1) :
    FO_W[i] = 0.5 * (FO[i]+FO[i-1])
    FO_E[i] = 0.5 * (FO[i]+FO[i+1]) 


#################################################################
############### explicit time march solution ####################
#################################################################

# time loop
for t in np.arange(0,t_final,dt):
    # x loop
    Theta_old[:] = Theta[:]
    for i in range(1, len(Theta)-1) :
        #explicit time marching scheme  (taken from Incropera & DeWitt, Heat & Mass Transfer, 5th edition, chap 5.9, page 281, eq 5.73)
        Theta[i]= Theta_old[i]   +   FO_W[i] * ( Theta_old[i-1] - Theta_old[i] )   +   FO_E[i] * ( Theta_old[i+1] - Theta_old[i] )

    #### boundary conditions
    # for BC1 symmetry BC1 or BC2
    if BC1 == 'symmetry' : Theta[0] = Theta_old[1]
    if BC1 == 'fixed'    : pass # do nothing, no change

    #for BC2 on (right side)
    if BC2 == 'symmetry' : Theta[-1] = Theta_old[-2]
    if BC2 == 'fixed'    : pass # do nothing, no change
    #if convection BC (taken from Incropera & DeWitt, Heat & Mass Transfer, 5th edition, chap 5.9, page 283, eq 5.77)
    # if BC2 == 'convection' : Theta[-1] = 2*FO[-1] * ( Theta_old[-2]  +  Bi*Tinf )   +   (1 - 2*FO[-1] - 2*Bi*FO[-1] ) * Theta_old[-1]

###############################################################################
###############################################################################
###############################################################################
######################## Post-Processing and Plotting #########################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

################################################################
######################### Calculate integral values ############
################################################################

print('The exterior temperature is : '+str(Theta[-1])+' deg C')
print('The mean temperature is : '+str(mean(Theta))+' deg C')

#mean, min and max temperature in different materials
for m,material in enumerate(assembly):
    meanT = mean( Theta[ i_start[m] : i_start[m+1] ] )
    minT  = min(  Theta[ i_start[m] : i_start[m+1] ] )
    maxT  = max(  Theta[ i_start[m] : i_start[m+1] ] )
    print( 'The mean temperature of '      + material.full_name + ' is :'    + str(meanT) )
    print( 'The [min,max] temperature of ' + material.full_name + ' is : ['+str(minT)+','+str(maxT)+']')

#################################################################
############### Plotting temperature profile ####################
#################################################################

# create plot
plt.figure(figsize=(10, 6.6))
plt.plot(X, Theta,label='Temperature',color='r') 

# coloring materials regions on plot 
for m,material in enumerate(assembly):
    plt.axvspan(x_start[m], x_start[m+1], alpha = 0.5, color=material.color, label=material.full_name) 
    
# # showing max temp of icf 
# plt.axhline(379,color='black',linestyle='dotted', label = 'Max Temp ICF Insu')

# naming the x axis  and other considerations
plt.title('Temperature profile @ '+str(t_final/60)+' minutes ') 
plt.xlabel('Position (m)')  
plt.ylabel('Temperature (C)')  
#plt.ylim(0, Theta_init1*1.1)
plt.xlim(0, x_total*1.05)
plt.legend()    
 
# display plot  
plt.show()  