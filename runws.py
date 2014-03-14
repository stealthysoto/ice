#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Plot histogram of an entire 3D image and fit it with Weibull.'''

# ---------------- Imports

import matplotlib.pyplot as plt
import numpy as np
import wstuff as ws
from scipy.optimize import leastsq

# -- BEGIN PARAMETERS ----
bins = 40
rangemax = 0.25
namebase = '1110_3d_snp9'
namebase = '1200_3d_snp5_img' 
namebase = '524_3d_snp1_img'
namebase = '524_3d_snp2_img'
namebase = '524_3d_snp3_img'
limits = [0, 0, 0, 0]  # [x_min, x_max, y_min, y_max], if max=0: no limit
log_into_register = False  # Turn on/off if results should be logged
register_path = 'C:\ice_register.csv'

# ------ END PARAMETERS --

# Get histogram for set number of bins
data, zfixed, xgrid, ygrid = ws.getR(namebase, limits)
values, rhist = np.histogram(data, bins=bins, range=(0, rangemax),density=True)
    
# If there are gaps, find the maximum possible number of bins to not get them
while not (values > 0).all():
    bins = bins-1
    values, rhist = np.histogram(data, bins=bins, range=(0, rangemax),density=True)
    if (bins == 5):
        print('WARNING: Ideal bins are lower than 5, printing them anyway.')
        break

# Get the spacing (labels) and do some reporting
labels = rhist[1:]-(rhist[1]-rhist[0])/2
total_mean = np.mean(data)
print("Number of bins determined as: " + str(bins))
print("Completed histogram from "+str(data.size)+" r values.")
print("The total mean is: "+str(total_mean)+".")

# Setting estimated values for sigma and eta
sigma_0 = .2
eta_0 = 1.0
p0 = ([sigma_0, eta_0])  # initial set of parameters
plsq = leastsq(ws.residuals, p0, args=(values, labels), maxfev=200)  # actual fit

# Report sigma and eta in commandline
sigma_ret = plsq[0][0]
eta_ret = plsq[0][1]
print('Sigma is: '+str(sigma_ret))
print('Eta is: '+str(eta_ret))

# Setting estimated values for sigma and eta and gamma
sigma_0 = sigma_ret
eta_0 = eta_ret
gamma_0 = 0
p0 = ([sigma_0, eta_0, gamma_0])  # initial set of parameters
plsq3 = leastsq(ws.residuals3, p0, args=(values, labels), maxfev=200)  # actual fit

# Report sigma and eta and gamma in commandline
sigma_ret3 = plsq3[0][0]
eta_ret3 = plsq3[0][1]
gamma_ret3 = plsq3[0][2]
print('Sigma is: '+str(sigma_ret3))
print('Eta is: '+str(eta_ret3))
print('Gamma is: '+str(gamma_ret3))

# Plot it (first figure)
fignum = 1
plt.figure(fignum)
plt.clf()

# Plot in log scale
#plt.semilogy(labels, values, labels, ws.pWeibull(labels, sigma_ret, eta_ret))
plt.semilogy(labels, values, 'o', labels, ws.pWeibull(labels, sigma_ret, eta_ret), labels, ws.pWeibull3(labels, sigma_ret3, eta_ret3, gamma_ret3))
plt.grid()
plt.legend(('Expt', '2-parameter', '3-parameter'))

# Creating dual X axis
ax1 = plt.subplot(111)
ax2 = ax1.twiny()

# Setting proper labels
ax1.set_xlabel(r"$r$")

# Setting up functions to convert r <---> phi
r_to_phi = lambda x: 180*np.arccos(1-x)/np.pi
phi_to_r = lambda x: 1 - np.cos((x*np.pi)/180)
new_tick_locations = phi_to_r(np.array([5, 10, 15, 20, 25, 30, 35, 40]))
tick_function = lambda x: ["%.0f" % z for z in r_to_phi(x)]
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"$\phi$")
ax2.set_xlim((0, rangemax))

# Printing sigma and eta onto the plot
plt.text(0.87, 0.82,
         r'$\sigma$: %.3f $\eta$: %.3f' % (sigma_ret, eta_ret),
         fontsize=12,
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax1.transAxes)

# Printing bins and mean onto the plot
plt.text(0.84, 0.77,
         r'mean: %.3f bins: %i' % (total_mean, bins),
         fontsize=12,
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax1.transAxes)
plt.show()

# Saving figure to disk
plt.savefig(namebase+'weib_comp.png', dpi=100)

# If asked to, log into register
if (log_into_register):
    ws.logIntoRegister(register_path,
                    [namebase, sigma_ret, eta_ret, total_mean,
                     bins, data.size, rangemax])

# Display results as a mesh
fignum += 1
plt.close(fignum)
Nx, Ny = np.shape(zfixed)
i = int(Nx/2)
j = int(Ny/2)
x1=i-20; x2=i+20; y1=j-20; y2=j+20
ax = plt.figure(fignum).gca(projection='3d') # Set up a three dimensional graphics window 
ax.plot_surface(xgrid[y1:y2,x1:x2],ygrid[y1:y2,x1:x2],zfixed[y1:y2,x1:x2],rstride=1,cstride=1) # Make the mesh plot
#ax.plot_wireframe(xgrid[y1:y2,x1:x2],ygrid[y1:y2,x1:x2],zfixed[y1:y2,x1:x2],rstride=1,cstride=1) # Make the mesh plot
ax.set_xlabel('x ($\mu$m)') # Label axes
ax.set_ylabel('z ($\mu$m)')
ax.set_zlabel('y ($\mu$m)')
plt.show()

# Saving figure to disk
plt.savefig(namebase+'surface.png', dpi=200)

#plt.pcolor(xgrid[y1:y2,x1:x2],ygrid[y1:y2,x1:x2],zfixed[y1:y2,x1:x2])
#plt.colorbar()
#plt.title("After subtracting baseline")
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

