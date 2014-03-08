#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Plot histogram of an entire 3D image and fit it with Weibull.'''

# ---------------- Imports

import matplotlib.pyplot as plt
import numpy as np
import wstuff as ws
from scipy.optimize import leastsq

def getHistogram2(filebase, bins, rangemax, limits):
    ''' Filebase is the name of the folder, where the CSVs are.
        Num is the number of CSVs that are to be parsed and put together.
        Bins is the number of bins to use when making a histogram. Returns
        spacing (labels), hist (values), data (all r values in one list).'''

    # Getting all r values from file
    data, zfixed, xgrid, ygrid = ws.getR(filebase, limits)

    # Creating the histogram
    print data
    print bins
    
    hist, rhist = np.histogram(data, bins=bins, range=(0, rangemax),
                               density=True)

    # Making our own spacing - middle of all intervals.
    spacing = rhist[1:]-(rhist[1]-rhist[0])/2

    return spacing, hist, data, zfixed, xgrid, ygrid 


# -- BEGIN PARAMETERS ----
bins = 25
rangemax = 0.25
namebase = '1110_3d_snp9'
namebase = '1200_3d_snp5_img' 
limits = [0, 0, 0, 0]  # [x_min, x_max, y_min, y_max], if max=0: no limit
log_into_register = False  # Turn on/off if results should be logged
register_path = 'C:\ice_register.csv'

# ------ END PARAMETERS --

# Get histogram for set number of bins
labels, values, data, zfixed, xgrid, ygrid = getHistogram2(namebase, bins, rangemax, limits)

# If there are gaps, find the maximum possible number of bins to not get them
while not (values > 0).all():
    bins = bins-1
    labels, values, data, zfixed, xgrid, ygrid = getHistogram2(namebase, bins, rangemax, limits)
    if (bins == 5):
        print('WARNING: Ideal bins are lower than 5, printing them anyway.')
        break

print("Number of bins determined as: " + str(bins))
print("Completed histogram from "+str(data.size)+" r values.")

total_mean = np.mean(data)
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

# Plot it (first figure)
fignum = 1
plt.figure(fignum)
plt.clf()

# Plot in log scale
plt.semilogy(labels, values, labels, ws.pWeibull(labels, sigma_ret, eta_ret))
plt.grid()
plt.legend(('Expt', 'Best fit'))

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

