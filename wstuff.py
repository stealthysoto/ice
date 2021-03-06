#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Plot histogram of an entire 3D image and fit it with Weibull.'''

# ---------------- Imports
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import leastsq


# ---------------- Statistics
def pWeibull(r, sigma, eta):
    ''' Weibull function to be fit. '''

    from numpy import exp

    mu = 1-r
    ret = 2*eta/sigma**2/mu**3 * \
        (((mu**(-2)-1)/sigma**2)**(eta-1)) * \
        exp(-((mu**(-2)-1)/sigma**2)**eta)
    return ret


def pWeibull3(r, sigma, eta, gamma):
    ''' Weibull function to be fit. '''

    from numpy import exp

    mu = 1-r
    ret = 2*eta/sigma**2/mu**3 * (((mu**(-2)-1-gamma)/sigma**2)**(eta-1)) * exp(-((mu**(-2)-1-gamma)/sigma**2)**eta)
    return ret


def pWeibull3b(r, sigma, eta, gamma):
    ''' Weibull function to be fit. '''

    from numpy import exp

    mu = 1-r
    ret = 2*eta/sigma**2/mu**3 * (((mu**(-2)-1)/sigma**2)**(eta-1)) * exp(-((mu**(-2)-1)/sigma**2)**eta)+gamma
    return ret


def residuals(p, y, r):
    ''' Error function for fitting. '''

    from numpy import log

    sigma = p[0]
    eta = p[1]
    err = log(y)-log(pWeibull(r, sigma, eta))
    return err


def residuals3(p, y, r):
    ''' Error function for fitting. '''

    from numpy import log

    sigma = p[0]
    eta = p[1]
    gamma = p[2]
    err = log(y)-log(pWeibull3(r, sigma, eta, gamma))
    return err

def residuals3b(p, y, r):
    ''' Error function for fitting. '''

    from numpy import log

    sigma = p[0]
    eta = p[1]
    gamma = p[2]
    err = log(y)-log(pWeibull3b(r, sigma, eta, gamma))
    return err


# ---------------- File handling
def readArray(filename, dtype, separator=','):
    ''' (not ours) Read a file with an arbitrary number of columns. The type
        of data in each column is also arbitrary - it will be cast to the
        given dtype at runtime. '''

    cast = np.cast
    data = [[] for dummy in xrange(len(dtype))]
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            fields = line.strip().split(separator)
            for i, number in enumerate(fields):
                data[i].append(number)
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return np.rec.array(data, dtype=dtype)


def loadGrid(profile, horizontal):
    ''' Reads data from a single-image CSV file and distributes x,y,z values
        into separate arrays of picture dimensions. '''

    # Setting appropriate descriptions/type to columns for easy access
    mydescr = np.dtype([('x', 'float32'), ('y', 'float32'), ('z', 'float32')])
    data = readArray(profile+'.csv', mydescr)  # Loading the .csv file

    # Counting the number of 0.00000 in arrays to get dimensions
    ''' Note: This can possibly be done better, namely because there can be
        other zeros around in other than the first row/column. This is not
        likely to happen at all though. '''
    Ny = (0 == data['y']).sum()  # Y dimension
    Nx = (0 == data['x']).sum()  # X dimension

    # Reshaping into three separate fields of picture dimension

    if horizontal:
        print horizontal
        xgrid = data['x'].reshape(Nx, Ny)
        ygrid = data['y'].reshape(Nx, Ny)
        zgrid = data['z'].reshape(Nx, Ny)
        return xgrid, ygrid, zgrid

    else:
        print horizontal
        xgrid = data['x'].reshape(Nx, Ny).T
        ygrid = data['y'].reshape(Nx, Ny).T
        zgrid = data['z'].reshape(Nx, Ny).T
        return ygrid, xgrid, zgrid



def logIntoRegister(filename, data):
    ''' Logs parameters of current run into file at address "filename". If file
        doesn't exist, it will be created and added headers.'''

    # Add content to a new line
    contents = "\n"+','.join(map(str, data))

    # If file doesn't exist, include appropriate headers
    if not os.path.isfile(filename):
        headers = "Namebase,Sigma,Eta,Mean,Bins,Datasize,Rangemax"
        contents = headers + contents

    # Open and append into file
    g = open(filename, "a")  # Opening file to write into
    g.write(contents)
    g.close()


# ---------------- Histogram
def multiFit(y, z):
    ''' Will fit and substract a z baseline from all values in one line.
        Returns leveled z-line values. '''

    pfit = np.polyfit(y, z, 3)
    zbaseline = np.polyval(pfit, y)
    zfixed = z-zbaseline

    return zfixed


def linearFit(y, z):
    ''' Older way of linear fit that is substracted from all z values in line.
        Returns leveled z-line values. '''
    # Fitting with linearly generated sequence
    A = np.array([y, np.ones(y.size)])
    w = np.linalg.lstsq(A.T, z)[0]  # obtaining the parameters
    zline = w[0]*y+w[1]
    zfixed = z-zline  # substracting baseline from every point

    return zfixed

'''
def getRold(profile, limits):

    # Fetching grid from file
    xgrid, ygrid, zgrid = loadGrid(profile)

    # Fetching dimensions of the array
    xmax, ymax = xgrid.shape

    if (limits[1] == 0):
        limits[1] = xmax
    if (limits[3] == 0):
        limits[3] = ymax

    # Transposing y and z, because we're doing vertical measurements
    yT = np.transpose(ygrid)
    zT = np.transpose(zgrid)

    # Going through all lines
    for i in range(limits[2], limits[3]):
        # Selecting the i-th line from each transposed array
        y = yT[i][limits[0]:limits[1]]
        z = zT[i][limits[0]:limits[1]]

        # Fitting with (^3, ^2, ^1) fit, can use linearFit() if necessary
        zfixed = multiFit(y, z)

        # Getting the slope in every point
        dydz = np.diff(zfixed)/np.diff(y)  # Note: try reversing this? dz/dy?
        r = 1-(1/(1+dydz**2))**(0.5)

        # If first line, start r_total, else append
        if (i == limits[0]):
            r_total = r
        else:
            r_total = np.concatenate((r_total, r))

    return r_total
'''

def getR(profile, limits, wconv, horizontal):
    ''' Reads every line of z(y), eliminates slowly-changing behavior for the entire surface
        and computes the r value. Returns all results of r in one list. '''

    # Fetching grid from file
    xgrid, ygrid, zgrid = loadGrid(profile, horizontal)

    # Fetching dimensions of the array
    xmax, ymax = xgrid.shape
    horizontal = True
    if horizontal:
        if (limits[1] == 0):
            limits[1] = xmax
        if (limits[3] == 0):
            limits[3] = ymax
        xT = np.transpose(xgrid)
        yT = np.transpose(ygrid)
        zT = np.transpose(zgrid)

    else:
        if (limits[1] == 0):
            limits[1] = ymax
        if (limits[3] == 0):
            limits[3] = xmax
        xT = np.copy(ygrid)
        yT = np.copy(xgrid)
        zT = np.copy(zgrid)

    # Get a smoothed version
    x1d = xT[:,0]
    y1d = yT[0,:]
    dy = y1d[1]-y1d[0]
<<<<<<< HEAD
=======
    wconv = 5.0 # This should be microns
>>>>>>> FETCH_HEAD
    fNyconv = wconv/dy
    Nyconv = int(fNyconv)
    zTsmooth = polysmooth(xT,yT,zT,6,6)
    zTfixed = zT-zTsmooth
    Ny, Nx = np.shape(zTfixed)
    filt = np.ones(Nyconv)/fNyconv
    print 'Filter width in microns = ', wconv, ' meaning filt = ', filt
    #print np.shape(zTfixed), Nx, Ny

    # Going through all lines to smooth in the crystallographic z-direction
    for i in range(limits[2], limits[3]):
        # Selecting the i-th line from each transposed array
        zTfixed_line = zTfixed[i][limits[0]:limits[1]]
        
        # Filter, perhaps
        zTfilt_line = np.convolve(zTfixed_line,filt,'same')
        #print 'zTfilt_line', np.shape(zTfilt_line)
        #print zTfilt_line

        # If first line, start zTfixed_grid, else append
        if (i == limits[2]):
            zTfilt = zTfilt_line
        else:
            zTfilt = np.concatenate((zTfilt, zTfilt_line),axis=0)
    zTfilt = zTfilt.reshape(Ny,Nx)
    #print np.shape(zTfilt)
    
    # Going through all lines to smooth in the crystallographic x-direction
    for j in range(limits[0], limits[1]):
        # Selecting the i-th line from each transposed array
        #zTfixed_line = zTfilt[limits[2]:limits[3]][j]
        zTfixed_line = zTfilt[limits[2]:limits[3],j]
        #print limits[2],limits[3],np.shape(zTfixed_line)
        
        # Filter, perhaps
        zTfilt_line = np.convolve(zTfixed_line,filt,'same')
        #print 'zTfilt_line', np.shape(zTfilt_line)
        #print zTfilt_line

        # If first line, start zTfixed_grid, else append
        if (j == limits[0]):
            zTfilt2 = zTfilt_line
        else:
            zTfilt2 = np.concatenate((zTfilt2, zTfilt_line),axis=0)

    #print np.shape(zTfilt)
    zTfilt3 = zTfilt2.reshape(Nx,Ny).T
    #print np.shape(zTfilt2), np.shape(zTfilt3)

    # Going through all lines to get roughness
    for i in range(limits[2], limits[3]):
        # Selecting the i-th line from each transposed array
        y = yT[i][limits[0]:limits[1]]
        zTfilt_line = zTfilt3[i][limits[0]:limits[1]]
        
        # Getting the slope in every point
        #dydz = np.diff(zTfixed_line)/np.diff(y)  # Note: try reversing this? dz/dy?
        dydz = np.diff(zTfilt_line)/np.diff(y)  # Note: try reversing this? dz/dy?
        r = 1-(1/(1+dydz**2))**(0.5)

        # If first line, start r_total, else append
        if (i == limits[0]):
            r_total = r
        else:
            r_total = np.concatenate((r_total, r))
    
    # Display the results of 1-d slices
    i = int(Ny/2); print Ny, i
    j = int(Nx/2); print Nx, j

    fignum = 10
    plt.close(fignum)
    plt.figure(fignum)
    plt.plot(x1d,zT[:,j],x1d,zTsmooth[:,j])
    plt.xlabel('x')
    plt.legend(['original','tilt removed'])
    plt.show()

    fignum = 11
    plt.close(fignum)
    plt.figure(fignum)
    plt.plot(y1d,zT[i,:],y1d,zTsmooth[i,:])
    plt.xlabel('z')
    plt.legend(['original','tilt removed'])
    plt.show()

    fignum = 12
    zTfixed_line = zTfixed[i,limits[0]:limits[1]]
    zTfilt_line = zTfilt3[i,limits[0]:limits[1]]
    plt.close(fignum)
    plt.figure(fignum)
    plt.plot(y1d,zTfixed_line)
    plt.plot(y1d,zTfilt_line,linewidth=3)
    plt.xlabel('z')
    plt.ylabel('y (surface height)')
    plt.legend(['fixed', 'filtered'])
    plt.show()

    fignum = 13
    dydz_fixed = np.diff(zTfixed_line)/np.diff(y1d); theta_fixed = np.arctan(dydz_fixed)*180/np.pi
    dydz_filt = np.diff(zTfilt_line)/np.diff(y1d); theta_filt = np.arctan(dydz_filt)*180/np.pi
    plt.close(fignum)
    plt.figure(fignum)
    plt.plot(y1d[1:],theta_fixed,y1d[1:],theta_filt)
    plt.xlabel('z')
    plt.ylabel('surface tilt angle')
    plt.legend(['fixed', 'filtered'])
    plt.show()
    
    fignum = 14
    zTfixed_line = zTfixed[limits[2]:limits[3],j]
    zTfilt_line = zTfilt3[limits[2]:limits[3],j]
    plt.close(fignum)
    plt.figure(fignum)
    plt.plot(x1d,zTfixed_line)
    plt.plot(x1d,zTfilt_line,linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y (surface height)')
    plt.legend(['fixed', 'filtered'])
    plt.show()
    
    
    #zfilt = np.transpose(zTfilt)
    zfilt3 = np.transpose(zTfilt3)
    return r_total, zfilt3, xgrid, ygrid

def polysmooth(x,y,z,NI,NJ):

    # size of the incoming array
    Nx, Ny = np.shape(z)
    x1d = x[:,0]
    y1d = y[0,:]

    # Get the C coefficients
    #NI = 7
    CIj = np.zeros((NI,Ny))
    for j in range (Ny):
        CIj[:,j] = np.flipud(np.polyfit(x1d,z[:,j],NI-1))

    # Get the D coefficients
    #NJ = 7
    DIJ = np.zeros((NI,NJ))
    for I in range (NI):
        DIJ[I,:] = np.flipud(np.polyfit(y1d,CIj[I,:],NJ-1))
    
    # Reconstruct the entire surface
    zsmooth = np.zeros((Nx,Ny))
    for I in range(NI):
        for J in range(NJ):    
            zsmooth += DIJ[I,J]*x**I*y**J

    return zsmooth
 
