import numpy as np
import pylab as plt

"""
In order to build a look up table, perform the following steps:
    filesToTable(filenames) -> (produces 'average.txt')
    gprocessTable() -> (takes 'average.txt' and produces 'curve.txt')
    interpLUT() -> (takes 'curve.txt' and produces 'LUT.txt')
    LUT.txt can be given to hrl to correct textures at load time.

Each of these functions will save their results to files. The resulting
fit can be examined by running plotTables() in the same directory.
"""

def plotTables(avgfl="average.txt",lutfl="curve.txt"):
    """
    This function takes the averaged data, the generated lookup table, and plots them to
    see the quality of the fit.
    """
    avg = np.genfromtxt(avgfl,skip_header=1)
    lut = np.genfromtxt(lutfl,skip_header=1)
    plt.plot(avg[:,0],avg[:,1])
    plt.plot(lut[:,0],lut[:,1])
    #plt.plot(lut[:,0],lut[:,1]-np.sqrt(lut[:,2]))
    #plt.plot(lut[:,0],lut[:,1]+np.sqrt(lut[:,2]))
    plt.show()

def filesToTable(fls,wfl='average.txt'):
    """
    This is the first step of preparing a lookup table. Here we gather
    together a set of csvs of luminance measurements into one array,
    clear out useless rows, and average points at the same intensity.

    This returns a two column array with a number of elements derived
    from the csv file, i.e. a bit less than 2^16. This array can then be
    sampled by interpLUT right away producing a very rough
    linearization, or it can be fed into gprocessTable, which fits the
    data set using a Gaussian process model, which can then be fed into
    interpLUT.
    """
    hshmp = {}
    if type(fls) == str: fls = [fls]
    tbls = [ np.genfromtxt(fl,skip_header=1) for fl in fls ]
    # First we build up a big intensity to luminance map
    for tbl in tbls:
        for rw in tbl:
            if hshmp.has_key(rw[0]):
                hshmp[rw[0]] = np.concatenate([hshmp[rw[0]],rw[1:]])
            else:
                hshmp[rw[0]] = rw[1:]
    # Now we average the values, clearing all nans from the picture.
    for ky in hshmp.keys():
        hshmp[ky] = np.mean(hshmp[ky][np.isnan(hshmp[ky]) == False])
        if np.isnan(hshmp[ky]): hshmp.pop(ky)
    tbl = np.array([hshmp.keys(),hshmp.values()]).transpose()
    tbl = tbl[tbl[:,0].argsort()]
    ofl = open(wfl,'w')
    ofl.write('Intensity Luminance\r\n')
    np.savetxt(ofl,tbl)
    ofl.close()
    #return tbl[tbl[:,0].argsort()]

def interpLUT(tbl='curve.txt',wfl='LUT.txt',res=12):
    """
    This function takes a cleaned (i.e. by filesToTable and/or
    gprocessLUT) gamma measurement table (this can also can be given a
    file which corresponds to an appropriate table)  and linearly
    subsamples it at a given resolution so that the luminances are
    roughly evenly spaced across indicies.

    interpLUT will return a function based on numpy.interp
    which takes intensities from 0 to 1, and returns linearized
    intensities.
    """
    # Sample a linear subset of the gamma table
    tbl = np.genfromtxt(tbl,skip_header=1)
    idx = 0
    idxs = []
    itss = tbl[:,0]
    lmns = tbl[:,1]
    for smp in np.linspace(np.min(lmns),np.max(lmns),2**res):
        while smp > lmns[idx]: idx += 1
        idxs.append(idx)
    ofl = open(wfl,'w')
    ofl.write('IntensityIn IntensityOut Luminance\r\n')
    rslt = np.array([np.linspace(0,1,2**res),itss[idxs],lmns[idxs]]).transpose()
    np.savetxt(ofl,rslt)
    ofl.close()
    #return lambda x: np.interp(x,np.linspace(0,1,2**res),itss[idxs])

def gprocessTable(tbl='average.txt',ss=4000,mnns=0.1,mxns=10,wfl='curve.txt'):
    """
    This function uses a Gaussian process model to fit the gamma funciton.

    Since the dataset of 2^16 values is generally too large to run a
    complete Gaussian process on, this function breaks the dataset up
    into a set of regions ss (subsample) values long. We then build a
    set of overlapping gaussian process models, (up to two per input
    intensity) and average them at a point to get the estimation.

    gprocessTable then samples this predicted function with 2**16
    values, and saves it to a file.  This file can then be read by hrl to
    linearize the gamma function.
    """
    if type(tbl) == str: tbl = np.genfromtxt(tbl,skip_header=1)
    n = len(tbl)//ss
    dss = ss
    xsmps = tbl[:,0]
    ysmps = tbl[:,1]
    nss = np.linspace(mnns,mxns,n)
    funs = []
    xs = np.linspace(0,1,2**16)

    print 'Step 1' + ' of ' + str(n+1)
    print 'Building Prediction Function...'

    funs.append(predictor(xsmps[0:dss],ysmps[0:dss],nss[0]))

    print 'Sampling Prediction Function...'

    mx = xs[ss]
    rslts = [ ([x] + list(funs[0](x))) for x in xs[xs < mx] ]
    print 'Example Intensity: ' + str(rslts[-1][0]) + ', Predicted Luminance: ' + str(rslts[-1][1]) + ', Variance: ' + str(rslts[-1][2])

    for i in range(1,n):

        print 'Step ' + str(i+1) + ' of ' + str(n+1)
        print 'Building Prediction Function...'
        stp = ss*i
        if i == n-1:
            funs.append(predictor(xsmps[stp:],ysmps[stp:],nss[i]))
        else:
            funs.append(predictor(xsmps[stp:stp+dss],ysmps[stp:stp+dss],nss[i]))

        print 'Sampling Prediction Function...'
        mn = xs[stp]
        mx = xs[stp + ss]
        rslts += [ [x] + list((funs[0](x) + funs[1](x))/2) for x in xs[xs[xs < mx] >= mn] ]
        print 'Example Intensity: ' + str(rslts[-1][0]) + ', Predicted Luminance: ' + str(rslts[-1][1]) + ', Variance: ' + str(rslts[-1][2])
        funs.pop(0)

    print 'Step ' + str(n+1) + ' of ' + str(n+1)
    print 'Final Sampling of Prediction Function...'

    mn = xs[ss*n]
    rslts += [ [x] + list(funs[0](x)) for x in xs[xs >= mn] ]
    print 'Example Intensity: ' + str(rslts[-1][0]) + ', Predicted Luminance: ' + str(rslts[-1][1]) + ', Variance: ' + str(rslts[-1][2])

    print 'Saving to File...'
    rslt = np.array(rslts)
    ofl = open(wfl,'w')
    ofl.write('Input Luminance Variance\r\n')
    np.savetxt(ofl,rslt)
    ofl.close()

## Gaussian Process Implementation ##

def kernelMatrix(xsmps):
    return np.matrix([ [ rbfKernel(ixsmp,jxsmp) for jxsmp in xsmps ] for ixsmp in xsmps ])

def kernelColumn(x,xsmps):
    krn = np.vectorize(lambda y: rbfKernel(x,y))
    return np.matrix(krn(xsmps)).T

def rbfKernel(x1,x2,L=0.02):
    return np.exp((-(x1 - x2)**2)/(2*L**2))

def predictor(xsmps,ysmps,sg):
    """
    Returns a function for calculating an estimated y from a given x and its variance.
    """
    kmtx = kernelMatrix(xsmps)
    inv = (kmtx + sg * np.eye(len(ysmps))).I
    meanfun = lambda x: (kernelColumn(x,xsmps).T * inv * np.matrix(ysmps).T)[0,0]
    varfun = lambda x: rbfKernel(x,x) - (kernelColumn(x,xsmps).T * inv * kernelColumn(x,xsmps))[0,0]
    return lambda x: np.array([meanfun(x),varfun(x)])

def variance(xsmps,sg):
    """
    Returns a function for calculating an estimated prediciton variance.
    """
