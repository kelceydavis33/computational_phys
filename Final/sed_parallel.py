import astropy.units as u
import astropy.constants as c
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
%matplotlib inline
from astropy.io import fits
#Code written before the start of this project is stored in final_project.py
from final_project import *
#Initializing matplotlib
plt.rcParams['figure.figsize'] = (10, 10)
plt.rc('axes', labelsize=14)
plt.rc('axes', labelweight='bold')
plt.rc('axes', titlesize=16)
plt.rc('axes', titleweight='bold')
plt.rc('font', family='sans-serif')
#---
#Getting the EELG data for the SEDs
#---
eelgs = pd.read_pickle('eelgs')
sed_flux = []
sed_err = []
sed_filters = []
df = eelgs
sources = eelgs['Object']
n = 0
fs = []
fs_strings = []
a=df.index[0]
c=0
try:
    fs = []
    fs_strings = []
    for key in df.loc[a][4:17].keys():
        fs.append(float(key.split('F')[1]))
        fs_strings.append(key)
    start=4
    stop=17
    stop2 = 30
    f_final = []
    for f in fs:
        if f>600:
            f_final.append(f*.1)
        else:
            f_final.append(f)
except:
    fs = []
    fs_strings = []
    c+=1
    for key in df.loc[a][4:16].keys():
        fs.append(float(key.split('F')[1]))
        fs_strings.append(key)
    start = 4
    stop = 16
    stop2=28
    f_final = []
    for f in fs:
        if f>600:
            f_final.append(f*.1)
        else:
            f_final.append(f)
for obj in sources:
    #Increase the iteration tracker, grab the DataFrame for just this object
    n+=1
    #print(shapes[i])
    mk = np.array(df['Object']) == np.array(obj)
    a = df[mk]
    #Get the redshift and filters
    rs = a['PHOTOM_RED_SHIFT']
    filters = np.array(a.keys()[start:stop].values)
    filter_er = np.array(a.keys()[stop:stop2].values)
    #Mask the data so all filter values are greater than 3 times their error
    mask = (a[filters].values[0] > (3*a[filter_er].values[0]))& (a[filter_er].values[0]>0) 
    sed_flux.append(a[filters].values[0][mask])
    sed_err.append(a[filter_er].values[0][mask])
    sed_filters.append(np.array(f_final)[mask])
#CHANGE TO CORRECT PATH
template_paths = '/home/kelcey/computational_phys/Final/sed_templates/*.dat'
paths = glob.glob(template_paths)
ez_path = '/home/kelcey/computational_phys/Final/ez_temps/*.dat'
ez_paths = glob.glob(ez_path)
#---
#THE REDSHIFT GRID
#---
wl_new = []
dat_new = []
for path in paths:
    #wavelength in AA
    wl = np.loadtxt(path)[:,0]
    wl_new.append(wl)
    #flux in f lambda
    dat = np.loadtxt(path)[:,1]
    dat_new.append(dat)
wl_ez = []
dat_ez = []
for path in ez_paths:
    #wavelength in AA
    wl = np.loadtxt(path)[:,0]
    wl_ez.append(wl)
    #flux in f lambda
    dat = np.loadtxt(path)[:,1]
    dat_ez.append(dat)
wavelengths = []
wavelengths.extend(wl_ez)
wavelengths.extend(wl_new)
fluxes = []
fluxes.extend(dat_ez)
fluxes.extend(dat_new)
#We will look at redshfits between 0 and 10 in incriments of 0.1
xspan = np.arange(0,10,0.1)
Nx = len(xspan)
yspan = np.arange(0, len(wavelengths))
Ny = len(yspan)
zgrid = np.zeros([Nx,Ny], dtype = object)
flux_fv = []
#converting flux units:
for m in range(len(fluxes)):
    #Fluxes in Flambda
    flux_l = fluxes[m]
    #Wavelength in AA
    wave = wavelengths[m]
    source_flux = []
    for i in range(len(flux_l)):
        #c in m/s
        c_light = 2.9979246e8
        #flux in Flambda
        flamb = flux_l[i]
        #wavelength in m
        lamb = wave[i]*1e-10
        source_flux.append((lamb**2/c_light)*flamb)
    flux_fv.append(source_flux)
#Adding first row of SED matrix
for yi, y in enumerate(yspan):
    zgrid[0,yi] = np.array([wavelengths[yi], flux_fv[yi]], dtype = object)
#Adding other rows
ind = 0
for x in xspan[1:]:
    ind+=1
    for source in yspan:
        zgrid[ind,source] = np.array([(1+x)*wavelengths[source], flux_fv[source]], dtype = object)
#Adding error floor
new_sed_err = []
for errlist in sed_err:
    new_vals = []
    for val in errlist:
        if val < 10:
            new_vals.append(10.0)
        else:
            new_vals.append(val)
    new_sed_err.append(new_vals)
#Removing HST data
new_flux = []
new_filts = []
new_ferr = []
HST_filters = [81.4, 60.6, 105.0, 125.0, 160.0]
i = 0
#adjusting the flux arrays so that they do not include the HST data
for i in range(len(sed_flux)):
    fluxes = sed_flux[i]
    err = new_sed_err[i]
    fs = sed_filters[i]
    valflux = []
    valerr = []
    valfs = []
    point_ind = -1
    for f in fs:
        point_ind +=1
        if f not in HST_filters:
            valflux.append(fluxes[point_ind])
            valerr.append(err[point_ind])
            valfs.append(f)
    new_flux.append(valflux)
    new_filts.append(valfs)
    new_ferr.append(valerr)
#Getting remaining filters
used_filters = []
for n in new_filts:
    used_filters.extend(n)
used = sorted(set(used_filters))
#conversion factor for filter values
conv = 1e4
#conversion factor when getting filter values directly from name
conv_keyname = 1e-2*1e4
#first entry is filter center in AA
#second entry is leftmost end of filter in AA (bluest end)
#third entry is rightmost end of filter in AA (reddest end)
filter_dict = {115.0:[1.154*conv, 1.013*conv, 1.282*conv],
               140.0:[1.404*conv, 1.331*conv, 1.479*conv],
               150.0:[1.501*conv, 1.331*conv, 1.668*conv],
               200.0:[1.990*conv, 1.775*conv, 2.227*conv],
               277.0:[2.786*conv, 2.423*conv, 3.132*conv],
               356.0:[3.563*conv, 3.135*conv, 3.981*conv],
               410.0:[4.092*conv, 3.866*conv, 4.302*conv],
               444.0:[4.421*conv, 3.881*conv, 4.982*conv]}
#We use this cell to change our flux arrays by some multiplicative factor. This was useful when we were 
#playing with normalizing the curve but less useful now. Will re-write code to remove in the future (probably)
flux_corr = []
err_corr = []
for i in range(len(new_flux)):
    flux = np.array(new_flux[i])
    err = np.array(new_ferr[i])
    flux_corr.append(flux)
    err_corr.append(err)
#---
#Making the grid of SED plots
#---
sed_grid = np.zeros([Nx,Ny], dtype = object)
for xi, x in enumerate(xspan):
    for yi, y in enumerate(yspan):
        w = zgrid[xi,yi][0]
        f = zgrid[xi,yi][1]
        source_dict = {}
        for key in filter_dict.keys():
            lower = filter_dict[key][1]
            upper = filter_dict[key][2]
            #cover cases where there is no flux, or nothing falls within masked area
            try:
                mask = (w>lower) & (w<upper)
                source_dict[key] = np.sum(f[mask])
            except:
                source_dict[key] = 0
        sed_grid[xi, yi] = source_dict
def get_fit(num):
    """
    Generates a chi squared fit for a source in the EELGs candidate list indicated by its index
    
    Params
    ---
    num: integer index of relavent source
    
    Returns
    ---
    print statements: prints out information about the chi squared fit
    
    figure 1: chi squared figure, indicates chi squared values across the grid of possible fits by
    redshift (y axis) and index of fit in the array of templates (xaxis)
    
    figure 2: SED with highest chi squared value from grid, colors indicate NIRCam filters
    
    figure 3: Real SED being fit, colors indicate NIRCam filters
    """
    filters = new_filts[num]
    errors = err_corr[num]
    flux = flux_corr[num]
    chis = np.zeros([Nx,Ny], dtype = float)
    fmin = np.min(flux)
    fmax = np.max(flux)
    norm_flux = (flux - fmin)/(fmax - fmin)
    scaled_err = (errors - fmin)/(fmax - fmin)
    for xi, x in enumerate(xspan):
        for yi, y in enumerate(yspan):
            Nfilt = len(filters)
            chisqr = 0
            sed_vals = []
            for key in sed_grid[xi, yi].keys():
                sed_vals.append(sed_grid[xi, yi][key])
            sed_min = np.min(sed_vals)
            sed_max = np.max(sed_vals)
            for i in range(Nfilt):
                filt = filters[i]
                modelT = (sed_grid[xi,yi][filt]-sed_min)/(sed_max - sed_min)
                F = norm_flux[i]
                filterr = scaled_err[i]
                chisqr+=(((modelT-F)**2)/((filterr)**2))
            chis[xi,yi] = chisqr
    print('Minimum chi sqr: ', np.min(chis))  
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.imshow(chis, cmap = 'viridis', extent = [0,len(yspan), 0, 10], aspect = 'auto')
    plt.xlabel('Templates')
    plt.ylabel('Redshift')
    plt.title('$\chi^2$');
    plt.figure();
    plt.rcParams['figure.figsize'] = (10, 5)
    b = np.min(chis)
    x = np.where(chis == b)[0][0]
    y = np.where(chis == b)[1][0]
    print('Redshift = ', xspan[x])
    print('Grid location: x =', x,' y =', y)
    sed = []
    for key in sed_grid[x,y].keys():
        sed.append(sed_grid[x,y][key])
    sed_max = np.max(sed)
    sed_min = np.min(sed)
    simvals = (sed - sed_min)/(sed_max - sed_min)
    plt.plot(np.array(list(sed_grid[x,y].keys()))*conv_keyname, 
             simvals, marker = '.', markersize = 20, c = 'k')
    cols = ['violet','purple','c','blue','green', 'yellow', 'orange', 'r']
    c=-1
    for key in filter_dict.keys():
        c+=1
        plt.axvspan(filter_dict[key][1], filter_dict[key][2], color=cols[c],alpha = 0.2, label = f'{key}')
    plt.title('Simulated SED')
    plt.xlabel('Wavelength [AA]')
    plt.ylabel(r'Flux [$F_\nu $]');
    plt.figure()
    plt.plot(np.array(filters)*conv_keyname, norm_flux, marker = '.', markersize = 20, c = 'k')
    c = -1
    for key in filter_dict.keys():
        c+=1
        plt.axvspan(filter_dict[key][1], filter_dict[key][2], color=cols[c],alpha = 0.2, label = f'{key}')
    plt.title('Real SED')
    plt.xlabel('Wavelength [AA]')
    plt.ylabel(r'Flux [$nJy$]');
    print('')
    cvals = []
    for xi, x in enumerate(xspan):
        for yi, y in enumerate(yspan):
            cvals.append(chis[xi,yi])  
    print('The 10 most likely redshifts are: ')
    first_10 = sorted(cvals)[0:10]
    for chival in first_10:
        x = np.where(chis == chival)[0][0]
        print(x*.1, ' ',end = '')        
    print('')
    print('\nWith Chi^2 values of:')
    for i in first_10:
        print(round(i, 5),' ', end= '')
    print('')     
    print('\nFor templates numbered:')
    for chival in first_10:
        y = np.where(chis == chival)[1][0]
        print(y, ' ',end = '')
    print('') 
    return chis
#---
#Begin Parallelization
#---
master_chis = []
for v in range(len(flux_corr)):
    chi_vals = get_fit(v)
    master_chis.append(chi_vals)
np.savetxt('chi_vals.out', master_chis)