#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:53:57 2019

@author: fionamccarthy
"""

import numpy as np

    
Planck_ls=np.array([53,114,187,320,502,684,890,1158,1505,1956,2649])
'''
correction_3000=1/0.960
correction_857=1/0.995
correction_545=1/1.068 
correction_353=1/1.097
correction_217=1/1.119
'''
correction_3000=1
correction_857=1
correction_545=1
correction_353=1
correction_217=1
#correction_143=1.017

corrections=np.array([correction_217,correction_353,correction_545,correction_857,correction_3000])

corrections=np.outer(corrections,corrections)

#Planck_Cl_143_143=np.array([None,None,3.64e1,3.23e1,2.81e1,2.27e1,1.84e1,1.58e1,1.25e1,1.25e1,None])[2:-1]
Planck_Cl_217_217=np.array([None,None,4.17e2,2.62e2,1.75e2,1.17e2,8.82e1,6.42e1,3.34e1,4.74e1,None])[2:-1]
Planck_Cl_857_857=np.array([None,None,2.87e5,1.34e5,7.20e4,4.38e4,3.23e4,2.40e4,1.83e4,1.46e4,1.16e4])[2:-1]
Planck_Cl_545_545=np.array([None,None,6.63e4,3.34e4,1.91e4,1.25e4,9.17e3,6.83e3,5.34e3,4.24e3,3.42e3])[2:-1]
Planck_Cl_353_353=np.array([None,None,7.88e3,4.35e3,2.6e3,1.74e3,1.29e3,9.35e2,7.45e2,6.08e2,None])[2:-1]
Planck_Cl_3000_3000=np.array([None,None,5.69e5,7.89e4,3.46e4,2.90e4,1.87e4,1.43e4,1.09e4,1.05e4,None])[2:-1]

#Planck_Cl_217_143=Planck_Cl_143_217=np.array([None,None,1.04e2,7.49e1,5.87e1,3.93e1,2.64e1,2.21e1,1.07e1,1.45e1,None])[2:-1]
#Planck_Cl_353_143=Planck_Cl_143_353=np.array([None,None,3.61e2,2.32e2,1.48e2,9.42e1,6.33e1,4.56e1,2.77e1,3.53e1,None])[2:-1]
#Planck_Cl_545_143=Planck_Cl_143_545=np.array([None,None,1.01e3,5.98e2,3.77e2,2.29e2,1.54e2,1.03e2,7.09e1,5.89e1,None])[2:-1]
#Planck_Cl_857_143=Planck_Cl_143_857=np.array([None,None,1.84e3,1.06e3,6.52e2,3.86e2,2.55e2,1.76e2,1.23e2,1.03e2,None])[2:-1]


Planck_Cl_353_217=Planck_Cl_217_353=np.array([None,None,1.75e3,1.02e3,6.21e2,3.97e2,2.87e2,1.99e2,1.59e2,1.35e2,None])[2:-1]
Planck_Cl_545_217=Planck_Cl_217_545=np.array([None,None,4.97e3,2.79e3,1.65e3,1.06e3,7.41e2,5.38e2,4.30e2,3.30e2,None])[2:-1]
Planck_Cl_857_217=Planck_Cl_217_857=np.array([None,None,9.7e3,5.26e3,3.03e3,1.88e3,1.31e3,9.18e2,7e2,5.38e2,None])[2:-1]
Planck_Cl_3000_217=Planck_Cl_217_3000=np.array([None,None,5.59e2,2.34e3,1.18e3,6.88e2,2.34e2,3.55e2,2.41e2,2.58e2,None])[2:-1]


Planck_Cl_545_353=Planck_Cl_353_545=np.array([None,None,2.22e4,1.19e4,6.93e3,4.61e3,3.39e3,2.50e3,1.93e3,1.52e3,None])[2:-1]
Planck_Cl_857_353=Planck_Cl_353_857=np.array([None,None,4.30e4,2.20e4,1.25e4,7.99e3,5.88e3,4.25e3,3.24e3,2.54e3,None])[2:-1]
Planck_Cl_3000_353=Planck_Cl_353_3000=np.array([None,None,8.92e3,1.13e4,4.49e3,2.87e3,1.99e3,1.25e3,1.14e3,9.67e2,None])[2:-1]

Planck_Cl_857_545=Planck_Cl_545_857=np.array([None,None,1.30e5,6.36e4,3.53e4,2.21e4,1.63e4,1.22e4,9.31e3,7.38e3,5.91e3])[2:-1]
Planck_Cl_3000_545=Planck_Cl_545_3000=np.array([None,None,3.94e4,3.04e4,1.23e4,9.35e3,6.61e3,4.71e3,3.12e3,3.16e3,2.61e3])[2:-1]

Planck_Cl_3000_857=Planck_Cl_857_3000=np.array([None,None,1.12e5,7.49e4,3.10e4,2.24e4,1.62e4,1.17e4,8.30e3,8.04e3,6.47e3])[2:-1]

#Planck_error_143_143=np.array([None,None,0.73e1,0.35e1,0.3e1,0.29e1,0.35e1,0.91e1,1.28e1,1.28e1,None])[2:-1]
Planck_error_217_217=np.array([None,None,0.47e2,0.2e2,0.13e2,0.1e2,0.89e1,1.61e1,2.15e1,0.65e1,None])[2:-1]
Planck_error_857_857=np.array([None,None,0.37e5,0.08e5,0.26e4,0.18e4,0.09e4,0.05e4,0.03e4,0.02e4,0.01e4])[2:-1]
Planck_error_545_545=np.array([None,None,0.51e4,0.12e4,0.04e4,0.03e4,0.17e3,0.1e3,0.06e3,0.04e3,0.04e3])[2:-1]
Planck_error_353_353=np.array([None,None,0.53e3,0.18e3,0.10e3,0.07e3,0.05e3,0.33e2,0.22e2,0.16e2,None])[2:-1]
Planck_error_3000_3000=np.array([None,None,5.69e5,3.14e4,1.15e4,0.70e4,0.47e4,0.40e4,0.41e4,0.58e4,None])[2:-1]

'''
Planck_error_217_143=Planck_error_143_217=np.array([None,None,0.19e2,0.81e1,0.58e1,0.5e1,0.52e1,1.19e1,1.65e1,0.54e1,None])[2:-1]
Planck_error_353_143=Planck_error_143_353=np.array([None,None,0.62e2,0.24e2,0.14e2,1.06e1,0.83e1,0.91e1,1.11e1,0.69e1,None])[2:-1]
Planck_error_545_143=Planck_error_143_545=np.array([None,None,0.19e3,0.67e2,0.39e2,0.27e2,0.19e2,0.14e2,1.8e1,1.73e1,None])[2:-1]
Planck_error_857_143=Planck_error_143_857=np.array([None,None,0.45e3,0.12e3,0.59e2,0.41e2,0.3e2,0.23e2,0.16e2,0.15e2,None])[2:-1]
'''

Planck_error_353_217=Planck_error_217_353=np.array([None,None,0.15e3,0.06e3,0.38e2,0.27e2,0.20e2,0.14e2,0.10e2,0.05e2,None])[2:-1]
Planck_error_857_217=Planck_error_217_857=np.array([None,None,1.22e3,0.53e3,0.32e3,0.22e3,0.16e3,0.87e2,0.23e2,0.12e2,None])[2:-1]
Planck_error_545_217=Planck_error_217_545=np.array([None,None,0.48e3,0.21e3,0.12e3,0.09e3,0.63e2,0.35e2,0.12e2,0.07e2,None])[2:-1]
Planck_error_3000_217=Planck_error_217_3000=np.array([None,None,26.7e2,0.71e3,0.34e3,2.61e2,1.66e2,1.43e2,1.07e2,1.14e2,None])[2:-1]


Planck_error_545_353=Planck_error_353_545=np.array([None,None,0.16e4,0.05e4,0.23e3,0.16e3,0.11e3,0.07e3,0.04e3,0.03e3,None])[2:-1]
Planck_error_857_353=Planck_error_353_857=np.array([None,None,0.41e4,0.11e4,0.06e4,0.39e3,0.27e3,0.17e3,0.1e3,0.07e3,None])[2:-1]
Planck_error_3000_353=Planck_error_353_3000=np.array([None,None,12.4e3,0.27e4,1.15e3,0.73e3,0.45e3,0.31e3,0.3e3,3.44e2,None])[2:-1]


Planck_error_857_545=Planck_error_545_857=np.array([None,None,0.13e5,0.3e4,0.1e4,0.07e4,0.04e4,0.02e4,0.11e3,0.07e3,0.06e3])[2:-1]
Planck_error_3000_545=Planck_error_545_3000=np.array([None,None,3.88e4,0.85e4,0.33e4,2.10e3,1.34e3,0.93e3,0.76e3,1e3,1.38e3])[2:-1]

Planck_error_3000_857=Planck_error_857_3000=np.array([None,None,0.85e5,2.09e4,0.79e4,0.47e4,0.30e4,0.22e4,1.85e3,2.39e3,3.27e3])[2:-1]

Planck_cls=corrections[:,:,np.newaxis]*np.array([
                    [Planck_Cl_217_217,Planck_Cl_353_217,Planck_Cl_545_217,Planck_Cl_857_217,Planck_Cl_3000_217],
                    [Planck_Cl_217_353,Planck_Cl_353_353,Planck_Cl_545_353,Planck_Cl_857_353,Planck_Cl_3000_353],
                    [Planck_Cl_217_545,Planck_Cl_353_545,Planck_Cl_545_545,Planck_Cl_857_545,Planck_Cl_3000_545],
                    [Planck_Cl_217_857,Planck_Cl_353_857,Planck_Cl_545_857,Planck_Cl_857_857,Planck_Cl_3000_857],
                    [Planck_Cl_217_3000,Planck_Cl_353_3000,Planck_Cl_545_3000,Planck_Cl_857_3000,Planck_Cl_3000_3000]])


yerror=corrections[:,:,np.newaxis]*np.array([
                    [Planck_error_217_217,Planck_error_353_217,Planck_error_545_217,Planck_error_857_217,Planck_error_3000_217],
                    [Planck_error_217_353,Planck_error_353_353,Planck_error_545_353,Planck_error_857_353,Planck_error_3000_353],
                    [Planck_error_217_545,Planck_error_353_545,Planck_error_545_545,Planck_error_857_545,Planck_error_3000_545],
                    [Planck_error_217_857,Planck_error_353_857,Planck_error_545_857,Planck_error_857_857,Planck_error_3000_857],
                    [Planck_error_217_3000,Planck_error_353_3000,Planck_error_545_3000,Planck_error_857_3000,Planck_error_3000_3000]])          

    
    
Planck_ls_cross=np.array([163,290,417,543,670,797,923,1050,1177,1303,1430,1557,1683,1810,1937])
'''
correction_3000=1/0.960
correction_857=1/0.995
correction_545=1/1.068 
correction_353=1/1.097
correction_217=1/1.119
'''
correction_3000=1
correction_857=1
correction_545=1 
correction_353=1
correction_217=1
#correction_143=1.017

corrections=np.array([correction_217,correction_353,correction_545,correction_857])


#Planck_Cl_143_143=np.array([None,None,3.64e1,3.23e1,2.81e1,2.27e1,1.84e1,1.58e1,1.25e1,1.25e1,None])[2:-1]
Planck_Cl_217_phi= 483.48*1/1000*np.array([ 5.08 ,8.99, 3.19, 3.75, 2.15, 1.91, 1.15, 1.47, 0.66, 2.11, 3.15, 2.04 ,-0.15 ,1.84 ,3.16])
Planck_Cl_857_phi=( 2.26907) *np.array([None,None,12.34 ,14.32 ,11.08 ,8.73 ,9.00 ,8.19, 7.37 ,9.85 ,9.35 ,4.42 ,5.75 ,5.99 ,4.09])[2:]
Planck_Cl_545_phi=( 57.9766)*1/100*np.array([None,None,28.92 ,29.05, 23.38, 20.12, 21.37 ,18.32 ,15.38 ,19.36 ,18.78 ,12.94 ,12.67, 11.70, 7.13])[2:]
Planck_Cl_353_phi=(287.22)*1/1000*np.array([None,None,21.47 ,21.79, 16.56, 16.08, 14.83 ,12.76, 11.76 ,15.60 ,14.98 ,10.44 ,11.33, 10.67 ,12.76])[2:]

Planck_error_stat_217_phi= 483.48*1/1000*np.array([ 2.49, 2.17, 1.31, 1.34, 1.16, 1.24, 1.07, 0.93, 0.97, 0.90 ,0.95 ,0.90 ,0.89 ,0.94 ,1.02])
Planck_error_stat_857_phi=( 2.26907) *np.array([None,None,1.40 ,1.27 ,1.21 ,1.14, 1.14 ,1.14, 1.18 ,1.25 ,1.30 ,1.36 ,1.39, 1.43, 1.48])[2:]
Planck_error_stat_545_phi=( 57.9766)*1/100*np.array([None,None, 2.09 ,1.97 ,1.91 ,1.87 ,1.94 ,2.00 ,2.12 ,2.31 ,2.44 ,2.60 ,2.73 ,2.87, 3.04])[2:]
Planck_error_stat_353_phi=(287.22)*1/1000*np.array([None,None, 1.88 ,1.87, 1.75, 1.81, 1.79, 1.81, 1.96 ,2.13 ,2.34, 2.55, 2.78 ,3.07, 3.44])[2:]

Planck_error_sys_217_phi= 483.48*1/10000*np.array([  1.07 ,1.90 ,0.68 ,0.79 ,0.46 ,0.41 ,0.24 ,0.31 ,0.14 ,0.44, 0.65 ,0.39 ,0.10, 0.29, 0.52])
Planck_error_sys_857_phi=( 2.26907) *np.array([None,None,1.26 ,1.46 ,1.13 ,0.89 ,0.92, 0.83 ,0.75 ,1.01 ,0.96 ,0.46 ,0.60 ,0.64 ,0.45])[2:]
Planck_error_sys_545_phi=( 57.9766)*1/100*np.array([None,None,2.95 ,2.96 ,2.38, 2.05 ,2.18 ,1.87 ,1.57 ,1.98 ,1.94 ,1.36, 1.36 ,1.30 ,0.88])[2:]
Planck_error_sys_353_phi=(287.22)*1/1000*np.array([None,None,0.69 ,0.70 ,0.53 ,0.52, 0.48 ,0.41 ,0.38 ,0.50 ,0.48, 0.33 ,0.35 ,0.32 ,0.37])[2:]


#Planck_error_143_143=np.array([None,None,0.73e1,0.35e1,0.3e1,0.29e1,0.35e1,0.91e1,1.28e1,1.28e1,None])[2:-1]
#Planck_error_217_phi=np.array([None,None,0.47e2,0.2e2,0.13e2,0.1e2,0.89e1,1.61e1,2.15e1,0.65e1,None])[2:-1]
#Planck_error_857_phi=np.array([None,None,0.37e5,0.08e5,0.26e4,0.18e4,0.09e4,0.05e4,0.03e4,0.02e4,0.01e4])[2:-1]
#Planck_error_545_phi=np.array([None,None,0.51e4,0.12e4,0.04e4,0.03e4,0.17e3,0.1e3,0.06e3,0.04e3,0.04e3])[2:-1]
#Planck_error_353_phi=np.array([None,None,0.53e3,0.18e3,0.10e3,0.07e3,0.05e3,0.33e2,0.22e2,0.16e2,None])[2:-1]

Planck_cls_cross=corrections[:,np.newaxis]*np.array([Planck_Cl_217_phi[2:],Planck_Cl_353_phi,Planck_Cl_545_phi,Planck_Cl_857_phi])
Planck_errors_cross=corrections[:,np.newaxis]*np.array([Planck_error_stat_217_phi[2:]+Planck_error_sys_217_phi[2:],Planck_error_stat_353_phi+Planck_error_sys_353_phi,Planck_error_stat_545_phi+Planck_error_sys_545_phi,Planck_error_stat_857_phi+Planck_error_sys_857_phi])
                    
websky_217=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/power_spectra/websky/CIB_217.npz")
cells_217=websky_217["cells"]
webskyells=websky_217["ells"]
websky_353=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/power_spectra/websky/CIB_353.npz")
cells_353=websky_353["cells"]
webskyells=websky_353["ells"]
websky_545=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/power_spectra/websky/CIB_545.npz")
cells_545=websky_545["cells"]
webskyells=websky_545["ells"]
websky_857=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/power_spectra/websky/CIB_857.npz")
cells_857=websky_857["cells"]
webskyells=websky_857["ells"]
    
    
webskies=np.array([cells_217,cells_353,cells_545,cells_857])

websky_353_pointsremoved=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/power_spectra/websky/CIB_353_pointsremoved.npz")
cells_353_pointsremoved=websky_353_pointsremoved["cells"]
webskyells=websky_353["ells"]
websky_545_pointsremoved=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/power_spectra/websky/CIB_545_pointsremoved.npz")
cells_545_pointsremoved=websky_545_pointsremoved["cells"]

websky_857_pointsremoved=np.load("/Users/fionamccarthy/Documents/PI/Projects/kSZ_CIB/power_spectra/websky/CIB_857_pointsremoved.npz")
cells_857_pointsremoved=websky_857_pointsremoved["cells"]
webskies_pointsremoved=np.array([cells_353_pointsremoved,cells_545_pointsremoved,cells_857_pointsremoved])