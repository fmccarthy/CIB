#gather all config parameters here
import numpy as np
import multiprocessing

data_path = '/Users/jcayuso/Work/PhD/ksz/pszr/forecast/data/'

n_cores = max(1, int( multiprocessing.cpu_count() - 2 ))

################ red shift binning

z_max = 6.0 #highest red shift.
z_min = 0.1 #lowest red shift.

N_bins = 6#40 #number of red shift bins, uniform in conformal distance.

zbins_nr = N_bins #double naming to be unified. don't change, ignore.

n_samples = 120 # Number of samples used to average transfer functions inside bins 100 is very close

################ halomodel

use_halomodel = True #set false to speed up. if false, the bias is taken as z+1 and eg Pgg= b^2 P_mm_camb.
gasprofile = 'AGN' #only used if use_halomodel=True

################ LSS shot noise

lss_shot_noise = True 
LSSexperiment = 'LSST' #can be 'ngalsMpc3Fixed' or 'LSST'  #if not fixed, this selects the HODthreshold for a given experiment. Must be implemented in galaxies.py.
ngalsMpc3Fixed = 0.01 #only used when LSSexperiment='ngalsMpc3Fixed' and use_halomodel = True


################ cosmological parameters


As = 2.2 # Scalar (curvature) amplitude (times 10**-9)
ns = .96 # Scalar spectral index
Delta_c = 0.0 # Tensor chirality parameter
r = 0.0 # Tensor to scalar ratio
nt = -r/8. # Tensor spectral index
k0 = 0.05 # Pivot scale for tensors
H0 = 67.5#70.5#67.5 # Hubble constant in km s^-1 Mpc^-1
h = H0 / 100 # Reduced Hubble constant (dimensionless number)
Omega_c = 0.2647
Omega_b = 0.0528
Omega_m = Omega_c + Omega_b # 0.3175#0.274#Omega_c + Omega_b # 0.3175
Omega_r = 9.236 * 10**-5
Omega_K = 0.0
w = -1.0
wa = 0.0
zdec = 1090 # Redshift at decoupling
adec = 1 / (1 + zdec) # Scale factor at decoupling
tau = 0.06 # Optical depth to reionization

omch2 = Omega_c*h**2
ombh2 = Omega_b*h**2 # we should have Omega_m = (omch2+ombh2)/h**2.

T_CMB = 2.725*10.**6. # muK

############### Primordial spectrum mofifications


#Supression model 1
lsup1 = 0.5
ksup1 = 0.0003422

#Supression model 2
ksup2 = 0.0005261
deltasup2 = 1.14

#Oscilation model 1
Alog = 0.0278
wlog = 10**(1.51)
philog = 0.634*2*np.pi

#Oscilation model 2
Alin = 0.0292
nlin = 0.662
wlin = 10**(1.73)
philin = 0.554*2*np.pi

#Cutoff model
kcut =  10**(-3.44)

#Step in inflaton potential model
Asip = 0.374
ksip =10**(-3.10)
xsip =np.exp(0.342)

#Dipolar modulation model
w1 = 0.23



################ estimator settings

#for ksz
ksz_estim_signal_lmin = 0
ksz_estim_signal_lmax = 51 #6 #not included

#for psz
psz_estim_signal_lmin = 2 #NO DIPOLE
psz_estim_signal_lmax = 51 #not included

#for CMB
T_estim_signal_lmin = 1
T_estim_signal_lmax = 31 #6 #not included

#for E
E_estim_signal_lmin = 2
E_estim_signal_lmax = 31 #6 #not included

#small scale
estim_smallscale_lmin = 101
estim_smallscale_lmax = 10000#4000#2500#4000#10000#2000#2500#10000#10000#2500#10000#2000#10001 #3001 5001 #not included


################ CMB experimental noise

#CMB noise
cmb_experimental_noise = True #set false to switch it off
beamArcmin_T =1#7.3#1 #7.3#1#7.30 #see table 12 of https://arxiv.org/pdf/1807.06207.pdf,

 #1.0 # S44.41	4.47	4.23


noiseTuKArcmin_T =1.5#23.8174#20#20.6265/10 #20#20.6265/10 #.1#23.8174 #see table 12 of https://arxiv.org/pdf/1807.06207.pdf,  0.36e-4 square rooted and divided by 60pi/180


##.1 # 1.5 S4



beamArcmin_pol = 1.0
noiseTuKArcmin_pol =1.5# .1#1.5#.1 #1.5


################ CIB experimental noise

CIB_model="Websky"

cib_experimental_noise = True #set false to switch it off

CIB_frequencies=[353,545,857] #in GHz

fluxcuts=np.array([400,350,225,315,350,710,1000])*1/50 # see table 1 of Planck XXX. IN mJy!!

CIB_beamArcmin_T=np.array([1,1,1])#)[4.94,4.83,4.64])#see table 12 of https://arxiv.org/pdf/1807.06207.pdf
#CIB_beamArcmin_T=np.array([4.94,4.83,4.64])#see table 12 of https://arxiv.org/pdf/1807.06207.pdf

CIB_noiseTuKArcmin_T=np.array([117.085, 695.803 ,16658.7 ])*1/10 ###this is slightly too low at 353 :-/ what is going on here try 123.95 
#CIB_noiseTuKArcmin_T=np.array([123.95, 695.803 ,16658.7 ]) ###this is slightly too low at 353 :-/ what is going on here try 123.95 


################ samplings

k_min = -5  #logscale
k_max =  -1  #logscale
k_res = 30000 # 6000 per decade (might be exagerated)
kaux = np.append(np.logspace(k_min, -3, 200),np.logspace(-2.99, k_max, 2000)) # SZCosmo k-grid

k_min_tens = -5  # tensor minimum k, logscale
k_max_tens = -1  #tensor maximum k, logscale
k_res_tens = 1000 # tensor, number of points in k

k_min_GR = -5  #logscale
k_max_GR = 2  #logscale
k_res_GR = 800*(k_max_GR-k_min_GR) # number of integration points per decade

transfer_integrand_sampling = 40 # number of sampling points for ISW/Lensing integrals
