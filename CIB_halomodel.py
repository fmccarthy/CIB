#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:59:35 2019

@author: fionamccarthy
"""

import scipy




import numpy as np



import remote_spectra as remote_spectra


planck=6.626e-34         # planck constant in J s
lightspeed=2.99792458e8  # speed of light in m/s
kBoltzmann=1.38e-23      # Boltzmann constant in SI units
conversion_factor=1 #1/((3e22)**4*(1e-26)**2)*(3.8e26)**2 # this goes from L_0**2/Mpc**4/Hz**2 -> Jy**2



class CIB_fluxes(object):
    
    def __init__(self,halomodel,parameters,frequencies,CIBmodel="Planck",experiment="Planck"):
    
        
        
        
        self.halomodel=halomodel
        self.zs=self.halomodel.z
        
        self.frequencies=frequencies
        
        self.model=CIBmodel
        
        # CIB model parameters:
        [alpha,T0,beta,gamma,delta,log10Meff,L0,log10Mmin,sigmasq,zplateau]=parameters
        self.alpha=alpha           # redshift evolution of temperature
        self.T0=T0                 # temperature at z=0
        self.beta=beta             # sed = nu**betaB(nu)
        self.gamma=gamma           # slope of SED at nigh frequency
        self.delta=delta           # redshift evolution of L-M relation
        self.Meff=10**log10Meff    # most efficient L-M mass
        self.L0=L0                 # normalisation of luminosity relation
        self.Mmin=10**log10Mmin    # minimum halo mass to host a galaxy
        self.sigmamsq=sigmasq    
        self.zplateau=zplateau     # z at which luminosity-mass normalisation plateaus 
        self.experiment=experiment
        
    def TD(self,z): 
            # Dust temperature of starforming galaxy, equation 51 of https://arxiv.org/pdf/1309.0382.pdf
            return self.T0*(1+z)**self.alpha


    def Planck(self,nu,T): 
            # Planck law for black body radiation. Note unnormalised and missing appropriate units.
            return 2*(nu)**3*(1/(np.exp(planck*nu/(kBoltzmann*T))-1))

    
    def dlnsed_dnu(self,nu,T):
        # equation 52 of https://arxiv.org/pdf/1309.0382.pdf
        nu=nu*1e9
        beta=self.beta
        gamma=self.gamma
        return 3+beta+planck*nu/(T*kBoltzmann)*(-1+1/(1-np.exp(planck*nu/(kBoltzmann*T))))+gamma
    
    def SED_oneT(self,nu,T):
        
        # used for computing SED at given redshift / temperature. Equation 50 of https://arxiv.org/pdf/1309.0382.pdf
        
        if self.model=="Planck" or self.model=="websky":

            nu0=(scipy.optimize.brentq(self.dlnsed_dnu,10,10000,args=T))*1e9
            
            nu=nu*1e9 # multiply by 10^^9 to go from GHz->Hz
            SED=np.zeros(nu.shape)
            beta=self.beta
            gamma=self.gamma
            
            SED[nu<nu0]=(nu[nu<nu0]/1)**beta*self.Planck(nu[nu<nu0],T)*1
            if len(nu[nu>=nu0])>0:
                SED[nu>=nu0]=(nu[nu>=nu0]/1)**-gamma*((nu0/1)**gamma)*(nu0/1)**beta*self.Planck(nu0,T)*1
       
        return SED
    
    def SED(self,nu,T):
    
        # issue: what is normalisation / z-dependence?
    
        
        if self.model=="Planck" or self.model=="websky":
                
                
            nu0s=np.array([scipy.optimize.brentq(self.dlnsed_dnu,10,10000,args=self.TD(z))for z in self.zs])*1e9
            nu=nu*1e9 #multiply by 10^9 to go from GHz->Hz
            SED=np.zeros(nu.shape)
            beta=self.beta
            gamma=self.gamma
            
            SED[nu<nu0s]=(nu[nu<nu0s]/T[nu<nu0s])**beta*self.Planck(nu[nu<nu0s],T[nu<nu0s])*T[nu<nu0s]**-4
            if len(nu[nu>=nu0s])>0:
                SED[nu>=nu0s]=(nu[nu>=nu0s]/T[nu>=nu0s])**-gamma*((nu0s[nu>=nu0s]/T[nu>=nu0s])**gamma)*(nu0s[nu>=nu0s]/T[nu>=nu0s])**beta*self.Planck(nu0s[nu>=nu0s],T[nu>=nu0s])*T[nu>=nu0s]**-4
                #SED[nu>=nu0s]=(nu[nu>=nu0s]/1)**-gamma*((nu0s[nu>=nu0s]/1)**gamma)*(nu0s[nu>=nu0s]/1)**beta*self.Planck(nu0s[nu>=nu0s],T[nu>=nu0s])*1

            SED=SED
        else: #elif self.model=="Viero": (this has no power law at high frequency)
            nu=nu*1e9 # multiply by 10^^9 to go from GHz->Hz
            beta=self.beta
            SED=(nu/T)**beta*(1/T**4)*self.Planck(nu,T)

        return SED
    
    def Sigma(self,M):
            
            # equation 54 of https://arxiv.org/pdf/1309.0382.pdf
        
            sigmasqlm=self.sigmamsq
            Meff=self.Meff
            answer= M*(1/(2*np.pi*sigmasqlm)**(1/2))*np.exp(-(np.log10(M)-np.log10(Meff))**2/(2*sigmasqlm))
            answer[M<self.Mmin]=0
            return answer
        
    def redshiftevolutionofl(self,z):
            
            # equation 53 of https://arxiv.org/pdf/1309.0382.pdf    
        
            answer=np.zeros(len(z))
            
            answer[z<self.zplateau]=(1+z[z<self.zplateau])**self.delta
            if len(answer[z>=self.zplateau])>0:
                answer[z>=self.zplateau]=(1+self.zplateau)**self.delta
            return answer
        
    def Lnu(self,nu,z,M): 
        
            # equation 55 of https://arxiv.org/pdf/1309.0382.pdf. Issue with normalisation.
            
            L0=self.L0
            sed=self.SED(nu*(1+z),self.TD(z)) 
                
            return L0*sed*self.Sigma(M)*self.redshiftevolutionofl(z)
        
        
    
    def Scentral(self,nu,z,Mhalo):
    
        # flux from luminosity; eg eq 42 of https://arxiv.org/pdf/1309.0382.pdf
        chi=remote_spectra.chifromz(z)
        answer= self.Lnu(nu,z,Mhalo)/((4*np.pi)*chi**2*(1+z)) 
        if self.L0!=1:  # implementing Scut. 
                        # Note that if you are playing with L0 you should remove this as if L0 is too high this will set all fluxes to 0.
            answer[answer>self.Scut(nu)]=0
        return answer

    def Luminosity_from_flux(self,S,z):
        
        #inverse of above
        
        return  4 * np.pi * remote_spectra.chifromz(z)**2*(1+z)*S
    
    def subhalo_mass_function(self,Msub,Mhost):
        
        
        
        if self.model=="Viero" or self.model=="websky": # websky uses a different subhalo mass function to Planck
            jiang_gamma_1  = 0.13
            jiang_alpha_1   = -0.83
            jiang_gamma_2   = 1.33
            jiang_alpha_2 = -0.02
            jiang_beta_2     = 5.67
            jiang_zeta      = 1.19
            return (((jiang_gamma_1*((Msub/Mhost)**jiang_alpha_1))+
             (jiang_gamma_2*((Msub/Mhost)**jiang_alpha_2)))*
             (np.exp(-(jiang_beta_2)*((Msub/Mhost)**jiang_zeta))))
    
        # equation 12 of https://iopscience.iop.org/article/10.1088/0004-637X/719/1/88/pdf
        
        answer= 0.3*(Msub/Mhost)**-0.7*np.exp(-9.9*(Msub/Mhost)**2.5)
        answer[Msub>Mhost]=0
        return answer

    
    def satellite_intensity(self,nu,zs,mhalos):
        satellite_masses=mhalos.copy()[::4]   # ::4 or ::2 or :: depending on how many masses you have, just to make the integration faster by taking lower resolution
    
        dndms=self.subhalo_mass_function(satellite_masses,mhalos[:,np.newaxis])
        
        return np.trapz((dndms[:,:,np.newaxis]*self.Scentral(nu,zs,satellite_masses[:,np.newaxis])[np.newaxis,:,:]),np.log(satellite_masses),axis=1)


        
    def jbar(self,freqind):
        
            # equation C3 of https://arxiv.org/pdf/1309.0382.pdf
            
            mhalos1=np.exp(self.halomodel.lnms) 
            mhalos=mhalos1[mhalos1>self.Mmin]

            jbar=1/(4*np.pi)*np.trapz(self.halomodel.nfn[mhalos1>self.Mmin]*
                   (self.Luminosity_from_flux(self.central_intensities[freqind]+self.satellite_intensities[freqind],self.zs)),
                   mhalos,axis=0)
            
            return jbar
        
    def setup_fluxes(self):
        
        self.mhalos=np.exp(self.halomodel.lnms)
        self.mhalos=self.mhalos[self.mhalos>self.Mmin]
        self.central_intensities=[]
        
        self.satellite_intensities=[]
        self.chis=remote_spectra.chifromz(self.zs)
        for frequency in self.frequencies:
            self.central_intensities.append(self.Scentral(frequency,self.zs,self.mhalos[:,np.newaxis]))
            self.satellite_intensities.append(self.satellite_intensity(frequency,self.zs,self.mhalos))
            
        self.central_intensities.append(self.Scentral(frequency,self.zs,self.mhalos[:,np.newaxis]))
        self.satellite_intensities.append(self.satellite_intensity(frequency,self.zs,self.mhalos))
        
    def Fnu(self,nu,intensities,satellites,chis,zs):
        
        #equation 14 of 1611.04517

            
        mhalos1=np.exp(self.halomodel.lnms)
        mhalos=mhalos1[mhalos1>self.Mmin]

      
    
        Fnu_integrand=self.halomodel.nfn[mhalos1>self.Mmin]*self.halomodel.halobias[mhalos1>self.Mmin]*(intensities+satellites) 
    
        return np.trapz(Fnu_integrand,mhalos,axis=0) #units are [M]*[Fnu_integrand]

    def ukm(self,zs,k):
    # from halomodel.py, I just copy and pasted so that I could calculate at any k.
    # Calculate the normalized FT of the NFW profile as a function of halo mass and z. Note, the FT is in terms of the 
    # comoving wavenumber.

     
        c = self.halomodel.conc
        mc = np.log(1+c)-c/(1.+c)
        rs = (self.halomodel.rvir3**(0.33333333))/c
        x = k*rs*(1+zs)
        Si, Ci = scipy.special.sici(x)
        Sic, Cic = scipy.special.sici((1.+c)*x)
        ukm = (np.sin(x)*(Sic-Si) - np.sin(c*x)/((1+c)*x) + np.cos(x)*(Cic-Ci))/mc

        return ukm
    def Gnu(self,zs,k,chis,intensities_nu1,intensities_nu2,satellites_nu1,satellites_nu2):

        #equation 12 of 1611.04517
    
   
        Ms=np.exp(self.halomodel.lnms)
        DENSITYPROFILE=self.ukm(zs,k)[Ms>self.Mmin]

        Gnu_integrand=self.halomodel.nfn[Ms>self.Mmin]*((intensities_nu1*satellites_nu2+intensities_nu2*satellites_nu1)*DENSITYPROFILE+satellites_nu1*satellites_nu2*DENSITYPROFILE**2)
        Ms=Ms[Ms>self.Mmin]

        return np.trapz( Gnu_integrand,Ms,axis=0) 
    
    def Cl_2halo(self,nu1,nu2,intensities_nu1,intensities_nu2,satellites_nu1,satellites_nu2,ells,Plin):
        
        # equation 11 of 1611.04517
        
        plin=np.zeros((len(ells),len(self.zs)))
        
        chis=remote_spectra.chifromz(self.zs)
        cls_2halo=np.zeros(len(ells))
        if nu1==nu2:
            Cl2_integrand=chis**2*self.Fnu(nu1,intensities_nu1,satellites_nu1,chis,self.zs)**2
        else:
            Cl2_integrand=chis**2*self.Fnu(nu1,intensities_nu1,satellites_nu1,chis,self.zs)*self.Fnu(nu2,intensities_nu2,satellites_nu2,chis,self.zs)

   

        for ell_index,ell in enumerate(ells):
            plin[ell_index]=np.array([Plin(self.zs[i],ell/chis[i])for i in range(0,len(self.zs))] )
        
        
            cls_2halo[ell_index]=np.trapz(Cl2_integrand*(plin[ell_index]),chis)

        return cls_2halo*conversion_factor
    
    def Cl_1halo(self,nu1,nu2,intensities_nu1,intensities_nu2,satellites_nu1,satellites_nu2,ells):
        
        # equation 13 of 1611.04517
        
        chis=remote_spectra.chifromz(self.zs)
        mhalos=np.exp(self.halomodel.lnms)
        mhalos=mhalos[mhalos>self.Mmin]
        
        Cl1_integrand=np.array([chis**2*self.Gnu(self.zs,ell/chis,chis,intensities_nu1,intensities_nu2,satellites_nu1,satellites_nu2)for ell in ells]) #in units of Mpc**2 [Gnu]
        return np.trapz(Cl1_integrand[:,self.zs>0.1],chis[self.zs>0.1],axis=1)*conversion_factor

    def lensing_kernel(self,chi):
        
        #possibly off by a factor of 2, check this.
        
        chi_s=remote_spectra.chifromz(1100)
        return 3/2*self.halomodel.omegam*self.halomodel.H0**2*(chi_s-chi)/(chi_s*chi)*1/(lightspeed/1000)**2*(1+remote_spectra.zfromchi(chi))
    
    
    
    def Cl_CIB_phi(self,nu,intensities,satellites,ells,Plin):
        
        
        plin=np.zeros((len(ells),len(self.zs)))
        
        chis=remote_spectra.chifromz(self.zs)
        cls_CIBkappa=np.zeros(len(ells))
        
        cls_CIBkappa_integrand=chis**2*self.Fnu(nu,intensities,satellites,chis,self.zs)*(self.lensing_kernel(chis))
      
   

        for ell_index,ell in enumerate(ells):
            plin[ell_index]=np.array([Plin(self.zs[i],ell/chis[i])for i in range(0,len(self.zs))] )
        
        
            cls_CIBkappa[ell_index]=np.trapz(cls_CIBkappa_integrand*plin[ell_index],chis)/ell**2

        return cls_CIBkappa*np.sqrt(conversion_factor)
    
   
        
        
    def compute_auto_cls(self,ells,Plin):
        
        self.auto_Cls=np.zeros((len(self.frequencies),len(self.frequencies),len(ells)))
        self.Cls_1halo=np.zeros((len(self.frequencies),len(self.frequencies),len(ells)))
        self.Cls_2halo=np.zeros((len(self.frequencies),len(self.frequencies),len(ells)))

        
        for freqind1, nu1 in enumerate(self.frequencies):
            for freqind2, nu2 in enumerate(self.frequencies):
                if freqind1<=freqind2:

                    Cls_1halo=self.Cl_1halo(nu1,nu2,self.central_intensities[freqind1],self.central_intensities[freqind2],
                                            self.satellite_intensities[freqind1],self.satellite_intensities[freqind2],ells)
                    
                    Cls_2halo=self.Cl_2halo(nu1,nu2,self.central_intensities[freqind1],self.central_intensities[freqind2],
                                            self.satellite_intensities[freqind1],self.satellite_intensities[freqind2],ells,Plin)
                    self.Cls_1halo[freqind1,freqind2]=self.Cls_1halo[freqind2,freqind1]=Cls_1halo
                    self.Cls_2halo[freqind1,freqind2]=self.Cls_2halo[freqind2,freqind1]=Cls_2halo
                    self.auto_Cls[freqind1,freqind2]=self.auto_Cls[freqind2,freqind1]=(Cls_1halo+Cls_2halo)
        
    def recalibrate_L0(self,reference_value,value):
        # reference value is some measurement of Cl. value is the value originally computed here. note that there is no Scut implemented.
        previousL0=self.L0
        newL0=np.sqrt(reference_value/value)
        self.L0=newL0
        for x in range(0,len(self.frequencies)):
            a=self.central_intensities[x]*self.L0/previousL0
            a[a>self.Scut(self.frequencies[x])]=0
            self.central_intensities[x]=a
            b=self.satellite_intensities[x]*self.L0/previousL0
            self.satellite_intensities[x]=b            
            
        self.Cls_1halo=self.Cls_1halo*self.L0**2/previousL0**2
        self.Cls_2halo=self.Cls_2halo*self.L0**2/previousL0**2
        self.auto_Cls=self.auto_Cls*self.L0**2/previousL0**2
        self.Cls_cibphi=self.Cls_cibphi*self.L0/previousL0


    def compute_lensing_cib_cls(self,ells,Plin):
        
        self.Cls_cibphi=np.zeros((len(self.frequencies),len(ells)))
        for freqind, nu in enumerate(self.frequencies):
            self.Cls_cibphi[freqind]=self.Cl_CIB_phi(nu,self.central_intensities[freqind],self.satellite_intensities[freqind],ells,Plin)#*norm
    
    
    def shot_noise_chi_integral(self,nu,Lcuts,mhalos):

        # I think this is an old version that I haven't fixed, probably is wrong.
        
        integrand=np.zeros(mhalos.shape)
        freqind=list(self.frequencies).index(nu)
        Lnu=self.Luminosity_from_flux(self.central_intensities[freqind]+self.satellite_intensities[freqind],self.zs)/(4*np.pi)
        Ms=np.exp(self.halomodel.lnms)

        integrand=self.halomodel.nfn[Ms>self.Mmin]*(Lnu)**2
       
        
        for i in range(0,integrand.shape[0]):
            for j in range(0,integrand.shape[1]):
    
                if Lnu[i,j]>Lcuts[j]:
                    integrand[i,j]=0
                    
        
        
        return np.trapz(integrand,self.mhalos[self.mhalos>self.Mmin],axis=0)
    
    def Scut(self,nu):
        
          #cut off frequencies, experiment-dependent.  
        
          experiment=self.experiment

          if experiment=="Planck":
              fluxcuts=np.array([400,350,225,315,350,710,1000])*1e-3   #see table 1 of https://arxiv.org/pdf/1309.0382.pdf
              frequencies=[100,143,217,353,545,857,3000] # in gHz!!
        
              if nu in frequencies:
                  return fluxcuts[frequencies.index(nu)]
              else:
                  return 1000000000000 #shouldn't be here....
          elif experiment=="Ccatprime":
              frequencies=[220,280,350,410,850,3000]
              fluxcuts=np.array([225,300,315,350,710,1000])*1e-3  # i don't actually know what the fluxcuts are for ccatp
                                                                  # so I used Planck values at the frequencies relevant for Planck
                                                                  # and I put in arbitrary values in between the Planck values for
                                                                  # the other frequencies. 
              if nu in frequencies:
                  return fluxcuts[frequencies.index(nu)]
         
                  
    def prob(self,dummys,logexpectation_s,sigma):
        return 1/np.sqrt((2*np.pi*sigma**2))*np.exp(-(dummys[:,np.newaxis,np.newaxis]-logexpectation_s)**2/(2*sigma**2))
              
    def dndlns(self,dummys,logexpectation_s,sigma):
        mhalos=np.exp(self.halomodel.lnms)
        nfn=self.halomodel.nfn[np.newaxis,mhalos>self.Mmin]
        p=self.prob(dummys,logexpectation_s,sigma)
        integrand=nfn*p
      
        return np.trapz(integrand,mhalos[mhalos>self.Mmin],axis=1)     
    def shot_noise(self,freqind,sigma):
        chis=remote_spectra.chifromz(self.zs)
        centrals=self.central_intensities[freqind]
        
        centrals[centrals==0]=1e-100
        logcentrals=np.log(centrals)
        dummylogs=np.linspace(np.min(logcentrals[logcentrals>-200])-0.5,self.Scut(self.frequencies[freqind]),200)
        dnds=self.dndlns(dummylogs,logcentrals,sigma)
       
         
        
        return np.trapz(chis**2*(np.trapz(dnds*np.exp(dummylogs[:,np.newaxis])**2,dummylogs,axis=0)),chis)

    '''
    def shot_noise(self,nu1,nu2,zs,mhalos):
        chis=remote_spectra.chifromz(zs) 
    
        #Scuts=self.Scut(nu,self.experiment)  
        
        freqs=list(self.frequencies)
        
        centrals1=self.central_intensities[freqs.index(nu1)]
        centrals2=self.central_intensities[freqs.index(nu2)]

       # centrals[centrals>Scuts]=0
        
       # satellites=self.satellite_intensities[freqs.index(nu)]
        mhalos=np.exp(self.halomodel.lnms)
    
        entire_flux1=centrals1#+satellites
        entire_flux1[entire_flux1>self.Scut(nu1)]=0
        entire_flux2=centrals2#+satellites
        entire_flux2[entire_flux2>self.Scut(nu2)]=0
        
        integrand=np.trapz(chis**2* self.halomodel.nfn[mhalos>self.Mmin]*entire_flux1*entire_flux2,mhalos[mhalos>self.Mmin],axis=0)
                                        #in 1/MPc**2*[shot_noise_chi_integral]=1/MPc**2*[stellar_mass_function]*L**2 = Mpc**-5*[L]**2=
                                        #=Mpc**-5* [solar luminosity/Hz]**2
        
        shot_noise=np.trapz(integrand,chis) #in MPc*[integrand]=1/Mpc*[shot_noise_chi_integral]=Mpc**-4*[L]**2=
            
        
        return shot_noise
    '''
    def compute_shot_noises(self):
        
        self.shot_noises=np.zeros((len(self.frequencies),len(self.frequencies)))
        for nu_ind,nu in enumerate(self.frequencies):
                self.shot_noises[nu_ind,nu_ind]=self.shot_noise(nu_ind,0.3)
        for nu_ind,nu  in enumerate(self.frequencies):
            for nu_ind2,nu2  in enumerate(self.frequencies):
                self.shot_noises[nu_ind,nu_ind2]=np.sqrt(self.shot_noises[nu_ind,nu_ind]*self.shot_noises[nu_ind2,nu_ind2])
        if self.experiment=="Planck":

            for nu_ind, nu in enumerate(self.frequencies):
                for nu_ind2,nu2  in enumerate(self.frequencies):
                    self.shot_noises[nu_ind,nu_ind2]=sn(nu,nu2)
        elif self.experiment=="Ccatprime":
            for nu_ind, nu in enumerate(self.frequencies):
                    for nu_ind2,nu2  in enumerate(self.frequencies):

                      # if nu2==220 or nu2==350 or nu2==850 or nu2==3000:
                           self.shot_noises[nu_ind,nu_ind2]=sn(nu,nu2)
def correction(frequency):
    correction_3000=0.960
    correction_857=0.995
    correction_545=1.068 
    correction_353=1.097
    correction_217=1.119
    if frequency==3000:
        return correction_3000
    elif frequency==857:
        return correction_857
    elif frequency==545:
        return correction_545
    elif frequency==353:
        return correction_353
    elif frequency==217:
        return correction_217
    else:
        print("no colour correction for frequency",frequency)
        return 1
def sn(nu1,nu2,experiment="Planck"):
    
        # shot noise parameters as listed in table 9 of https://arxiv.org/pdf/1309.0382.pdf
        
        # for ccatp I use Planck shot noise values and for frequencies that are not shared I 
        # round up to the next Planck frequency (most conservative thing to do)
    
        if nu1==350 or nu1==280:
            nu1=353
        if nu2==350 or nu2==280:
            nu2=353
        if nu1==220:
            nu1=217
        if nu2==220:
            nu2=217
        if nu1==410:
            nu1=545
        if nu2==410:
            nu2=545
        if nu1==850:
            nu1=857
        if nu2==850:
            nu2=857
        if experiment=="Planck":
            if [nu1,nu2]==[3000,3000]:
                ans=  9585
            if [nu1,nu2]==[3000,857]or [nu1,nu2]==[857,3000]:
                ans= 4158
            if [nu1,nu2]==[3000,545]or [nu1,nu2]==[545,3000]:
                ans= 1449
            if [nu1,nu2]==[3000,353]or [nu1,nu2]==[353,3000]:
                ans= 411
            if [nu1,nu2]==[3000,217] or [nu1,nu2]==[217,3000]:
                ans= 95
            if [nu1,nu2]==[857,857]:
                ans= 5364
            if [nu1,nu2]==[857,545] or [nu1,nu2]==[545,857]:
                ans= 2702
            if [nu1,nu2]==[857,353] or [nu1,nu2]==[353,857]:
                ans= 953
            if [nu1,nu2]==[857,217] or [nu1,nu2]==[217,857]:
                ans= 181
            if [nu1,nu2]==[545,545]:
                ans= 1690
            if [nu1,nu2]==[545,353] or [nu1,nu2]==[353,545]:
                ans= 626
            if [nu1,nu2]==[545,217] or [nu1,nu2]==[217,545]:
                ans= 121
            if [nu1,nu2]==[353,353] :
                ans= 262
            if [nu1,nu2]==[353,217]or [nu1,nu2]==[217,353]:
                ans= 54
            if [nu1,nu2]==[217,217]:
                ans= 21
            return ans*1/correction(nu1)*1/correction(nu2)  # is the correction included? it is not clear to me.
                                    
        
        else:
            print("no shot noise at",nu1,nu2,experiment)
            return 0    
        
def frequency_correction(nu):
    
        # frequency corrections relevant for Planck spectra
    
        if nu==217: 
            return 1.119
        if nu==353: 
            return 1.097
        if nu==545: 
            return 1.068
        if nu==857:
            return  0.995
        if nu==3000: 
            return 1.096
        else:
            return 1
        
def main(argv):    
    
    cibmodel = CIB_fluxes()


if (__name__ == "__main__"):
    main(sys.argv[1:])

