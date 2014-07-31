#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Using gaussian realizations to generate maps of galaxies and analize it with a FKP method
	
	Arthur E. M. Loureiro & Lucas F. Secco
			18/07/2014
			  IFUSP

"""
import numpy as np
import grid3D as gr
import sys
from time import clock
from scipy import interpolate
import pylab as pl
from matplotlib import cm

#################################################
# Reading the input file and converting the data
#################################################
camb_file, cell_size, n_x, n_y, n_z, num_realiz, bias, num_bins, n_bar0, realiz_type = np.loadtxt('input.dat', dtype=str)
cell_size = float(cell_size); n_x=int(n_x); n_y=int(n_y); n_z=int(n_z); num_realiz=int(num_realiz); bias=float(bias) ; num_bins=int(num_bins); realiz_type = int(realiz_type); n_bar0 = float(n_bar0);

######################
# Reading CAMB's file
######################
k_camb , Pk_camb = np.loadtxt(camb_file, unpack=True)
k_camb = np.insert(k_camb,0,0.)						
Pk_camb = np.insert(Pk_camb,0,0.)
Pk_camb_interp = interpolate.InterpolatedUnivariateSpline(k_camb,Pk_camb)	     #interpolate camb's Power Spectrum

#######################
# Initial calculations
#######################
L_x = n_x*cell_size ; L_y = n_y*cell_size ; L_z = n_z*cell_size 		     # size of the box
box_vol = L_x*L_y*L_z								     # Box's volume
print("Generating the k-space Grid...\n")
grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)					     # generates the grid
grid_bins = gr.grid3d(num_bins, num_bins, num_bins, L_x,L_y,L_z)		     # generates the bins grid
										     # multiplying the grid for the cell_size will give us a grid in physical units 
######################################
# Finding Camb's Correlation Function
######################################
print("Finding the Correlation Function...\n")
r_k=1.0*np.linspace(0.5,200.5,201)                                                   # r vector goes from 0.5 to 201 h^-1 MPc
dk_r=np.diff(k_camb)                                           			     # makes the diff between k and k + dk
dk_r=np.append(dk_r,[0.0])

krk=np.einsum('i,j',k_camb,r_k)
sinkr=np.sin(krk)
dkkPk=dk_r*k_camb*Pk_camb*np.exp(-1.0*np.power(k_camb/0.8,6.0))
rm1=np.power(r_k,-1.0)
termo2=np.einsum('i,j',dkkPk,rm1)
integrando=sinkr*termo2

corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*np.sum(integrando,axis=0)       		     # uses the trace in the r axis to make the integral
corr_g = np.log(1.+corr_ln) 	
						     # Gaussian Correl. Func.
######################################
# Finding the gaussian power spectrum
######################################
print("Calculating the Gaussian P(k)...\n Any Error Warning here is expected. \n")
dr = np.diff(r_k)
dr = np.append(dr,[0.0])
rkr = np.einsum('i,j', r_k,k_camb)
sinrk2 = np.sin(rkr)
drCorr = dr*r_k*corr_g
km1 = np.power(k_camb,-1.)
terms = np.einsum('i,j', drCorr,km1)
integrando2 = sinrk2*terms

Pk_gauss = 4.0*np.pi*np.sum(integrando2, axis=0)
Pk_gauss[0] = 0.0
#Pk_gauss_interp = interpolate.InterpolatedUnivariateSpline(k_camb,Pk_gauss)	
Pk_gauss_interp = interpolate.UnivariateSpline(k_camb,Pk_gauss)	

###############################################################
# Generating the P(K) grid using the gaussian interpolated Pkg
###############################################################
print("\nCalculating the P(k)-Grid...\n")
Pkg_vec = np.vectorize(Pk_gauss_interp)
p_matrix = Pkg_vec(grid.grid_k)
p_matrix[0][0][0] = 1. 						     # Needs to be 1.
######################
# Defining the p.d.fs 
######################
def A_k(P_):									     # The Gaussian Amplitude #
	return np.random.normal(0.0,np.sqrt(2.*P_*box_vol))			     # Zero Medium and STD=SQRT(2*P(k)*Volume)
										     # It must have the 2 factor to take the complex
										     # part into account after the iFFT

def phi_k(P_): 									     # Random regular phase #
	return (np.random.random(len(P_)))*2.*np.pi	

def delta_k_g(P_):								     # The density contrast in Fourier Space
	return A_k(P_)*np.exp(1j*phi_k(P_))		

###############################
# the log-normal density field
###############################
def delta_x_ln(d_,sigma_):
	return np.exp(bias*d_ - ((bias**2.)*(sigma_))/2.0) -1.
	
####################
# Selection funtion
####################
def phi_selec(r_):
	#f1 = np.exp(-(r_*r_)/(2.*al*al))
	#norm = np.sqrt(2.*np.pi)*al
	#return f1/norm
	a1 = np.power((r_ - 350.),2.)/((950.)**2.)
	a2 = np.power((r_-400.), 2.)/((930.)**2.)
	return (2.7)*np.exp(-a1-a2)
phi_selec_vec = np.vectorize(phi_selec)
n_bar = phi_selec_vec(grid.grid_r*cell_size)*n_bar0

N_bar_bins=np.zeros(((num_bins,num_bins,num_bins)))

################################################################
# FFT Loops for Gaussian and Gaussian + Poissonian Realizations
################################################################
k_bar = np.arange(0,num_bins,1)*(np.max(grid.grid_k)/num_bins)
inicial = clock()
pl.figure()
if realiz_type == 1:
	print "Doing both Gaussian + Poissonian realizations... \n"
	for m in range(num_realiz):
		#########################
		# gaussian density field
		#########################
		delta_x_gaus = ((delta_k_g(p_matrix).size)/box_vol)*np.fft.ifftn(delta_k_g(p_matrix))	#the iFFT
		var_gr = np.var(delta_x_gaus.real)
		var_gi = np.var(delta_x_gaus.imag)
		delta_xr_g = delta_x_gaus.real
		delta_xi_g = delta_x_gaus.imag
		###########################
		# Log-Normal Density Field
		###########################
		delta_xr = delta_x_ln(delta_xr_g, var_gr)
		delta_xi = delta_x_ln(delta_xi_g, var_gi)
		#######################
		#poissonian realization
		#######################
		N_r = np.random.poisson(n_bar*(1.+delta_xr)*(cell_size**3.))			     # This is the final galaxy Map
		N_i = np.random.poisson(n_bar0*(1.+delta_xi)*(cell_size**3.))
		n_bar0_new = np.mean(N_r)
		#############################
		# Spliting the Map into bins
		#############################
		Nx = np.split(N_r,num_bins)
		for i in range(len(Nx)):
			Nxy = np.split(Nx[i], num_bins, axis=1)
			for j in range(len(Nxy)):
				Nxyz = np.split(Nxy[j], num_bins, axis=2)
				for l in range(len(Nxyz)):
					N_bar_bins[i][j][l] = np.mean(Nxyz[l])/(cell_size**3.)
		#pol_coef = np.polyfit(grid_bins.grid_r[0][0], np.log(N_bar_bins[0][0]),4 )
		#fit = np.poly1d(pol_coef)
		#pol_coef2 = np.polyfit(grid_bins.grid_r[0][0], np.log(N_bar_bins[0][0]),3 )
		#fit2 = np.poly1d(pol_coef2)
		#pl.plot(fit(grid_bins.grid_r[0][0]),color="r")
		#pl.plot(fit2(grid_bins.grid_r[0][0]),color="k")
		pl.plot(np.log(N_bar_bins[0][0]), linewidth=1, color="k")
		##########################################
		#$%%$ AQUI SEGUE O CÓDIGO PARA O FKP $%%$#
		##########################################
		
	print "\nDone.\n"
elif realiz_type == 2:
	print "Doing Poissonian realizations only \n"
	#########################
	# gaussian density field
	#########################
	delta_x_gaus = ((delta_k_g(p_matrix).size)/box_vol)*np.fft.ifftn(delta_k_g(p_matrix))	#the iFFT
	var_gr = np.var(delta_x_gaus.real)
	var_gi = np.var(delta_x_gaus.imag)
	delta_xr_g = delta_x_gaus.real
	delta_xi_g = delta_x_gaus.imag
	###########################
	# Log-Normal Density Field
	###########################
	delta_xr = delta_x_ln(delta_xr_g, var_gr)
	delta_xi = delta_x_ln(delta_xi_g, var_gi)
	for m in range(num_realiz):
		#######################
		#poissonian realization
		#######################
		N_r = np.random.poisson(n_bar*(1.+delta_xr)*(cell_size**3.))			     # This is the final galaxy Map
		N_i = np.random.poisson(n_bar0*(1.+delta_xi)*(cell_size**3.))
		n_bar0_new = np.mean(N_r)
		#############################
		# Spliting the Map into bins
		#############################
		Nx = np.split(N_r,num_bins)
		for i in range(len(Nx)):
			Nxy = np.split(Nx[i], num_bins, axis=1)
			for j in range(len(Nxy)):
				Nxyz = np.split(Nxy[j], num_bins, axis=2)
				for l in range(len(Nxyz)):
					N_bar_bins[i][j][l] = np.mean(Nxyz[l])
		pol_coef = np.polyfit(grid_bins.grid_r[0][0], np.log(N_bar_bins[0][0]),4 )
		fit = np.poly1d(pol_coef)
		pl.plot(fit(grid_bins.grid_r[0][0]), color="r")
		##########################################
		#$%%$ AQUI SEGUE O CÓDIGO PARA O FKP $%%$#
		##########################################
	print "\nDone.\n" 
else:
	print "Error, invalid option for realization's type \n"
	sys.exit(-1)

#pl.plot(np.log(N_bar_bins[0][0]), linewidth=2)
pl.plot(np.log(phi_selec((grid_bins.grid_r[0][0]*5.*cell_size))*n_bar0),linewidth=2.5)
final = clock()
print "time = " + str(final - inicial)		
pl.figure()
pl.imshow(N_r[0], cmap=cm.jet)
pl.figure()
pl.imshow(N_i[0], cmap=cm.jet)
pl.figure()
pl.imshow(N_bar_bins[1], cmap=cm.jet, interpolation="nearest")
pl.show()






















