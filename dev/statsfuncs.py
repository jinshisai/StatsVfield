'''
Python script to calculate statistic functions
 like the autocorrelation function and the second-order structure function.

Developed by J. Sai.
7/23/2021
'''


# modules
import numpy as np
from scipy.fft import ifft
from scipy.fft import fft
from scipy.fft import ifftn
from scipy.fft import fftn
from scipy.fft import fftfreq
from scipy.fft import fftshift
from scipy.fft import ifftshift
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_palette('gist_earth')



# functions for debug
def gaussian2D(x, y, A, mx, my, sigx, sigy, pa=0, peak=True):
	'''
	Generate normalized 2D Gaussian

	Parameters
	----------
	 x: x value (coordinate)
	 y: y value
	 A: Amplitude. Not a peak value, but the integrated value.
	 mx, my: mean values
	 sigx, sigy: standard deviations
	 pa: position angle [deg]. Counterclockwise is positive.
	'''
	x, y   = rotate2d(x,y,pa)
	mx, my = rotate2d(mx, my, pa)


	if peak:
		coeff = A
	else:
		coeff = A/(2.0*np.pi*sigx*sigy)
	expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
	expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
	gauss=coeff*expx*expy
	return gauss


# 2D rotation
def rotate2d(x, y, angle, deg=True, coords=False):
	'''
	Rotate Cartesian coordinates.
	Right hand direction will be positive.

	array2d: input array
	angle: rotational angle [deg or radian]
	axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
	deg (bool): If True, angle will be treated as in degree. If False, as in radian.
	'''

	# degree --> radian
	if deg:
		angle = np.radians(angle)
	else:
		pass

	if coords:
		angle = -angle
	else:
		pass

	cos = np.cos(angle)
	sin = np.sin(angle)

	xrot = x*cos - y*sin
	yrot = x*sin + y*cos

	return xrot, yrot



# main functions
# autocorrelation function
def ac_1d(data):
	'''
	Calculate auto-correlation.

	Parameters
	----------


	Return
	------
	'''
	#from itertools import product

	nx   = len(data)
	d_in = data - np.nanmean(data)

	# auto-correlation
	d_ac = np.array(
		[np.sum([d_in[i]*d_in[i+j] for i in range(nx-j)])/(nx-j)
		for j in range(nx)])/np.nanvar(data)

	return d_ac


def ac_fft1(data, realfreq=False):
	'''
	Calculate auto-correlation using FFT.
	'''

	nx = len(data)
	d_in = np.r_[data - np.nanmean(data), np.zeros(nx-1)] # zero-padding

	d_ft = fft(d_in)                     # Fourier transform
	d_ft_cnj = np.conjugate(fft(d_in))   # complex conjugate

	d_ac = ifft(d_ft*d_ft_cnj).real
	d_ac /= np.r_[np.arange(1,len(d_ac)//2+2,1)[::-1], np.arange(1,len(d_ac)//2+1,1)] # weighting
	d_ac /= np.nanvar(data)

	if realfreq:
		d_ac = d_ac[:len(d_ac)//2+1]
	else:
		d_ac = fftshift(d_ac)

	return d_ac


def ac_2d(data, normalize=False):
	'''
	Calculate auto-correlation.

	Parameters
	----------


	Return
	------
	'''

	nx, ny = data.shape
	m_data = np.nanmean(data)

	d_ac = np.array([
		[np.nanmean([
			(data[i,j] - m_data)*(data[i+k, j+l] - m_data)
			for j in range(ny - l) for i in range(nx - k)])#/((nx-k)*(ny-l))
		for l in range(ny)] for k in range(nx)])
	d_ac /= np.nanvar(data)

	if normalize:
		d_ac /= d_ac[0,0]

	return d_ac


def ac_fft2(data, realfreq=False):
	nx, ny = data.shape

	d_in = data - np.nanmean(data)
	d_in = np.r_[d_in, np.zeros((d_in.shape[0]-1,d_in.shape[1]))] # zero-padding for convolution
	d_in = np.c_[d_in, np.zeros((d_in.shape[0],d_in.shape[1]-1))] # zero-padding for convolution

	d_ft = fftn(d_in)             # Fourier transform
	d_ft_cnj = np.conjugate(d_ft) # complex conjugate
	d_ac = ifftn(d_ft*d_ft_cnj).real

	# weighting with sample number
	#print(d_ac.shape[0], nx)
	wx = np.r_[np.arange(1, d_ac.shape[0]//2+2, 1)[::-1], np.arange(1,d_ac.shape[0]//2+1,1)]
	wy = np.r_[np.arange(1, d_ac.shape[1]//2+2, 1)[::-1], np.arange(1,d_ac.shape[1]//2+1,1)]
	wxx, wyy = np.meshgrid(wx, wy)
	d_ac /= (wxx*wyy)*np.nanvar(data)

	if realfreq:
		print("Resultant ACF has only the positive axis.")
		print("The output axis length is nx/2.")
		d_ac = d_ac[0:d_ac.shape[1]//2+1,0:d_ac.shape[0]//2+1]
	else:
		#pass
		d_ac = ifftshift(d_ac)

	return d_ac


# structure function
def sf_1d(data):
	'''
	Calculate the structure function.

	Parameters
	----------


	Return
	------
	'''

	nx = len(data)

	d_sf = np.array([
		np.sum([(data[i] - data[i+j])**2. for i in range(nx-j)])/(nx-j)
		for j in range(nx)
		])

	return d_sf


def sf_2d(data, normalize=False):
	'''
	Calculate auto-correlation.

	Parameters
	----------


	Return
	------
	'''

	nx, ny = data.shape

	d_sf = np.array([
		[np.sum([
			(data[i,j] - data[i+k, j+l])**2.
			for j in range(ny - l) for i in range(nx - k)])/((nx-k)*(ny-l))
		for l in range(ny)] for k in range(nx)])

	if normalize:
		d_sf /= d_sf[0,0]

	return d_sf


# for debug
def main():
	# --------- input --------
	# test with sin curve
	nx, ny = [32, 32]
	x   = np.linspace(-np.pi,np.pi,nx)
	y   = np.linspace(-np.pi,np.pi,nx)
	dx  = x[1] - x[0]
	dy  = y[1] - y[0]
	phi = 0.*np.pi # phase shift
	# ------------------------


	# ---------- start ---------
	# grid
	xx, yy = np.meshgrid(x, y, indexing='ij')
	z = np.sin(xx+phi) + np.sin(yy+phi)
	# --------------------------


if __name__ == '__main__':
	main()