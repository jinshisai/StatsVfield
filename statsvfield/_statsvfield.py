'''
Python script to calculate statistic functions
 like the autocorrelation function (ACF), the second-order structure function (SF)
 and so on.

Developed by J. Sai.
7/23/2021
8/19/2021
'''


# modules
import numpy as np
from scipy.fft import fft, ifft, fftn, ifftn, fftfreq, fftshift, ifftshift
from scipy.fft import rfftfreq
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import seaborn as sns
#sns.set_palette('gist_earth')



# Class StatsVF
class StatsVfield():

	def __init__(self, data, axes, derr=[]) -> None:
		self.data      = data
		self.datashape = data.shape
		self.ndim      = len(data.shape)
		self.derr      = derr

		if type(axes) == list:
			if len(axes) != self.ndim:
				print ('ERROR: Dimension of given data and axes do not match.')
				return
		elif type(axes).__name__ == 'ndarray':
			if len(axes.shape) != self.ndim:
				print ('ERROR: Dimension of given data and axes do not match.')
				return
		else:
			print ('ERROR: axes must be list or ndarray containing xi, or ndarray of x.')
			return

		if self.ndim == 1:
			self.nx = self.datashape[0]
			if type(axes) == list:
				self.x  = axes[0]
			elif type(axes).__name__ == 'ndarray':
				self.x = axes
			self.dx = self.x[1] - self.x[0]
		elif self.ndim == 2:
			self.nx, self.ny = self.datashape
			self.x, self.y = axes
			self.dx = self.x[1] - self.x[0]
			self.dy = self.y[1] - self.y[0]
		elif self.ndim == 3:
			self.nx, self.ny, self.nz = self.datashape
			self.x, self.y, self.z = axes
			self.dx = self.x[1] - self.x[0]
			self.dy = self.y[1] - self.y[0]
			self.dz = self.z[1] - self.z[0]
		elif self.ndim > 3:
			print ('ERROR: Dimension must be <= 3.')
			return

		self.acf   = []
		self.sf    = []
		self.tau_x = []


	def calc_sf(self, p_order=2):
		'''
		Calculate the second-order structure function (SF).
		 Other orders will be supported in future.

		Usage
		-----
		vf = StatsVfield(data, axes)
		vf.calc_sf()
		vf.sf # call the calculated SF

		Parameters
		----------
		 - p_order: Order of the structuer function. Currently not used.
		'''
		if self.ndim == 1:
			if len(self.derr) == 0:
				self.sf = sf_1d(self.data)
			else:
				self.sf, self.sf_err = sf_1d(self.data, derr=self.derr)
		elif self.ndim == 2:
			if len(self.derr) == 0:
				self.sf = sf_2d(self.data)
			else:
				self.sf, self.sf_err = sf_2d(self.data, derr=self.derr)
		elif self.ndim == 3:
			print ('3D is being developed.')
			return

		self.get_tau(realfreq=True)


	def calc_ac(self, method='FFT', realfreq=False):
		'''
		Calculate autocorrelation (AC).

		Usage
		-----
		vf = StatsVfield(data, axes)
		vf.calc_ac()
		vf.acf # call the calculated ACF

		Parameters
		----------
		 - method: Calculation ways; FFT or iterative. FFT mode uses Fast Fourier Transform, while
		  iterative mode calculates ACF iteratively sliding an input data set.
		 - realfreq: If True, only ACF within positive tau will be return. Option only for in one-dimensional data set.
		'''
		if self.ndim == 1:
			if method == 'FFT':
				self.acf = ac_fft1(self.data, realfreq=realfreq)
			elif method == 'iterative':
				if len(self.derr) == 0:
					self.acf = ac_1d(self.data, realfreq=realfreq)
				else:
					self.acf, self.acf_err = ac_1d(self.data, derr=self.derr, realfreq=realfreq)
		elif self.ndim == 2:
			if method == 'FFT':
				self.acf = ac_fft2(self.data)
			elif method == 'iterative':
				if len(self.derr) == 0:
					self.acf = ac_2d(self.data)
				else:
					self.acf, self.acf_err = ac_2d(self.data, derr=self.derr)

		#if len(self.tau_x) == 0:
		self.get_tau(realfreq=realfreq)


	def calc_ps(self, method='FFT', realfreq=False):
		'''
		Calculate power-spectrum (PS). Still under development.

		Usage
		-----
		Coming soon..
		'''
		if self.ndim == 1:
			self.ps = pspec_1d(self.data, realfreq=realfreq)
		elif self.ndim == 2:
			print ('Still being developed, sorry.')
			#self.ps = pspec_2d(self.data, realfreq=realfreq)

		if realfreq:
			self.freq_x = rfftfreq(self.nx + self.nx - 1, self.dx) # nx -1 is for zero-padding
		else:
			self.freq_x = fftshift(fftfreq(self.nx + self.nx - 1, self.dx))

		#print(len(self.ps), len(self.freq_x))


	def get_tau(self, realfreq=False):
		'''
		Get tau for ACF and SF.

		Parameters
		----------
		 - realfreq: For one-dimensional data set, if True, only positive tau will be returned.
		'''
		if self.ndim == 1:
			if realfreq:
				self.tau_x = np.arange(0, self.nx, 1)*self.dx
			else:
				self.tau_x = np.concatenate([np.arange(-(self.nx - 1), 0, 1)*self.dx, np.arange(0, self.nx, 1)*self.dx])
		elif self.ndim == 2:
			self.tau_x = np.concatenate([np.arange(-(self.nx - 1), 0, 1)*self.dx, np.arange(0, self.nx, 1)*self.dx])
			self.tau_y = np.concatenate([np.arange(-(self.ny - 1), 0, 1)*self.dy, np.arange(0, self.ny, 1)*self.dy])
		elif self.ndim == 3:
			print ('3D is being developed.')
			return


	def collapse(self):
		if self.ndim == 1:
			print ('Data is one dimensional. No more collapse.')
			return
		elif self.ndim == 2:
			tau_xx, tau_yy = np.meshgrid(self.tau_x, self.tau_y)
			tau_rr         = np.sqrt(tau_xx*tau_xx + tau_yy*tau_yy)
			tau_sort       = np.unique(tau_rr)
			self.tau_col   = tau_sort

		if len(self.acf) != 0:
			self.acf_col = np.array([
				np.nanmean(self.acf[tau_rr == tau_i]) for tau_i in tau_sort])
			self.acf_err_col = np.array([
				np.sqrt(np.nansum(self.acf_err[tau_rr == tau_i]**2))/np.count_nonzero(~np.isnan(self.acf_err[tau_rr == tau_i]))
				 for tau_i in tau_sort])

		if len(self.sf) !=0:
			self.sf_col = np.array([
				np.nanmean(self.sf[tau_rr == tau_i]) for tau_i in tau_sort])
			self.sf_err_col = np.array([
				np.sqrt(np.nansum(self.sf_err[tau_rr == tau_i]**2))/np.count_nonzero(~np.isnan(self.sf_err[tau_rr == tau_i]))
				 for tau_i in tau_sort])


	def get_tauzero(self):

		if self.ndim == 2:
			print ('Currently get_tauzero only supports one-dimensional data.')
			return

		if 'acf' in self.__dict__.keys():
			indx = [i for i in range(len(self.acf)-1) if self.acf[i]*self.acf[i+1] <=0]
			if len(indx) > 0:
				indx_tau0 = indx[0]
				self.tau0 = self.tau_x[indx_tau0]
			else:
				self.tau0 = np.nan
		else:
			print ('ACF is not found. Calculate ACF first by vf.calc_ac().')
			return


	def sf_plawfit(self, pini, taurange=[], cutzero=True):
		'''
		'''

		from scipy.optimize import leastsq

		# fit function
		# power law
		plaw    = lambda x, param: param[0]*(x**(param[1]))
		errfunc = lambda param, x, y: plaw(x, param) - y
		#res = leastsq(errfunc, [1e-3, -3], args=(freq_fft[1:], np.abs(res_spec[1:])**2.))

		# linear
		fln      = lambda x, param: param[0] + param[1]*x
		errfunc2 = lambda param, x, y: fln(x, param) - y

		# fit param
		if cutzero:
			tau_fit = self.tau_x[1:]
			sf_fit  = self.sf[1:]
		else:
			tau_fit = self.tau_x
			sf_fit  = self.sf

		# fitting range
		if len(taurange) == 2:
			where_fit = (tau_fit > taurange[0]) & (tau_fit <= taurange[-1])
			sf_fit    = sf_fit[where_fit]
			tau_fit   = tau_fit[where_fit]

		#res = leastsq(errfunc2, [-3, -3], args=(np.log10(tau_sf[where_fit]), np.log10(sf_slice[where_fit])))
		#p_out = res[0]
		res = leastsq(errfunc2, pini, args=(np.log10(tau_fit), np.log10(sf_fit)))
		pout = res[0]
		self.fit_results = dict({'pini': pini, 'pout': pout})



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


# main functions
# autocorrelation function
def ac_1d(data, derr=[], realfreq=True):
	'''
	Calculate auto-correlation.

	Parameters
	----------


	Return
	------
	'''
	#from itertools import product

	nx   = len(data)
	d_in = data.copy() - np.nanmean(data)

	if realfreq:
		# auto-correlation
		d_ac = np.array([
			np.nanmean(d_in[0:nx-j]*d_in[j:nx]) for j in range(nx)])/np.nanvar(data)
	else:
		# zero-padding
		d_in = np.concatenate([d_in, np.zeros(nx-1)])
		d_shift = data.copy() - np.nanmean(data)
		d_shift = np.concatenate([np.zeros(nx-1), d_shift])

		# replace zero with nan to skip
		d_in[d_in == 0.] = np.nan
		d_shift[d_shift == 0.] = np.nan

		nx_out = 2*nx - 1
		d_ac = np.array([
			np.nanmean(d_in[0:nx_out-i]*d_shift[i:nx_out]) for i in range(nx_out)
			])/np.nanvar(data)

	if len(derr) == 0:
		return d_ac
	else:
		# error propagation
		if realfreq:
			d_in_err = derr.copy() # assuming error of mean can be ignored
			d_ac_err = np.array([
				np.sqrt(np.nansum((d_in[0:nx-j]*d_in_err[j:nx])**2\
					+ (d_in[j:nx]*d_in_err[0:nx-j])**2 ))\
					/np.count_nonzero(~np.isnan(d_in[0:nx-j]*d_in[j:nx])) for j in range(nx)])/np.nanvar(data)
		else:
			# zero-padding
			d_in_err    = np.concatenate([derr, np.zeros(nx-1)])
			d_shift_err = np.concatenate([np.zeros(nx-1), derr])
			d_in_err[d_in_err == 0.]       = np.nan
			d_shift_err[d_shift_err == 0.] = np.nan

			# error of each element:
			#  (m1 +/- sig1)*(m2 +/- sig2) = m1*m2 +/- sqrt((m1*sig2)^2 + (m2*sig1)^2)
			# error of mean
			#  sqrt(Sum(sig_i^2))/N
			d_ac_err = np.array([
				np.sqrt(np.nansum((d_in[0:nx_out-i]*d_shift_err[i:nx_out])**2 \
					+ (d_in_err[0:nx_out-i]*d_shift[i:nx_out])**2))\
				/np.count_nonzero(~np.isnan(d_in[0:nx_out-i]*d_shift[i:nx_out])) for i in range(nx_out)
				])/np.nanvar(data)
		return d_ac, d_ac_err


def ac_fft1(data, realfreq=False):
	'''
	Calculate auto-correlation using FFT.
	'''

	nx = len(data)
	d_in = np.r_[data - np.nanmean(data), np.zeros(nx-1)] # zero-padding

	d_ft = fft(d_in)                     # Fourier transform
	d_ft_cnj = np.conjugate(fft(d_in))   # complex conjugate

	d_ac = ifft(d_ft*d_ft_cnj).real
	d_ac /= np.r_[np.arange(1,nx+1,1)[::-1], np.arange(1,nx,1)] # weighting
	d_ac /= np.nanvar(data)

	if realfreq:
		d_ac = d_ac[:len(d_ac)//2+1]
	else:
		d_ac = fftshift(d_ac)

	return d_ac


def ac_2d(data, derr=[]):
	'''
	Calculate auto-correlation.

	Parameters
	----------


	Return
	------
	'''

	nx, ny = data.shape

	# zero-padding for convolution
	d_in = data.copy() - np.nanmean(data)
	d_in = np.r_[d_in, np.zeros((d_in.shape[0]-1,d_in.shape[1]))]
	d_in = np.c_[d_in, np.zeros((d_in.shape[0],d_in.shape[1]-1))]

	d_shift = data.copy() - np.nanmean(data)
	d_shift = np.r_[np.zeros((d_shift.shape[0]-1,d_shift.shape[1])), d_shift]
	d_shift = np.c_[np.zeros((d_shift.shape[0],d_shift.shape[1]-1)), d_shift]

	# replace zero with nan to skip
	d_in[d_in == 0.] = np.nan
	d_shift[d_shift == 0.] = np.nan

	# autocorrelation
	nx_out = 2*nx - 1
	ny_out = 2*ny - 1

	d_ac = np.array([
		[np.nanmean(
			d_in[:nx_out - k, :ny_out - l] * d_shift[k:nx_out, l:ny_out])
		for l in range(ny_out)] for k in range(nx_out)])

	d_ac /= np.nanvar(data)


	if len(derr) == 0:
		return d_ac
	else:
		# error propagation

		# zero-padding
		d_in_err = derr.copy()
		d_in_err = np.r_[d_in_err, np.zeros((d_in_err.shape[0]-1, d_in_err.shape[1]))]
		d_in_err = np.c_[d_in_err, np.zeros((d_in_err.shape[0], d_in_err.shape[1]-1))]

		d_shift_err = derr.copy()
		d_shift_err = np.r_[np.zeros((d_shift_err.shape[0]-1, d_shift_err.shape[1])), d_shift_err]
		d_shift_err = np.c_[np.zeros((d_shift_err.shape[0], d_shift_err.shape[1]-1)), d_shift_err]

		d_in_err[d_in_err == 0.]       = np.nan
		d_shift_err[d_shift_err == 0.] = np.nan

		# error of each element:
		#  (m1 +/- sig1)*(m2 +/- sig2) = m1*m2 +/- sqrt((m1*sig2)^2 + (m2*sig1)^2)
		# error of mean
		#  sqrt(Sum(sig_i^2))/N
		d_ac_err = np.array([[
			np.sqrt(np.nansum((d_in[:nx_out - k, :ny_out - l]*d_shift_err[k:nx_out, l:ny_out])**2 \
					+ (d_in_err[:nx_out - k, :ny_out - l]*d_shift[k:nx_out, l:ny_out])**2))\
				/np.count_nonzero(~np.isnan(d_in[:nx_out - k, :ny_out - l]*d_shift[k:nx_out, l:ny_out]))
				for l in range(ny_out)] for k in range(nx_out)]
				)/np.nanvar(data)
		return d_ac, d_ac_err


def ac_fft2(data):
	nx, ny = data.shape

	d_in = data.copy()
	d_in[np.isnan(d_in)] = 0. # fill nan with zero
	d_in -= np.nanmean(data)

	# zero-padding
	d_in = np.r_[d_in, np.zeros((d_in.shape[0]-1,d_in.shape[1]))] # zero-padding for convolution
	d_in = np.c_[d_in, np.zeros((d_in.shape[0],d_in.shape[1]-1))] # zero-padding for convolution

	d_ft = fftn(d_in)             # Fourier transform
	d_ft_cnj = np.conjugate(d_ft) # complex conjugate
	d_ac = ifftn(d_ft*d_ft_cnj).real

	# weighting with sample number
	#print(d_ac.shape[0], nx)
	wx = np.concatenate([np.arange(1, nx+1, 1), np.arange(nx-1, 0, -1)])
	wx = ifftshift(wx)
	wy = np.concatenate([np.arange(1, ny+1, 1), np.arange(ny-1, 0, -1)])
	wy = ifftshift(wy)
	#wx = np.r_[np.arange(1, d_ac.shape[0]//2+2, 1)[::-1], np.arange(1,d_ac.shape[0]//2+1,1)]
	#wy = np.r_[np.arange(1, d_ac.shape[1]//2+2, 1)[::-1], np.arange(1,d_ac.shape[1]//2+1,1)]
	wxx, wyy = np.meshgrid(wx, wy)
	d_ac /= (wxx*wyy)*np.nanvar(data)

	#if realfreq:
	#	print("Resultant ACF has only the positive axis.")
	#	print("The output axis length is nx/2.")
	#	d_ac = d_ac[0:d_ac.shape[1]//2+1,0:d_ac.shape[0]//2+1]
	#else:

	d_ac = ifftshift(d_ac)

	return d_ac


# structure function
def sf_1d(data, derr=[]):
	'''
	Calculate the structure function.

	Parameters
	----------


	Return
	------
	'''

	nx = len(data)

	d_sf = np.array([
		np.nanmean((data[:nx-i] - data[i:nx])**2.) for i in range(nx)
		])

	if len(derr) == 0:
		return d_sf
	else:
		# error propagation
		d_sf_err = np.array([
			np.sqrt(np.nansum((4.* (data[:nx-i] - data[i:nx])**2. * (derr[:nx-i]**2 + derr[i:nx]**2.))))\
			/np.count_nonzero(~np.isnan((data[:nx-i] - data[i:nx]))) for i in range(nx)
			])

		return d_sf, d_sf_err


def sf_2d(data, derr=[], normalize=False):
	'''
	Calculate auto-correlation.

	Parameters
	----------


	Return
	------
	'''

	nx, ny = data.shape

	# zero-padding for convolution
	d_in = data.copy() - np.nanmean(data)
	d_in = np.r_[d_in, np.zeros((d_in.shape[0]-1,d_in.shape[1]))]
	d_in = np.c_[d_in, np.zeros((d_in.shape[0],d_in.shape[1]-1))]

	d_shift = data.copy() - np.nanmean(data)
	d_shift = np.r_[np.zeros((d_shift.shape[0]-1,d_shift.shape[1])), d_shift]
	d_shift = np.c_[np.zeros((d_shift.shape[0],d_shift.shape[1]-1)), d_shift]

	# replace zero with nan to skip
	d_in[d_in == 0.] = np.nan
	d_shift[d_shift == 0.] = np.nan

	# structure function
	nx_out = 2*nx - 1
	ny_out = 2*ny - 1

	d_sf = np.array([[
		np.nanmean(
			(d_in[:nx_out - k, :ny_out - l] - d_shift[k:nx_out, l:ny_out])**2. )
		for l in range(ny_out)] for k in range(nx_out)])

	if normalize:
		d_sf /= d_sf[0,0]

	if len(derr) == 0:
		return d_sf
	else:
		# error propagation

		# zero-padding
		d_in_err = derr.copy()
		d_in_err = np.r_[d_in_err, np.zeros((d_in_err.shape[0]-1, d_in_err.shape[1]))]
		d_in_err = np.c_[d_in_err, np.zeros((d_in_err.shape[0], d_in_err.shape[1]-1))]

		d_shift_err = derr.copy()
		d_shift_err = np.r_[np.zeros((d_shift_err.shape[0]-1, d_shift_err.shape[1])), d_shift_err]
		d_shift_err = np.c_[np.zeros((d_shift_err.shape[0], d_shift_err.shape[1]-1)), d_shift_err]

		d_in_err[d_in_err == 0.]       = np.nan
		d_shift_err[d_shift_err == 0.] = np.nan

		d_sf_err = np.array([[
			np.sqrt(np.nansum((4.* (d_in[:nx_out - k, :ny_out - l] - d_shift[k:nx_out, l:ny_out])**2.\
				* (d_in_err[:nx_out - k, :ny_out - l]**2. + d_shift_err[k:nx_out, l:ny_out]**2.))))\
			/np.count_nonzero(~np.isnan(d_in[:nx_out - k, :ny_out - l] - d_shift[k:nx_out, l:ny_out]))
			for l in range(ny_out)] for k in range(nx_out)])
		return d_sf, d_sf_err


def pspec_1d(data, realfreq=False):
	'''
	Calculate Power-spectrum using FFT.
	'''

	nx = len(data)
	d_in = np.r_[data - np.nanmean(data), np.zeros(nx-1)] # zero-padding

	d_ft = fft(d_in)                     # Fourier transform
	d_ft_cnj = np.conjugate(fft(d_in))   # complex conjugate

	d_ps = (d_ft*d_ft_cnj).real # Power spectrum

	if realfreq:
		d_ps = d_ps[:len(d_ps)//2+1]
	else:
		d_ps = fftshift(d_ps)

	return d_ps


def binning(bin_e, coordinates, data):
	'''
	Binning data according to given bins and a set of coordinates and data.

	'''
	#bin_c = 0.5 .*(bin_e[2:length(bin_e)] .+ bin_e[1:length(bin_e)-1])
	d_bin = np.zeros(len(bin_e)-1)
	for i in range(len(bin_e)-1):
		indx = np.where( (coordinates >= bin_e[i]) & (coordinates < bin_e[i+1]))
		if len(indx[0]) == 0:
			d_bin[i] = np.nan
		else:
			d_bin[i] = np.nanmean(data[indx])

	return d_bin


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