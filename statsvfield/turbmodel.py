# modules
import numpy as np
from scipy.fft import fft, ifft, fftn, ifftn, fftfreq
from scipy.fft import rfft, irfft, rfftn, irfftn, rfftfreq
from scipy.fft import fftshift, ifftshift




class TurbModel():

	def __init__(self, ngrid: int = 32, ndim: int = 3, lmax: float = 0.2,
		lam_max=None, lam_min=None):
		self.ngrid = ngrid
		self.ndim  = ndim
		self.lmax  = lmax

		if lam_max:
			self.lam_max = lam_max
		else:
			self.lam_max = lmax

		if lam_min:
			self.lam_min = lam_min
		else:
			self.lam_min = 2.*lmax/ngrid # Nyquist theorem

		if ndim == 1:
			self.makegrid_1d()
		elif ndim == 3:
			self.makegrid_3d()

	def makegrid_1d(self):
		# grid in x-space
		nx = self.ngrid
		dx = self.lmax/nx
		x  = np.arange(-nx//2, nx//2 +1, 1)*dx # pixel
		if nx % 2 == 0:
			x_e = np.arange(-nx//2,nx//2+1,1)
			x   = (x_e[1:] + x_e[:-1])*0.5*dx
		else:
			x = np.arange(-nx//2 + 1, nx//2 +1, 1)*dx

		self.x  = x
		self.dx = dx
		self.nx = nx

		# grid in k-space
		kmax = 1/self.lam_min
		kmin = 1/self.lam_max
		self.kmax, self.kmin = kmax, kmin

		kx  = rfftfreq(self.nx, self.dx)
		dkx = kx[1] - kx[0]
		nkx = len(kx)

		self.kx  = kx
		self.dkx = dkx
		self.nkx = nkx

		# print input summary to check iFFT reserve condition you want
		print("lambda_min: %.4e"%self.lam_min)
		print("lambda_max: %.4e"%self.lam_max)


	def makegrid_3d(self):
		print('Still being developed. Sorry..')
		return


	def get_velocity(self, pk=4., c=1., fix_amp=False, fix_phase=False):
		'''
		Calculate trubulent velocity field.

		'''
		if self.ndim == 1:
			self.getvfield_1d(pk=pk, c=c, fix_amp=fix_amp, fix_phase=fix_phase)
		elif self.ndim == 3:
			self.getvfield_3d(pk=pk, c=c, fix_amp=fix_amp, fix_phase=fix_phase)


	def getvfield_1d(self, pk=2., c=1., fix_amp=False, fix_phase=False):
		nx, nkx  = self.nx, self.nkx
		kx   = self.kx
		dkx, dx  = self.dkx, self.dx
		kmin    = self.kmin
		lam_max = self.lam_max


		# Power spectrum
		sig2_vk = c*(kx*kx + kmin*kmin)**(-pk*0.5)  # <|sigma_vk|^2> at kx
		# In 1D, <|vk^2|> ~ E(k), which is the energy spectrum
		#  in an unit of energy per wavenumber,
		# meaning that sig2_vk has the unit [V^2 L]=[L^2 T^-2 L]


		# Random process if the fixing option is not selected
		vk_amp = sig2_vk**0.5 if fix_amp else np.random.normal(0., sig2_vk**0.5)  # Amplitude of Ak vector at (kx, ky, kz)
		vk_phs = 0. if fix_phase else np.random.uniform(0., 2.*np.pi, nkx) # Phase of Ak vector

		# vk
		vk = vk_amp*np.exp(vk_phs*1j) # vk complex vector
		# vk = <|vk^2|>^{1/2} ~ [km s^-1 L^{1/2}] when one dimension
		# Need to convert into [km s^-1] to get velocity after iFFT

		self.sig2_vk = sig2_vk
		self.vk      = vk

		# iFFT
		v = irfft(vk, nx)*nx #/np.sqrt(lam_max)
		# Multiplied:
		#  - nx because DFT results in 1/N. nx is fft length
		#  - 1/sqrt(lam_max) because F[vk] is in a unit of [km s^-1 L^{-1/2}].
		#    The unit becomes [km s^-1 L^{-1}] and L^-1 is corrected by fft length.
		#v = fftshift(v)
		self.vx = v.real


	def getvfield_3d(self, pk=2., c=1., fix_amp=False, fix_phase=False):
		print ('Still being developed. Sorry.')


	def smoothing(self, size, kernel='gauss'):
		from .modelfuncs import gaussian
		from astropy.convolution import convolve_fft

		beam = gaussian(self.x, 1, 0, size)
		beam /= np.sum(beam) # normalize
		v_smoothed = convolve_fft(self.vx, beam, nan_treatment='fill')
		self.vx_sm = v_smoothed