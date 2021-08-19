import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq



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



def plawfit(x, y, pini, xlim=[], cutzero=True):
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

	# fitting range
	if len(xlim) == 2:
		where_fit = (x > xlim[0]) & (x <= xlim[-1])
		y_fit     = y[where_fit]
		x_fit     = x[where_fit]
	else:
		y_fit = y
		x_fit = x

	#res = leastsq(errfunc2, [-3, -3], args=(np.log10(tau_sf[where_fit]), np.log10(sf_slice[where_fit])))
	#p_out = res[0]
	res = leastsq(errfunc2, pini, args=(np.log10(x_fit), np.log10(y_fit)))
	pout = res[0]

	return pout