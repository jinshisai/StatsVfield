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



def plawfit(x, y, pini, sig=None, xlim=[], cutzero=True, mode='lin', printres=True):
	'''
	'''

	from scipy.optimize import leastsq

	# fit function
	# power law
	plaw    = lambda x, param: param[0]*(x**(param[1]))
	errfunc = lambda param, x, y, sig: (plaw(x, param) - y)/sig
	#res = leastsq(errfunc, [1e-3, -3], args=(freq_fft[1:], np.abs(res_spec[1:])**2.))

	# linear
	fln      = lambda x, param: param[0] + param[1]*x
	errfunc2 = lambda param, x, y, sig: (fln(x, param) - y)/sig

	# fitting range
	if len(xlim) == 2:
		where_fit = (x > xlim[0]) & (x <= xlim[-1])
		y_fit     = y[where_fit]
		x_fit     = x[where_fit]

		if type(sig).__name__ == 'ndarray':
			sig_fit = sig[where_fit]
		else:
			sig_fit = sig
	else:
		y_fit = y
		x_fit = x
		sig_fit = sig

	if mode == 'lin':
		if type(sig).__name__ == 'NoneType':
			sig_fit = 1
		res = leastsq(errfunc, pini, args=(x_fit, y_fit, sig_fit), full_output=True)
		pout = res[0]
		pcov = res[1]
		chi2 = np.sum(errfunc(pout, x_fit, y_fit, sig_fit)**2.)
	elif mode == 'log':
		if type(sig).__name__ == 'NoneType':
			sig_fit = 1
			res = leastsq(errfunc2, pini,
				args=(np.log10(x_fit), np.log10(y_fit), sig_fit),
				full_output=True)
		else:
			res = leastsq(errfunc2, pini,
				args=(np.log10(x_fit), np.log10(y_fit), sig_fit/(y_fit*np.log(10))),
				full_output=True)
		pout = res[0]
		pcov = res[1]
		chi2 = np.sum(errfunc2(pout, np.log10(x_fit), np.log10(y_fit), sig_fit/(y_fit*np.log(10)))**2.)
	else:
		print('ERROR\tplawfit: mode must be lin or log.')
		return

	ndata  = len(x_fit)
	nparam = len(pout)
	dof    = ndata - nparam - 1
	reduced_chi2 = chi2/dof

	# parameter errors
	if (dof >= 0) and (pcov is not None):
		pcov = pcov*reduced_chi2
	else:
		pcov = np.full((nparam, nparam),np.inf)

	perr = np.array([
		np.abs(pcov[j][j])**0.5 for j in range(nparam)
		])

	if printres:
		print('Power-law fit')
		print('pini: (c, p) = (%.4e, %.4e)'%(pini[0], pini[1]))
		print('pout: (c, p) = (%.4e, %.4e)'%(pout[0], pout[1]))
		print('perr: (sig_c, sig_p) = (%.4e, %.4e)'%(perr[0], perr[1]))
		print('reduced chi^2: %.4f'%reduced_chi2)

	return pout, perr