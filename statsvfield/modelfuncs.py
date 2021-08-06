import numpy as np



# functions

# gaussian
def gaussian(x, height, center, width):
	return height*np.exp(-(x - center)**2/(2*width**2)) #+ offset

def double_gaussian(x, h1, c1, w1, h2, c2, w2, base):
	return (gaussian(x, h1, c1, w1) +
		gaussian(x, h2, c2, w2) + base)

def chi_gauss(param, xdata, ydata, ysig):
	chi = (ydata - (gaussian(xdata, *param))) / ysig
	return chi

def chi_dgauss(param, xdata, ydata, ysig):
	chi = (ydata - (double_gaussian(xdata, *param))) / ysig
	return chi

def lorentzian(x, a, x0, gamma):
	return a*gamma**2 /((x - x0)**2 + gamma**2) #+ offset

def double_lorentzian(x, a1, c1, g1, a2, c2, g2, base):
	return (lorentzian(x, a1, c1, g1) +
		gaussian(x, a2, c2, g2) + base)

def chi_dlorentz(param, xdata, ydata, ysig):
	chi = (ydata - (double_lorentzian(xdata, *param))) / ysig
	return chi

# discontinuous log-normal gaussian
def loggauss(x, height, center, width, base):
	logg = np.zeros(x.shape)

	# x > 0
	logg[x > 0.] = height*np.exp(-(np.log(x[x > 0.]) - center)**2/(2*width**2))/x[x > 0.] #+ offset

	# x = 0
	logg[x == 0.] = 0.

	# x <0
	logg[x < 0.] = height*np.exp(-(np.log(-x[x < 0.]) - center)**2/(2*width**2))/x[x < 0.] #+ offset

	return logg + base

def chi_loggauss(param, xdata, ydata, ysig):
	chi = (ydata - loggauss(xdata, *param)) / ysig
	return chi