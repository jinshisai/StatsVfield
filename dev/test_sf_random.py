import numpy as np
import matplotlib.pyplot as plt


from statsvfield.modelfuncs import loggauss
from statsvfield import StatsVfield
from statsvfield.turbmodel import TurbModel
import pyfigures as pyfg



def main():
	# ------- input --------
	x     = np.linspace(-14,14,32) # arcsec
	rfit  = 10
	# ----------------------


	y = np.random.uniform(-0.2, 0.2, len(x))
	vf = StatsVfield(y, [x])
	vf.calc_sf()

	vf.sf_plawfit([1,1], taurange=[1e-6, rfit])
	print(vf.fit_results['pout'])

	plt.plot(vf.tau_x, vf.sf)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()


if __name__ == '__main__':
	main()