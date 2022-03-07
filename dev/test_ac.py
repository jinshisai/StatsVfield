import numpy as np
import matplotlib.pyplot as plt


from statsvfield import StatsVfield
import pyfigures as pyfg



def main():
	# ------- input --------
	x     = np.linspace(-np.pi,np.pi,128) # arcsec
	omega = 5
	rfit  = 10
	# ----------------------


	y = np.sin(omega*x)
	vf = StatsVfield(y, [x])
	vf.calc_sf()
	vf.calc_ac(realfreq=True)

	vf.sf_plawfit([1,1], taurange=[1e-6, rfit])
	print(vf.fit_results['pout'])


	fig = plt.figure()
	ax  = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	ax.plot(x, y)
	#ax2.plot(vf.tau_x, vf.acf)
	ax2.plot(vf.tau_x, vf.sf)

	ax2.set_xscale('log')
	ax2.set_yscale('log')
	plt.show()


if __name__ == '__main__':
	main()