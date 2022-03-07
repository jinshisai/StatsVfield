import numpy as np
import matplotlib.pyplot as plt


from statsvfield.modelfuncs import loggauss
from statsvfield import StatsVfield
from statsvfield.turbmodel import TurbModel
import pyfigures as pyfg


def main():
	# ------- input --------
	x     = np.linspace(-14,14,32) # arcsec
	param = [0.73777159, 0.27011437, 0.91976761, 5.82527105]


	dist    = 140.
	ngrid   = 128
	lam_max = 2e4 # au
	pk = 2
	c  = 0.02/(lam_max)**pk
	# ----------------------



	# model
	model = loggauss(x, *param)
	vf    = StatsVfield(model, [x])
	vf.calc_sf()

	vf_turbmodel = TurbModel(ngrid, vf.ndim, lam_max)
	vf_turbmodel.get_velocity(pk, c, fix_amp=True, fix_phase=False)
	#print(np.std(vf_turbmodel.vx))

	model     = loggauss(vf_turbmodel.x/dist, *param)
	model_sum = model + (vf_turbmodel.vx - vf_turbmodel.vx[ngrid//2])
	vf_mod = StatsVfield(model_sum, [vf_turbmodel.x])
	vf_mod.calc_sf()

	# plot
	fig = plt.figure()
	ax  = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	ax.plot(vf.x*dist, vf.data)
	ax.plot(vf_turbmodel.x, vf_turbmodel.vx + param[-1])
	ax.plot(vf_turbmodel.x, model_sum)


	ax2.plot(vf.tau_x, vf.sf)
	ax2.plot(vf_mod.tau_x, vf_mod.sf)
	ax2.set_ylim(1e-2,1)

	ax2.set_xscale('log')
	ax2.set_yscale('log')

	pyfg.change_aspect_ratio(ax2, 1, plottype='loglog')


	plt.show()


if __name__ == '__main__':
	main()