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
	param = [0.29836702, 1.57390289, -0.67626107, 5.86712886]


	dist    = 140.
	ngrid   = 64
	lam_max = 2e4 # au
	pk = 2.5
	c  = 0.002/(lam_max)**pk
	cs = c*np.array([1e-1, 0.5])#, 1])

	rfit = 1000. # au
	pini = [-3, 1]
	# ----------------------


	# envelope model
	model = loggauss(x, *param)
	vf    = StatsVfield(model, [x])
	vf.calc_sf()


	# plot
	fig = plt.figure()
	ax  = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)


	# turbulent model
	for i in range(len(cs)):
		c_i = cs[i]
		vf_turbmodel = TurbModel(ngrid, vf.ndim, lam_max)
		vf_turbmodel.get_velocity(pk, c_i, fix_amp=True, fix_phase=False)
		#print(np.std(vf_turbmodel.vx))

		model     = loggauss(vf_turbmodel.x/dist, *param)
		model_sum = model + (vf_turbmodel.vx - vf_turbmodel.vx[ngrid//2])
		vf_mod = StatsVfield(model_sum, [vf_turbmodel.x])
		vf_mod.calc_ac(realfreq=True)
		vf_mod.calc_ps(realfreq=True)
		#print(vf_mod.tau_x)#, len(vf_mod.acf))

		#vf_mod.sf_plawfit(pini, taurange=[1e-6, rfit])
		#print(vf_mod.fit_results['pout'])


		#ax.plot(vf.x*dist, vf.data)
		#ax.plot(vf_turbmodel.x, vf_turbmodel.vx + param[-1])
		ax.plot(vf_turbmodel.x, model_sum)


		#ax2.plot(vf.tau_x, vf.sf)
		ax2.plot(vf_mod.tau_x, vf_mod.acf)
		ax3.plot(1/vf_mod.freq_x, vf_mod.ps, marker='o', alpha=0.7)

	ax3.set_ylim(1e-4,100)

	model_tau = np.linspace(10,2e4,128)
	#ax3.plot(model_tau, 1e-1*(model_tau/1e3)**1, color='r')

	ax3.set_xscale('log')
	ax3.set_yscale('log')

	pyfg.change_aspect_ratio(ax, 1)#, plottype='loglog')
	pyfg.change_aspect_ratio(ax2, 1)#, plottype='loglog')
	pyfg.change_aspect_ratio(ax3, 1, plottype='loglog')

	fig.subplots_adjust(wspace=0.4)
	plt.show()


if __name__ == '__main__':
	main()