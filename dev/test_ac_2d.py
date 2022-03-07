import numpy as np
import matplotlib.pyplot as plt


from statsvfield import StatsVfield
import pyfigures as pyfg



def main():
	# ------- input --------
	x     = np.linspace(-np.pi,np.pi,32) # arcsec
	y     = np.linspace(-np.pi,np.pi,32) # arcsec
	omega = 5
	rfit  = 10
	# ----------------------


	xx, yy = np.meshgrid(x, y)
	z      = np.sin(omega*xx) + np.sin(0.5*omega*yy)
	vf = StatsVfield(z, [x, y])
	#vf.calc_sf()
	vf.calc_ac(method='FFT')
	acf_fft = vf.acf
	vf.calc_ac(method='iterative')
	acf_itr = vf.acf

	# center
	print(np.unravel_index(np.abs(acf_fft - 1).argmin(), np.abs(acf_fft - 1).shape))
	print(np.unravel_index(np.abs(acf_itr - 1).argmin(), np.abs(acf_itr - 1).shape))
	print(np.nanargmin(np.abs(acf_fft - 1)))
	print(np.nanargmin(np.abs(acf_itr - 1)))

	# grid
	tau_xx, tau_yy = np.meshgrid(vf.tau_x, vf.tau_y)
	tau_rr = np.sqrt(tau_xx*tau_xx + tau_yy*tau_yy)

	fig = plt.figure(figsize=(11.69, 8.27))
	ax  = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)


	ax.imshow(acf_fft, origin='lower', vmin=-1, vmax=1)
	ax2.imshow(acf_itr, origin='lower', vmin=-1, vmax=1)
	ax3.imshow(acf_fft[1:,1:] - acf_itr[:-1,:-1], origin='lower', vmin=-1, vmax=1) # re-center

	ax4.scatter(acf_itr[:-1,:-1], acf_fft[1:,1:], color='k', alpha=0.7)
	pyfg.change_aspect_ratio(ax4, 1)
	#ax4.scatter(tau_rr, acf_itr, color='r', alpha=0.7)

	plt.show()


if __name__ == '__main__':
	main()