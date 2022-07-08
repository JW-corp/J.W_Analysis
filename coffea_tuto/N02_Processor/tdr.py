import matplotlib.pyplot as plt
import mplhep as hep


lumi_factor = 59.97

plt.style.use(hep.style.CMS)
#plt.rcParams.update({
#	'font.size': 18,
#	'axes.titlesize': 24,
#	'axes.labelsize': 24,
#	'xtick.labelsize': 16,
#	'ytick.labelsize': 16
#})


fill_opts = {
	'edgecolor': (0,0,0,0.3),
	'alpha': 0.8
}
error_opts = {
	'label': 'stat. unc.',
	'hatch': '///',
	'facecolor': 'none',
	'edgecolor': (0,0,0,.5),
	'linewidth': 0
}




