import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time


hsum_pho_EB_sieie =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_sieie","Photon EB sieie", 100, 0, 0.012),
)


histdict = {'pho_EB_sieie': hsum_pho_EB_sieie}



def reduce(folder,sample_list,histname):
	hists={}
	

	sumwdict ={
		"Egamma":0,
	}

	
	for filename in os.listdir(folder):
		hin = load(folder + '/' + filename)
		hists[filename] = hin.copy()
		if filename.split('_')[0] not in sample_list:
			continue
		sumwdict['Egamma'] += hists[filename]['sumw']['Egamma']


		hist_ = histdict[histname]
		hist_.add(hists[filename][histname])
		
	return hist_ , sumwdict


## --File Directories
file_path = "210318RunABD"



## --Sample Lists
sample_list = ['Egamma']




##############################################################33 --Hist names

		
#histname = "pho_EB_sieie"; xmin=0; xmax=0.012; ymin=0.01; ymax=5e+6;

################################################################
## --All-reduce 
h1,sumwdict = reduce(file_path,sample_list,histname)

for i,j in sumwdict.items():
	print(i,": ",j)




## --Rebin
#h1 = h1.rebin(histname,hist.Bin("pho_EE_pt","Photon EE $P_{T}$ [GeV]", 30, 0, 600))






# ----> Plotting 
print("End processing.. make plot")
print(" ")
# make a nice ratio plot, adjusting some font sizes


import mplhep as hep
plt.style.use(hep.style.CMS)



plt.rcParams.update({
	'font.size': 14,
	'axes.titlesize': 18,
	'axes.labelsize': 18,
	'xtick.labelsize': 12,
	'ytick.labelsize': 12
})
fig, ax = plt.subplots(
	nrows=2,
	ncols=1,
	figsize=(7,7),
	gridspec_kw={"height_ratios": (3, 1)},
	sharex=True
)

fig.subplots_adjust(hspace=.07)


from cycler import cycler

colors= ['r','g','b','orange','pink','navy','yellow']

ax.set_prop_cycle(cycler(color=colors))

data_err_opts = {
	'linestyle': 'none',
'marker': '.',
'markersize': 10.,
'color': 'k',
'elinewidth': 1,
}






# DATA plotting
hist.plot1d(

	h1['Egamma'],
	ax=ax,
	clear=False,
	error_opts=data_err_opts
	
)




ax._get_lines.prop_cycler = ax._get_patches_for_fill.prop_cycler
ax.autoscale(axis='x', tight=True)
ax.set_ylim(ymin,ymax)
ax.set_xlim(xmin,xmax)
ax.set_xlabel('')
ax.set_yscale('log')


#leg = ax.legend()


lum = plt.text(1., 1., r"53.03 fb$^{-1}$ (13 TeV)",
				fontsize=16,
				horizontalalignment='right',
				verticalalignment='bottom',
				transform=ax.transAxes
			   )



outname = histname + "_" + file_path + ".png"

plt.savefig(outname)
plt.show()
