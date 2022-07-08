import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time
import sys
import tdr



hsum_Mee = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("mass","Z mass",100,0,200)	
)
histdict = {'mass':hsum_Mee}



def reduce(folder,sample_list,histname):
	hists={}
	
	for filename in os.listdir(folder):
		
		hin = load(folder + '/' + filename)
		hists[filename] = hin.copy()
		
		if filename.split('.')[0] not in sample_list:
			continue

		hist_ = histdict[histname]
		hist_.add(hists[filename][histname])
	return hist_


	



if __name__ == "__main__":



	# >> I/O
	file_path = 'output'
	sample_list = ['DY']
	histname = 'mass'
	h1  = reduce(file_path,sample_list,histname)
	# <<
	
	
	# >> Noramlize
	lumi_factor = 59.97
	scales={
	   'DY'	  : lumi_factor * 1000  * 2137.0 / 1933600,
	}
	h1.scale(scales,axis='dataset')
	#h1 = h1.rebin(histname,hist.Bin("mass","Z mass",100,50,150))
	# <<
	
	
	# >> Basic plot set
	fig, ax = plt.subplots(
		nrows=1,
		ncols=1,
		figsize=(10,10),
		sharex=True
	)
		
	from cycler import cycler
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	ax.set_prop_cycle(cycler(color=colors))
	fill_opts = tdr.fill_opts
	error_opts = tdr.error_opts



	# >> Draw hist
	XMIN=0
	XMAX=200
	YMIN=0
	YMAX=30000000
	XMIN=0

	hist.plot1d(
		h1['DY'],
		ax=ax,
		clear=False,
		#stack=True,
		fill_opts=fill_opts,
		error_opts = error_opts,
	)
	
	ax._get_lines.prop_cycler = ax._get_patches_for_fill.prop_cycler
	ax.autoscale(axis="x", tight=True)
	ax.set_ylim(YMIN, YMAX)
	ax.set_xlim(XMIN, XMAX)
	# <<

	
	# >> Just for Lumi text
	import mplhep as hep
	hep.cms.text("Simulation")
	hep.cms.lumitext("{} fb$^{{-1}}$".format(lumi_factor))
	# <<	


	# output 
	outname = "output" + "/"  + histname +  ".png"
	plt.savefig(outname)
