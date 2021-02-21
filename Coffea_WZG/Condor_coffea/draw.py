import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time
import sys


## I/O
file_list = sys.argv[1:]



## Parameter set
lumi = 21.01 * 10000
GenDY = 1933600
xsecDY=2137.0


## Null Histograms
hsum_nPV = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("nPV","Number of Primary vertex",100,0,100)	
)

hsum_cutflow = hist.Hist(
	'Events',
	hist.Cat('dataset', 'Dataset'),
	hist.Bin('cutflow', 'Cut index', [0, 1, 2, 3,4,5])
)

histdict = {"nPV": hsum_nPV, "cutflow": hsum_cutflow}



## Allreduce helper function
def reduce(histname):

	hists={}
	sumw_DY=0
	sumw_Egamma=0
	for filename in file_list:
		hin = load(filename)
		hists[filename] = hin.copy()
		sumw_DY += hists[filename]['sumw']['DY']
		sumw_Egamma += hists[filename]['sumw']['Egamma_RunAB']

		hist_ = histdict[histname]
		hist_.add(hists[filename][histname])

	return hist_, sumw_DY, sumw_Egamma



## Read hist and Scale

#histname = "nPV"
histname = "cutflow"


h1, sumw_DY, sumw_Egamma = reduce(histname)

#scales={
#	'DY' : lumi*xsecDY / sumw_DY
#}

#h1.scale(scales,axis='dataset')
print("# of mc: ",sumw_DY)
print("# of Egamma: ",sumw_Egamma)


print(h1.values())


## Draw hist
import mplhep
plt.style.use(mplhep.style.CMS)
plt.yscale('log')
hist.plot1d(h1,overlay='dataset')
plt.show()


#plt.savefig("weight.png")







