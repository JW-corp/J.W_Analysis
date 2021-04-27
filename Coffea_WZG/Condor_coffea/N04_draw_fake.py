import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time


isData=True
isFake=True
isReal=True


## Parameter set
lumi= 53.03 * 1000


GenDict={
'WZG':128000,
"DY":1933600,
"WZ":7986000,
"ZZ":2000000,
"TTWJets":4963867,
"TTZtoLL":13914900,
"tZq":12748300,
"ZGToLLG" :28636926,
"TTGJets" :4647426,
"WGToLNuG":20371504
}


xsecDict={
"WZG":0.0196,
"DY":2137.0,
"WZ":27.6,
"ZZ":12.14,
"TTWJets":0.2149,
"TTZtoLL":0.2432,
"tZq":0.07358,
"ZGToLLG" : 55.48,
"TTGJets" : 4.078,
"WGToLNuG": 1.249
}




hsum_PT_1_eta_1 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_1_eta_1","20 < pt <30 & |eta| < 1", 200, 0, 0.02),
),
hsum_PT_1_eta_2 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_1_eta_2","20 < pt <30 & 1 < |eta| < 1.5", 200, 0, 0.02),
),
hsum_PT_1_eta_3 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_1_eta_3","20 < pt <30 & 1.5 < |eta| < 2", 200, 0, 0.05),
),
hsum_PT_1_eta_4 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_1_eta_4","20 < pt <30 & 2 < |eta| < 2.5", 200, 0, 0.05),
),
hsum_PT_2_eta_1 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_2_eta_1","30 < pt <40 & |eta| < 1", 200, 0, 0.02),
),
hsum_PT_2_eta_2 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_2_eta_2","30 < pt <40 & 1 < |eta| < 1.5", 200, 0, 0.02),
),
hsum_PT_2_eta_3= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_2_eta_3","30 < pt <40 & 1.5 < |eta| < 2", 200, 0, 0.05),
),
hsum_PT_2_eta_4 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_2_eta_4","30 < pt <40 & 2 < |eta| < 2.5", 200, 0, 0.05),
),
hsum_PT_3_eta_1 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_3_eta_1","40 < pt <50 & |eta| < 1", 200, 0, 0.02),
),
hsum_PT_3_eta_2 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_3_eta_2","40 < pt <50 & 1 < |eta| < 1.5", 200, 0, 0.02),
),
hsum_PT_3_eta_3= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_3_eta_3","40 < pt <50 & 1.5 < |eta| < 2", 200, 0, 0.05),
),
hsum_PT_3_eta_4= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_3_eta_4","40 < pt <50 & 2 < |eta| < 2.5", 200, 0, 0.05),
),
hsum_PT_4_eta_1= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_4_eta_1","50 < pt & |eta| < 1", 200, 0, 0.02),
),
hsum_PT_4_eta_2= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_4_eta_2","50 <pt  & 1 < |eta| < 1.5", 200, 0, 0.02),
),
hsum_PT_4_eta_3 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_4_eta_3","50 < pt  & 1.5 < |eta| < 2", 200, 0, 0.05),
),
hsum_PT_4_eta_4 =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("PT_4_eta_4","50 < pt  & 2 < |eta| < 2.5", 200, 0, 0.05),
)


histdict = {
'PT_1_eta_1' :hsum_PT_1_eta_1 ,'PT_1_eta_2' :hsum_PT_1_eta_2 ,'PT_1_eta_3' :hsum_PT_1_eta_3 ,'PT_1_eta_4' :hsum_PT_1_eta_4 ,'PT_2_eta_1' :hsum_PT_2_eta_1 ,'PT_2_eta_2' :hsum_PT_2_eta_2 ,'PT_2_eta_3' :hsum_PT_2_eta_3 ,'PT_2_eta_4' :hsum_PT_2_eta_4 ,'PT_3_eta_1' :hsum_PT_3_eta_1 ,'PT_3_eta_2' :hsum_PT_3_eta_2 ,'PT_3_eta_3' :hsum_PT_3_eta_3 ,'PT_3_eta_4' :hsum_PT_3_eta_4 ,'PT_4_eta_1' :hsum_PT_4_eta_1 ,'PT_4_eta_2' :hsum_PT_4_eta_2 ,'PT_4_eta_3' :hsum_PT_4_eta_3 ,'PT_4_eta_4' :hsum_PT_4_eta_4
}



def reduce(folder,sample_list,histname):
	hists={}
	
	for filename in os.listdir(folder):
		hin = load(folder + '/' + filename)
		hists[filename] = hin.copy()
		if filename.split('_')[0] not in sample_list:
			continue
		hist_ = histdict[histname] # -- for Last bin in Fake object
		#hist_ = histdict[histname][0] # -- for Fake object
		hist_.add(hists[filename][histname])
		
	return hist_ 


## --File Directories
#file_path = "210427_FakeTemplate"
file_path = "210427_FakeTemplate_v2"


## --Sample Lists
sample_list = ['Fake','Real','data']




##############################################################33 --Hist names


## Auto mode
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bin', type=str)
args = parser.parse_args()

key = args.bin

histname = key
bin_dict = {
"PT_1_eta_1": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":1000}
,"PT_1_eta_2": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":500}
,"PT_1_eta_3": {"xmin":0, "xmax":0.05, "ymin":0., "ymax":300}
,"PT_1_eta_4": {"xmin":0, "xmax":0.05, "ymin":0., "ymax":200}
,"PT_2_eta_1": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":500}
,"PT_2_eta_2": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":200}
,"PT_2_eta_3": {"xmin":0, "xmax":0.05, "ymin":0., "ymax":100 }
,"PT_2_eta_4": {"xmin":0, "xmax":0.05, "ymin":0., "ymax":100 }
,"PT_3_eta_1": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":500}
,"PT_3_eta_2": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":100}
,"PT_3_eta_3": {"xmin":0, "xmax":0.05, "ymin":0., "ymax":100 }
,"PT_3_eta_4": {"xmin":0, "xmax":0.05, "ymin":0., "ymax":100 }
,"PT_4_eta_1": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":500}
,"PT_4_eta_2": {"xmin":0, "xmax":0.02, "ymin":0., "ymax":125}
,"PT_4_eta_3": {"xmin":0, "xmax":0.05, "ymin":0., "ymax":100 }
,"PT_4_eta_4": {"xmin":0, "xmax":0.1,  "ymin":0., "ymax":100}
}
xmin = bin_dict[key]['xmin']
xmax = bin_dict[key]['xmax']
ymin = bin_dict[key]['ymin']
ymax = bin_dict[key]['ymax']
'''
## Hand mode

#histname = "PT_1_eta_1";xmin=0; xmax=0.02; ymin=0.; ymax=4000; 
#histname = "PT_1_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=1000;
#histname = "PT_1_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=500;
#histname = "PT_1_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=300;
#histname = "PT_2_eta_1";xmin=0; xmax=0.02; ymin=0.; ymax=1500;
#histname = "PT_2_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=500;
#histname = "PT_2_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=200;
#histname = "PT_2_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=150;
#histname = "PT_3_eta_1";xmin=0; xmax=0.02; ymin=0.; ymax=500;
#histname = "PT_3_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=200;
#histname = "PT_3_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=100;
#histname = "PT_3_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=60;
#histname = "PT_4_eta_1";xmin=0; xmax=0.02; ymin=0.; ymax=1000;
#histname = "PT_4_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=300;
#histname = "PT_4_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=200;
histname = "PT_4_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=100;


################################################################
## --All-reduce 
h1 = reduce(file_path,sample_list,histname)





## --Rebin
#h1 = h1.rebin(histname,hist.Bin("PT_1_eta_1","20 < pt <30 & |eta| < 1", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_1_eta_2","20 < pt <30 & 1 < |eta| < 1.5", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_1_eta_3","20 < pt <30 & 1.5 < |eta| < 2", 50, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_1_eta_4","20 < pt <30 & 2 < |eta| < 2.5", 50, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_1","30 < pt <40 & |eta| < 1", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_2","30 < pt <40 & 1 < |eta| < 1.5", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_3","30 < pt <40 & 1.5 < |eta| < 2", 50, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_4","30 < pt <40 & 2 < |eta| < 2.5", 50, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_1","40 < pt <50 & |eta| < 1", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_2","40 < pt <50 & 1 < |eta| < 1.5", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_3","40 < pt <50 & 1.5 < |eta| < 2", 50, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_4","40 < pt <50 & 2 < |eta| < 2.5", 50, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_4_eta_1","50 < pt & |eta| < 1", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_4_eta_2","50 <pt  & 1 < |eta| < 1.5", 50, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_4_eta_3","50 < pt  & 1.5 < |eta| < 2", 50, 0, 0.05))
h1 = h1.rebin(histname,hist.Bin("PT_4_eta_4","50 < pt  & 2 < |eta| < 2.5", 50, 0, 0.05))






# ----> Plotting 
print("End processing.. make plot")
print(" ")


print("##" * 20)
for i,j in h1.values().items():
	print(i,":",j[-2])



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
	nrows=1,
	ncols=1,
	figsize=(7,7),
	sharex=True
)


fake_error_opts = {
	'linestyle': 'none',
'marker': '.',
'markersize': 10.,
'color': 'royalblue',
'elinewidth': 1,
}

real_error_opts = {
	'linestyle': 'none',
'marker': '.',
'markersize': 10.,
'color': 'darkorange',
'elinewidth': 1,
}

data_error_opts = {
	'linestyle': 'none',
'marker': '.',
'markersize': 10.,
'color': 'black',
'elinewidth': 1,
}



# Fake template
if isFake:
	hist.plot1d(
	
		h1['Fake_template'],
		ax=ax,
		clear=False,
		error_opts=fake_error_opts,
		
	)

# Real template
if isReal:
	hist.plot1d(
	
		h1['Real_template'],
		ax=ax,
		clear=False,
		error_opts=real_error_opts,
		
	)

# Data template
if isData:
	hist.plot1d(
	
		h1['data_template'],
		ax=ax,
		clear=False,
		error_opts=data_error_opts,
		
	)

np.set_printoptions(suppress=True)

ax.autoscale(axis='x', tight=True)
ax.set_ylim(ymin,ymax)
ax.set_xlim(xmin,xmax)
#ax.set_xlabel('')
#ax.set_yscale('log')


lum = plt.text(1., 1., r"53.03 fb$^{-1}$ (13 TeV)",
				fontsize=16,
				horizontalalignment='right',
				verticalalignment='bottom',
				transform=ax.transAxes
			   )



outname = histname + "_" + file_path + ".png"

plt.savefig(outname)
plt.show()
