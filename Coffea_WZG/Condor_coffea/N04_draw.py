import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time


import sys

sys.path.append("util")
import Particle_Info_DB
import Hist_Base_dict
import Hist_CR_ZZA_dict
import Hist_SR_dict



# -- Muon -- #
hsum_mupt = hist.Hist(
    "Events",
    hist.Cat("dataset", "Dataset"),
	hist.Cat("region", "region"),
    hist.Bin("mupt","Leading Muon  $P_{T}$ [GeV]", 300, 0, 600),
)


hsum_mueta =  hist.Hist(
    "Events",
    hist.Cat("dataset", "Dataset"),
	hist.Cat("region", "region"),
    hist.Bin("mueta", "Leading Muon $\eta$ [GeV]", 20, -5, 5),
)


hsum_muphi =  hist.Hist(
    "Events",
    hist.Cat("dataset", "Dataset"),
	hist.Cat("region", "region"),
    hist.Bin("muphi", "Leading Muon $\phi$ [GeV]", 20, -3.15, 3.15),
)



hsum_mass_eee =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("mass_eee","$m_{eee}$ [GeV]", 300, 0, 600),
)


hsum_MT = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("MT","W MT [GeV]", 100, 0, 200)
)


hsum_dR_aj = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("dR_aj","$\delta R_{\gammaj}$", 100, 0, 4),
)
hsum_cutflow = hist.Hist(
	'Events',
	hist.Cat('dataset', 'Dataset'),
	hist.Cat("region", "region"),
	hist.Bin('cutflow', 'Cut index', [0, 1, 2, 3,4,5,6,7,8])
)

hsum_charge= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("charge","charge sum of electrons", 6, -3, 3),
)
hsum_nPV = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("nPV","Number of Primary vertex",100,0,100)	
)
hsum_nPV_nw = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("nPV_nw","Number of Primary vertex",100,0,100)	
)
hsum_Mee = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("mass","Z mass",100,0,200)	
)

hsum_ele1pt = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("ele1pt","Leading Electron $P_{T}$ [GeV]",300,0,600)	
)
hsum_ele2pt =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("ele2pt","Subleading $Electron P_{T}$ [GeV]", 300, 0, 600),
)
hsum_ele3pt =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("ele3pt","Third $Electron P_{T}$ [GeV]", 300, 0, 600),
)

hsum_ele1eta= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("ele1eta","Leading Electron $\eta$ [GeV]", 20, -5, 5),
)

hsum_ele2eta =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("ele2eta","Subleading Electron $\eta$ [GeV]", 20, -5, 5),
)
hsum_ele1phi =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("ele1phi","Leading Electron $\phi$ [GeV]", 20, -3.15, 3.15),
)

hsum_ele2phi =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("ele2phi","Subleading Electron $\phi$ [GeV]", 20, -3.15, 3.15),
)
hsum_nElectrons = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("nElectrons","# of Electrons",10,0,10)
)

hsum_pho_EE_pt =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EE_pt","Photon EE $P_{T}$ [GeV]", 300, 0, 600),
)

hsum_pho_EE_eta =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EE_eta","Photon EE $\eta$ ", 50, -5, 5),
)
hsum_pho_EE_phi =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EE_phi","Photon EE $\phi$ ", 50, -3.15, 3.15),
)
hsum_pho_EE_hoe = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EE_hoe","Photon EE HoverE", 100, 0, 0.6),
)
hsum_pho_EE_sieie =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EE_sieie","Photon EE sieie", 100, 0, 0.3),
)
hsum_pho_EE_Iso_all =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EE_Iso_all","Photon EE pfReoIso03_all", 100, 0, 0.3),
)
hsum_pho_EE_Iso_chg =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EE_Iso_chg","Photon EE pfReoIso03_charge", 100, 0, 0.03),
)





hsum_pho_EB_pt =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EB_pt","Photon EB $P_{T}$ [GeV]", 300, 0, 600),
)
hsum_pho_EB_eta =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EB_eta","Photon EB $\eta$ ", 50, -5, 5),
)
hsum_pho_EB_phi =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EB_phi","Photon EB $\phi$ ", 50, -3.15, 3.15),
)
hsum_pho_EB_hoe = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EB_hoe","Photon EB HoverE", 100, 0, 0.6),
)
hsum_pho_EB_sieie =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EB_sieie","Photon EB sieie", 100, 0, 0.012),
)
hsum_pho_EB_Iso_all =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EB_Iso_all","Photon EB pfReoIso03_all", 100, 0, 0.15),
)
hsum_pho_EB_Iso_chg =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("pho_EB_Iso_chg","Photon EB pfReoIso03_charge", 100, 0, 0.03),
)


hsum_met = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Cat("region", "region"),
	hist.Bin("met","met [GeV]", 300, 0, 600),
)


histdict = {'nPV':hsum_nPV,'nPV_nw':hsum_nPV_nw, "cutflow":hsum_cutflow,"ele1pt":hsum_ele1pt,"ele2pt":hsum_ele2pt,"ele3pt":hsum_ele3pt,"mass":hsum_Mee,\
"ele1eta":hsum_ele1eta, "ele2eta":hsum_ele2eta,"ele1phi":hsum_ele1phi, "ele2phi":hsum_ele2phi,\

"mupt":hsum_mupt, "mueta":hsum_mueta, "muphi":hsum_muphi,\

'pho_EE_pt':hsum_pho_EE_pt, 'pho_EE_eta':hsum_pho_EE_eta, 'pho_EE_phi':hsum_pho_EE_phi,\
'pho_EE_hoe':hsum_pho_EE_hoe,'pho_EE_sieie': hsum_pho_EE_sieie, 'pho_EE_Iso_all': hsum_pho_EE_Iso_all, 'pho_EE_Iso_chg':hsum_pho_EE_Iso_chg,\
'pho_EB_pt':hsum_pho_EB_pt, 'pho_EB_eta':hsum_pho_EB_eta, 'pho_EB_phi':hsum_pho_EB_phi,\
'pho_EB_hoe':hsum_pho_EB_hoe,'pho_EB_sieie': hsum_pho_EB_sieie, 'pho_EB_Iso_all': hsum_pho_EB_Iso_all, 'pho_EB_Iso_chg':hsum_pho_EB_Iso_chg,
'met':hsum_met,'dR_aj':hsum_dR_aj,'MT':hsum_MT,'mass_eee':hsum_mass_eee}




def reduce(folder,sample_list,histname):
	hists={}
	

	for filename in os.listdir(folder):
		hin = load(folder + '/' + filename)
		hists[filename] = hin.copy()
		
		if filename.split('_')[0] not in sample_list:
			continue

		hist_ = histdict[histname]
		hist_.add(hists[filename][histname])
	return hist_



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("hist_name", type=str, help="PT_1_eta_1")
parser.add_argument("year", type=str, help="2018")
parser.add_argument("channel", type=str, help="eee")
parser.add_argument("region", type=str, help="Signal")
parser.add_argument("filename", type=str, help="210531_eee_2018")
args = parser.parse_args()



if args.region == "Baseline":
	Hist_dict = Hist_Base_dict
elif args.region == "Signal":
	Hist_dict = Hist_SR_dict
else:
	Hist_dict = Hist_CR_ZZA_dict


histname = args.hist_name
hist_info = Hist_dict.hist_info



xmin = hist_info[histname]['xmin']
xmax = hist_info[histname]['xmax']
ymin = hist_info[histname]['ymin']
ymax = hist_info[histname]['ymax']

if not histname == 'cutflow':
	bins = hist_info[histname]['bins']
	binxmin = hist_info[histname]['binxmin']
	binxmax = hist_info[histname]['binxmax']
	name = hist_info[histname]['name']

# 2018
#year = "2018"
#file_name = "210531_eee_2018"
#file_name = "210531_eemu_2018"

#year = "2017"
#file_name = "210531_eee_2017"
#file_name = "210531_eemu_2017"


year = args.year
file_name = args.filename
channel = args.channel
file_path = "results/" + file_name


dict_ = Particle_Info_DB.DB
lumi_factor = dict_[year]["Lumi"]
GenDict = dict_[year]["Gen"]
xsecDict = dict_[year]["xsec"]

if (year =='2018') and (channel =='eee'):
	order = ['TTWJets', 'tZq', 'TTGJets', 'TTZtoLL', 'Fake_Photon', 'WZ', 'ZGToLLG', 'ZZ', 'WZG']

if (year =='2018') and (channel =='eemu'):
	order = ['tZq', 'TTWJets', 'ZGToLLG', 'TTGJets', 'TTZtoLL', 'ZZ', 'Fake_Photon', 'WZ', 'WZG']

if (year =='2017') and (channel =='eee'):
	order = ['TTWJets', 'tZq', 'TTGJets', 'TTZtoLL', 'Fake_Photon', 'WZ', 'ZGToLLG', 'ZZ', 'WZG']

if (year =='2017') and (channel =='eemu'):
	order = ['tZq', 'TTWJets', 'ZGToLLG', 'TTZtoLL', 'TTGJets', 'Fake_Photon', 'ZZ', 'WZ', 'WZG']

#order = ['TTWJets', 'tZq', 'TTGJets', 'TTZtoLL', 'Fake_Photon', 'WZ', 'ZGToLLG', 'ZZ', 'WZG']



## --Sample Lists
sample_list = ['DoubleEG','DY' ,'WZ' ,'ZZ' ,'TTWJets','TTZtoLL','tZq' ,'Egamma','WZG','ZGToLLG','TTGJets','WGToLNuG','FakePhoton','FakeLepton']



##############################################################33 --Hist names

		
		# --- MET --- #

#histname = "met"; xmin=0; xmax=200; ymin=0; ymax=20; 

		# --- Electron --- #

#histname = "ele1pt"; xmin=0; xmax=200; ymin=0; ymax=12;
#histname = "ele2pt"; xmin=0; xmax=200; ymin=0; ymax=20; 
#histname = "ele3pt"; xmin=0; xmax=200; ymin=0; ymax=17;

#histname = "ele1eta"; xmin=-3; xmax=3; ymin=0; ymax=12;
#histname = "ele2eta"; xmin=-3; xmax=3; ymin=0; ymax=12;

#histname = "ele1phi"; xmin=-3.15; xmax=3.15; ymin=0; ymax=10;
#histname = "ele2phi"; xmin=-3.15; xmax=3.15; ymin=0; ymax=10;

#histname = "cutflow"; xmin=0; xmax=7; ymin=1; ymax=5e+9

		# --- Muon --- #

#histname = "mupt"; xmin=0; xmax=200; ymin=0; ymax=10;
#histname = "mueta"; xmin=-2.5; xmax=2.5; ymin=100; ymax=10;
#histname = "muphi"; xmin=-3.15; xmax=3.15; ymin=100; ymax=10;


		# --- Photon --- #

#histname = "pho_EB_pt"; xmin=0; xmax=200; ymin=0; ymax=16;
#histname = "pho_EB_eta"; xmin=-3; xmax=3; ymin=0; ymax=10;
#histname = "pho_EB_phi"; xmin=-3.15; xmax=3.15; ymin=0; ymax=10;
#histname = "pho_EB_sieie"; xmin=0.006; xmax=0.012; ymin=0; ymax=16;

#histname = "pho_EE_pt"; xmin=0; xmax=200; ymin=0; ymax=10;
#histname = "pho_EE_eta"; xmin=-3; xmax=3; ymin=0; ymax=7;
#histname = "pho_EE_phi"; xmin=-3.15; xmax=3.15; ymin=0; ymax=6;
#histname = "pho_EE_sieie"; xmin=0.2; xmax=0.3; ymin=0; ymax=6;



		# --- Kinematics --- #
#histname = "mass"; xmin=0; xmax=200; ymin=0; ymax=20;
#histname = "MT"; xmin=0; xmax=200; ymin=0; ymax=10;

################################################################
## --All-reduce 
h1  = reduce(file_path,sample_list,histname)


## --Noramlize	
scales={
	'WZG'	  : lumi_factor * 1000  * xsecDict['WZG'] / GenDict['WZG'],
#	'DY'	  : lumi_factor * 1000  * xsecDict['DY'] / GenDict['DY'],
	'WZ'	  : lumi_factor * 1000  * xsecDict['WZ'] / GenDict['WZ'],
	'ZZ'	  : lumi_factor * 1000  * xsecDict['ZZ'] / GenDict['ZZ'],
	'TTWJets' : lumi_factor * 1000  * xsecDict['TTWJets'] / GenDict['TTWJets'],
	'TTZtoLL' : lumi_factor * 1000  * xsecDict['TTZtoLL'] / GenDict['TTZtoLL'],
	'tZq'	  : lumi_factor * 1000  * xsecDict['tZq'] / GenDict['tZq'],
	'ZGToLLG' : lumi_factor * 1000  * xsecDict['ZGToLLG'] / GenDict['ZGToLLG'],
    'TTGJets' : lumi_factor * 1000  * xsecDict['TTGJets'] / GenDict['TTGJets'],
 #   'WGToLNuG': lumi_factor * 1000  * xsecDict['WGToLNuG'] / GenDict['WGToLNuG'],
}


h1.scale(scales,axis='dataset')

## --Rebin

# MET
#h1 = h1.rebin(histname,hist.Bin("met","met [GeV]", 10, 0, 200))

# Kinematics
#h1 = h1.rebin(histname,hist.Bin("mass","Z mass",10,0,200))
#h1 = h1.rebin(histname,hist.Bin("MT","W MT [GeV]", 10, 0, 200))

# Electron

#h1 = h1.rebin(histname,hist.Bin("ele1pt","Leading electron from Z  $P_{T}$ [GeV]",30,0,600))
#h1 = h1.rebin(histname,hist.Bin("ele2pt","Subleading electron from Z  $P_{T}$ [GeV]",30,0,600))
#h1 = h1.rebin(histname,hist.Bin("ele3pt","electron from W  $P_{T}$ [GeV]",30,0,600))

#h1 = h1.rebin(histname,hist.Bin("ele1phi","Leading Electron $\phi$ [GeV]", 10, -3.15, 3.15))
#h1 = h1.rebin(histname,hist.Bin("ele2phi","Subleading Electron $\phi$ [GeV]", 10, -3.15, 3.15))


# Photon
#h1 = h1.rebin(histname,hist.Bin("pho_EB_pt","Photon EB $P_{T}$ [GeV]", 30, 0, 600))
#h1 = h1.rebin(histname,hist.Bin("pho_EB_eta","Photon EB $\eta$ ", 25, -5, 5))
#h1 = h1.rebin(histname,hist.Bin("pho_EB_phi","Photon EB $\phi$ ", 10, -3.15, 3.15))
#h1 = h1.rebin(histname,hist.Bin("pho_EB_sieie","Photon EB sieie", 25, 0, 0.012))


#h1 = h1.rebin(histname,hist.Bin("pho_EE_pt","Photon EE $P_{T}$ [GeV]", 30, 0, 600))
#h1 = h1.rebin(histname,hist.Bin("pho_EE_eta","Photon EE $\eta$ ", 25, -5, 5))
#h1 = h1.rebin(histname,hist.Bin("pho_EE_phi","Photon EE $\phi$ ", 10, -3.15, 3.15))
#h1 = h1.rebin(histname,hist.Bin("pho_EE_sieie","Photon EE sieie", 50, 0, 0.3))

# Muon
#h1 = h1.rebin(histname,hist.Bin("mupt","Leading Muon  $P_{T}$ [GeV]", 30, 0, 600))

if not histname == 'cutflow':
	h1 = h1.rebin(histname,hist.Bin(histname,name,bins,binxmin,binxmax))




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
fig, (ax, rax) = plt.subplots(
#fig, ax = plt.subplots(
	#nrows=1,
	nrows=2,
	ncols=1,
	figsize=(10,10),
	gridspec_kw={"height_ratios": (3, 1)},
	sharex=True
)

fig.subplots_adjust(hspace=.07)


from cycler import cycler
#colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

#colors= ['#e31a1c','navy','g','b','orange','pink','r','yellow']

ax.set_prop_cycle(cycler(color=colors))


fill_opts = {
	'edgecolor': (0,0,0,0.3),
	'alpha': 0.8
}
error_opts = {
	'label': 'Stat. Unc.',
	'hatch': '///',
	'facecolor': 'none',
	'edgecolor': (0,0,0,.5),
	'linewidth': 0
}
data_err_opts = {
	'linestyle': 'none',
'marker': '.',
'markersize': 10.,
'color': 'k',
'elinewidth': 1,
}


# -- Make order of stak
#print("##" * 20)
#order_dict = {}
#for i, j in h1.values().items():
#	order_dict[i[0]] = j[-3]
#	#order_dict[i[0]] = j[-2]
#
#orderd_dict = dict(sorted(order_dict.items(),key=(lambda x: x[1])))
#for i,j in orderd_dict.items():
#	order_dict[i[0]] = j[-3]
#	print("{0} : {1}".format(i,j))


region = args.region
order_dict = {}
for i, j in h1.integrate('region',region).values().items():
	order_dict[i[0]] = sum(j)

orderd_dict = dict(sorted(order_dict.items(),key=(lambda x: x[1])))
for i,j in orderd_dict.items():
	print("{0} : {1}".format(i,j))
	




# MC plotting
import re


if year == "2018":
	notdata = re.compile('(?!Egamma)')
	data = 'Egamma'

if year == "2017":
	notdata = re.compile('(?!DoubleEG)')
	data =  'DoubleEG'



# 2018 eee fake lepton baseline
order=['TTWJets','tZq','TTGJets','TTZtoLL','ZZ','WZ','Fake_Photon','ZGToLLG','FakeLepton','WZG']




hist.plot1d(
	h1[notdata].integrate('region',region),
	ax=ax,
	clear=False,
	stack=True,
	order= order,
	fill_opts=fill_opts,
	error_opts = error_opts,
)



# DATA plotting
hist.plot1d(
	h1[data].integrate('region',region),
	ax=ax,
	clear=False,
	error_opts=data_err_opts
	
)

# -- Ratio Plot
hist.plotratio(
    num=h1[data].integrate('region',region).sum("dataset"),
    denom=h1[notdata].integrate('region',region).sum("dataset"),
    ax=rax,
    error_opts=data_err_opts,
    denom_fill_opts={},
    guide_opts={},
    unc="num",
)

np.set_printoptions(suppress=True)


rax.set_ylabel("Data/MC")

rax.set_ylim(0, 2)


ax._get_lines.prop_cycler = ax._get_patches_for_fill.prop_cycler
ax.autoscale(axis="x", tight=True)
ax.set_ylim(ymin, ymax)
ax.set_xlim(xmin, xmax)
ax.set_xlabel('')


if histname == 'cutflow':
	ax.set_yscale('log')


lum = plt.text(
    1.0,
    1.0,
	r"%.2f fb$^{-1}$ (13 TeV)" % (lumi_factor),
    fontsize=16,
    horizontalalignment="right",
    verticalalignment="bottom",
    transform=ax.transAxes,
)


# -- Output png
outname = histname + "_" + file_name + ".png"

plt.savefig(outname)
#plt.show()

