import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time

## Parameter set
lumi= 53.03 * 1000


GenDict={
'WZG':55000,
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


hsum_mass_eee =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("mass_eee","$m_{eee}$ [GeV]", 300, 0, 600),
)


hsum_MT = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("MT","W MT [GeV]", 100, 0, 200)
)


hsum_dR_aj = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("dR_aj","$\delta R_{\gammaj}$", 100, 0, 4),
)
hsum_cutflow = hist.Hist(
	'Events',
	hist.Cat('dataset', 'Dataset'),
	hist.Bin('cutflow', 'Cut index', [0, 1, 2, 3,4,5,6,7])
)

hsum_charge= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("charge","charge sum of electrons", 6, -3, 3),
)
hsum_nPV = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("nPV","Number of Primary vertex",100,0,100)	
)
hsum_nPV_nw = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("nPV_nw","Number of Primary vertex",100,0,100)	
)
hsum_Mee = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("mass","Z mass",100,0,200)	
)

hsum_ele1pt = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("ele1pt","Leading Electron $P_{T}$ [GeV]",300,0,600)	
)
hsum_ele2pt =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("ele2pt","Subleading $Electron P_{T}$ [GeV]", 300, 0, 600),
)
hsum_ele3pt =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("ele3pt","Third $Electron P_{T}$ [GeV]", 300, 0, 600),
)

hsum_ele1eta= hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("ele1eta","Leading Electron $\eta$ [GeV]", 100, -5, 5),
)

hsum_ele2eta =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("ele2eta","Subleading Electron $\eta$ [GeV]", 100, -5, 5),
)
hsum_ele1phi =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("ele1phi","Leading Electron $\phi$ [GeV]", 100, -3.15, 3.15),
)

hsum_ele2phi =  hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("ele2phi","Subleading Electron $\phi$ [GeV]", 100, -3.15, 3.15),
)
hsum_nElectrons = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("nElectrons","# of Electrons",10,0,10)
)

hsum_pho_EE_pt =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EE_pt","Photon EE $P_{T}$ [GeV]", 300, 0, 600),
)

hsum_pho_EE_eta =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EE_eta","Photon EE $\eta$ ", 100, -5, 5),
)
hsum_pho_EE_phi =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EE_phi","Photon EE $\phi$ ", 100, -3.15, 3.15),
)
hsum_pho_EE_hoe = hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EE_hoe","Photon EE HoverE", 100, 0, 0.6),
)
hsum_pho_EE_sieie =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EE_sieie","Photon EE sieie", 100, 0, 0.3),
)
hsum_pho_EE_Iso_all =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EE_Iso_all","Photon EE pfReoIso03_all", 100, 0, 0.3),
)
hsum_pho_EE_Iso_chg =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EE_Iso_chg","Photon EE pfReoIso03_charge", 100, 0, 0.03),
)





hsum_pho_EB_pt =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_pt","Photon EB $P_{T}$ [GeV]", 300, 0, 600),
)
hsum_pho_EB_eta =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_eta","Photon EB $\eta$ ", 100, -5, 5),
)
hsum_pho_EB_phi =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_phi","Photon EB $\phi$ ", 100, -3.15, 3.15),
)
hsum_pho_EB_hoe = hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_hoe","Photon EB HoverE", 100, 0, 0.6),
)
hsum_pho_EB_sieie =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_sieie","Photon EB sieie", 100, 0, 0.012),
)
hsum_pho_EB_Iso_all =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_Iso_all","Photon EB pfReoIso03_all", 100, 0, 0.15),
)
hsum_pho_EB_Iso_chg =  hist.Hist(
    "Events",
    hist.Cat("dataset","Dataset"),
    hist.Bin("pho_EB_Iso_chg","Photon EB pfReoIso03_charge", 100, 0, 0.03),
)


hsum_met = hist.Hist(
	"Events",
	hist.Cat("dataset","Dataset"),
	hist.Bin("met","met [GeV]", 300, 0, 600),
)


histdict = {'nPV':hsum_nPV,'nPV_nw':hsum_nPV_nw, "cutflow":hsum_cutflow,"ele1pt":hsum_ele1pt,"ele2pt":hsum_ele2pt,"ele3pt":hsum_ele3pt,"mass":hsum_Mee,\
"ele1eta":hsum_ele1eta, "ele2eta":hsum_ele2eta,"ele1phi":hsum_ele1phi, "ele2phi":hsum_ele2phi,\
'pho_EE_pt':hsum_pho_EE_pt, 'pho_EE_eta':hsum_pho_EE_eta, 'pho_EE_phi':hsum_pho_EE_phi,\
'pho_EE_hoe':hsum_pho_EE_hoe,'pho_EE_sieie': hsum_pho_EE_sieie, 'pho_EE_Iso_all': hsum_pho_EE_Iso_all, 'pho_EE_Iso_chg':hsum_pho_EE_Iso_chg,\
'pho_EB_pt':hsum_pho_EB_pt, 'pho_EB_eta':hsum_pho_EB_eta, 'pho_EB_phi':hsum_pho_EB_phi,\
'pho_EB_hoe':hsum_pho_EB_hoe,'pho_EB_sieie': hsum_pho_EB_sieie, 'pho_EB_Iso_all': hsum_pho_EB_Iso_all, 'pho_EB_Iso_chg':hsum_pho_EB_Iso_chg,
'met':hsum_met,'dR_aj':hsum_dR_aj,'MT':hsum_MT,'mass_eee':hsum_mass_eee}



def reduce(folder,sample_list,histname):
	hists={}
	

	sumwdict ={
		"DY":0,
		"WZ":0,
		"ZZ":0,
		"TTWJets":0,
		"TTZtoLL":0,
		"tZq":0,
		"Egamma":0,
		"WZG":0,
		'ZGToLLG':0,	
        'TTGJets':0,
        'WGToLNuG':0
	}

	
	for filename in os.listdir(folder):
		hin = load(folder + '/' + filename)
		hists[filename] = hin.copy()
		if filename.split('_')[0] not in sample_list:
			continue
		sumwdict['DY'] += hists[filename]['sumw']['DY']
		sumwdict['WZ'] += hists[filename]['sumw']['WZ']
		sumwdict['ZZ'] += hists[filename]['sumw']['ZZ']
		sumwdict['TTWJets'] += hists[filename]['sumw']['TTWJets']
		sumwdict['TTZtoLL'] += hists[filename]['sumw']['TTZtoLL']
		sumwdict['tZq'] += hists[filename]['sumw']['tZq']
		sumwdict['Egamma'] += hists[filename]['sumw']['Egamma']
		sumwdict['WZG'] += hists[filename]['sumw']['WZG']
		sumwdict['ZGToLLG'] += hists[filename]['sumw']['ZGToLLG']
		sumwdict['TTGJets'] += hists[filename]['sumw']['TTGJets']
		sumwdict['WGToLNuG'] += hists[filename]['sumw']['WGToLNuG']


		hist_ = histdict[histname]
		hist_.add(hists[filename][histname])
		
	return hist_ , sumwdict


## --File Directories
file_path = "Base_line_210416_Allweight"
#file_path = "Base_line_210416_no_weight"
#file_path = "SR_210419"
#file_path = "CR_Zjets_210419"
#file_path = "CR_tenriched_210419"
#file_path = "CR_conversion_210419"
#file_path = "210420_SEN_BaseLine"


## --Sample Lists
sample_list = ['DY' ,'WZ' ,'ZZ' ,'TTWJets','TTZtoLL','tZq' ,'Egamma','WZG','ZGToLLG','TTGJets','WGToLNuG']




##############################################################33 --Hist names

		
		# --- MET --- #

#histname = "met"; xmin=0; xmax=200; ymin=0; ymax=20; 

		# --- Electron --- #
#histname = "mass"; xmin=0; xmax=200; ymin=0; ymax=20;
#histname = "mass_eee", xmin=0, xmax=600, ymin=0, ymax=10;
#histname = "MT"; xmin=0; xmax=200; ymin=0; ymax=8;
#histname = "nPV"; xmin=0; xmax=100; ymin=1; ymax=1e+3;
#histname = "nPV_nw"; xmin=0; xmax=100; ymin=1; ymax=1e+3;

#histname = "ele1pt"; xmin=0; xmax=200; ymin=0; ymax=10;
#histname = "ele2pt"; xmin=0; xmax=200; ymin=0; ymax=20; 
#histname = "ele3pt"; xmin=0; xmax=200; ymin=0; ymax=20;

#histname = "ele1eta"; xmin=-2.5; xmax=2.5; ymin=100; ymax=5e+6;
#histname = "ele2eta"; xmin=-2.5; xmax=2.5; ymin=100; ymax=5e+6;

#histname = "ele1phi"; xmin=-3.15; xmax=3.15; ymin=100; ymax=5e+6;
#histname = "ele2phi"; xmin=-3.15; xmax=3.15; ymin=100; ymax=5e+6;

#histname = "cutflow"; xmin=0; xmax=6; ymin=1; ymax=5e+6

		# --- Photon --- #

#histname = "pho_EB_pt"; xmin=0; xmax=200; ymin=0; ymax=15;
#histname = "pho_EB_eta"; xmin=-3; xmax=3; ymin=1; ymax=5e+6;
#histname = "pho_EB_phi"; xmin=-3.15; xmax=3.15; ymin=1; ymax=5e+6;
#histname = "pho_EB_hoe"; xmin=0; xmax=0.15; ymin=0.001; ymax=1e+7;
#histname = "pho_EB_sieie"; xmin=0; xmax=0.012; ymin=0.01; ymax=5e+6;
#histname = "pho_EB_Iso_all"; xmin=0; xmax=0.12; ymin=0.01; ymax=5e+7;
#histname = "pho_EB_Iso_chg"; xmin=0; xmax=0.03; ymin=0.01; ymax=5e+7;

#histname = "dR_aj"; xmin=0; xmax=1; ymin=0.01; ymax=10;
histname = "pho_EE_pt"; xmin=0; xmax=200; ymin=0; ymax=8;
#histname = "pho_EE_eta"; xmin=-3; xmax=3; ymin=1; ymax=5e+6;
#histname = "pho_EE_phi"; xmin=-3.15; xmax=3.15; ymin=1; ymax=5e+6;
#histname = "pho_EE_hoe"; xmin=0; xmax=0.2; ymin=0.001; ymax=5e+6;
#histname = "pho_EE_sieie"; xmin=0; xmax=0.3; ymin=0.01; ymax=5e+6;
#histname = "pho_EE_Iso_all"; xmin=0; xmax=0.2; ymin=0.01; ymax=5e+6;
#histname = "pho_EE_Iso_chg"; xmin=0; xmax=0.03; ymin=0.01; ymax=5e+6;



################################################################
## --All-reduce 
h1,sumwdict = reduce(file_path,sample_list,histname)

for i,j in sumwdict.items():
	print(i,": ",j)

## --Noramlize	
scales={
	'WZG'	  : lumi * xsecDict['WZG'] / GenDict['WZG'],
	'DY'	  : lumi * xsecDict['DY'] / GenDict['DY'],
	'WZ'	  : lumi * xsecDict['WZ'] / GenDict['WZ'],
	'ZZ'	  : lumi * xsecDict['ZZ'] / GenDict['ZZ'],
	'TTWJets' : lumi * xsecDict['TTWJets'] / GenDict['TTWJets'],
	'TTZtoLL' : lumi * xsecDict['TTZtoLL'] / GenDict['TTZtoLL'],
	'tZq'	  : lumi * xsecDict['tZq'] / GenDict['tZq'],
	'ZGToLLG' : lumi * xsecDict['ZGToLLG'] / GenDict['ZGToLLG'],
    'TTGJets' : lumi * xsecDict['TTGJets'] / GenDict['TTGJets'],
    'WGToLNuG': lumi * xsecDict['WGToLNuG'] / GenDict['WGToLNuG'],
}


h1.scale(scales,axis='dataset')

## --Rebin
#h1 = h1.rebin(histname,hist.Bin("met","met [GeV]", 10, 0, 200))
#h1 = h1.rebin(histname,hist.Bin("mass","Z mass",10,0,200))
#h1 = h1.rebin(histname,hist.Bin("MT","W MT [GeV]", 20, 0, 200))

#h1 = h1.rebin(histname,hist.Bin("ele1pt","Leading electron from Z  $P_{T}$ [GeV]",30,0,600))
#h1 = h1.rebin(histname,hist.Bin("ele2pt","Subleading electron from Z  $P_{T}$ [GeV]",30,0,600))
#h1 = h1.rebin(histname,hist.Bin("ele3pt","electron from W  $P_{T}$ [GeV]",30,0,600))
#h1 = h1.rebin(histname,hist.Bin("mass","M_{ee} [GeV]",50,0,200))

#h1 = h1.rebin(histname,hist.Bin("pho_EB_pt","Photon EB $P_{T}$ [GeV]", 30, 0, 600))
h1 = h1.rebin(histname,hist.Bin("pho_EE_pt","Photon EE $P_{T}$ [GeV]", 30, 0, 600))



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
	nrows=2,
	ncols=1,
	figsize=(7,7),
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


print("##" * 20)
for i,j in h1.values().items():
	print(i,":",j[-2])


# MC plotting
import re
notdata = re.compile('(?!Egamma)')

hist.plot1d(
	h1[notdata],
	ax=ax,
	clear=False,
	stack=True,
	order=['TTWJets' ,'tZq' ,'TTGJets' ,'TTZtoLL' ,'WZ' ,'ZZ' ,'ZGToLLG' ,'WZG'],
	fill_opts=fill_opts,
	error_opts = error_opts,
)


# DATA plotting
hist.plot1d(

	h1['Egamma'],
	ax=ax,
	clear=False,
	error_opts=data_err_opts
	
)

#print(h1['Egamma'].values())

# Ratio Plot
hist.plotratio(
	num=h1['Egamma'].sum("dataset"),
	denom=h1[notdata].sum("dataset"),
	ax=rax,
	error_opts=data_err_opts,
	denom_fill_opts={},
	guide_opts={},
	unc='num'
)

np.set_printoptions(suppress=True)



rax.set_ylabel('Data/MC')

rax.set_ylim(0,2)


ax._get_lines.prop_cycler = ax._get_patches_for_fill.prop_cycler
ax.autoscale(axis='x', tight=True)
ax.set_ylim(ymin,ymax)
ax.set_xlim(xmin,xmax)
ax.set_xlabel('')
#ax.set_yscale('log')


##leg = ax.legend()


lum = plt.text(1., 1., r"53.03 fb$^{-1}$ (13 TeV)",
				fontsize=16,
				horizontalalignment='right',
				verticalalignment='bottom',
				transform=ax.transAxes
			   )



outname = histname + "_" + file_path + ".png"

plt.savefig(outname)
plt.show()
