import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time
import sys
from tqdm import tqdm
sys.path.append("util")
import Particle_Info_DB
import Hist_fake_dict
import mplhep as hep

Closure_bins = ["from_3_to_8"
,"from_3_to_9"
,"from_3_to_10"
,"from_3_to_11"
,"from_3_to_12"
,"from_3_to_13"
,"from_4_to_8"
,"from_4_to_9"
,"from_4_to_10"
,"from_4_to_11"
,"from_4_to_12"
,"from_4_to_13"
,"from_5_to_8"
,"from_5_to_9"
,"from_5_to_10"
,"from_5_to_11"
,"from_5_to_12"
,"from_5_to_13"
,"from_6_to_8"
,"from_6_to_9"
,"from_6_to_10"
,"from_6_to_11"
,"from_6_to_12"
,"from_6_to_13"
,"from_7_to_8"
,"from_7_to_9"
,"from_7_to_10"
,"from_7_to_11"
,"from_7_to_12"
,"from_7_to_13"
,"from_8_to_9"
,"from_8_to_10"
,"from_8_to_11"
,"from_8_to_12"
,"from_8_to_13"]

Closure_bins = ["from_4_to_10"]


def reduce(folder, sample_list, histname):
	hists = {}

	idx = 0
	for filename in os.listdir(folder):
		if filename.split("_")[0] not in sample_list:
			continue
		hin = load(folder + "/" + filename)
		
		hists[filename] = hin.copy()

		if idx == 0:
			hist_ = hists[filename][histname]

		else:
			hist_.add(hists[filename][histname])

		idx += 1

	return hist_


def set_limit(Eta_index):
	isEB_sieie = 0.01015
	isEE_sieie = 0.0326

	if (Eta_index == 1) or (Eta_index == 2):
		return isEB_sieie
	elif (Eta_index == 3) or (Eta_index == 4):
		return isEE_sieie
	else:
		raise ValueError

def get_fake_fraction(h1,Closure_bin,histname):

	# is barrel or endcap?
	Eta_index = int(histname.split("_")[-1])
	limit = set_limit(Eta_index)
	
	# convert coffea hist to numpy hist to extract axes and bins
	np_hist = h1.integrate('Closure_bin',Closure_bin).sum('dataset').to_boost().to_numpy()

	print("Yield: ",sum(np_hist[0]))
	mask = np_hist[1][:-1] <= limit # Calculate Yield of selected sieie ranges
	# Why [:-1]? --> Last bins of numpy axes in meaningless	


	# Calculate known fake fraction
	sum_Xgamma = 0
	sum_others = 0
	for i,j in h1.integrate('Closure_bin',Closure_bin).values().items():

		if 'G' in i[0]:
			sum_Xgamma += sum(j[mask])
		else:
			sum_others += sum(j[mask])
	
	total = sum_Xgamma + sum_others

	# Known fake fraction
	print('fake fraction: ', sum_others / total)
	
	
	return sum_others/total


def draw(h1,Closure_bin):

	plt.style.use(hep.style.CMS)
	plt.rcParams.update(
		{
			"font.size": 14,
			"axes.titlesize": 18,
			"axes.labelsize": 18,
			"xtick.labelsize": 12,
			"ytick.labelsize": 12,
		}
	)
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), sharex=True)
	
	
	fake_error_opts = {
		"linestyle": "none",
		"marker": "+",
		"markersize": 10.0,
		"color": "royalblue",
		"elinewidth": 1,
	}
	
	# -- Draw hist
	hist.plot1d(
		h1.integrate('Closure_bin',Closure_bin).sum('dataset'),
		ax=ax,
		clear=False,
		error_opts=fake_error_opts,
		#density=True
	)
	
	np.set_printoptions(suppress=True)
	ax.autoscale(axis="x", tight=True)
	ax.set_ylim(ymin, ymax)
	ax.set_xlim(xmin, xmax)
	# ax.set_xlabel('')
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
	outname = histname + "_" + file_name + ".png"
	plt.savefig(outname)


# Here we do not use numpy converting to perfectly extract data 
# if there is a problem, this method will occur error
# However, the converting method has probaibilty that does not occur error
# We use internal functions in coffea
# --> This will be improved
class Hist_to_array:
	def __init__(self, hist, name):

		self.hist = hist
		self.name = name

	def bin_content(self):

		c = self.hist.values()
		key = list(c.keys())[0]

		self.length = len(self.hist.identifiers(self.name))
		self.content = c[key]
		self.bin = np.array(
			[self.hist.identifiers(self.name)[i].mid for i in range(self.length)]
		)
	
		return self.length, self.bin, self.content



if __name__ == "__main__":


	# --- Setup basic paramters
	# 2017
	#year = "2017"
	#file_name = "210531_FakeTemplate_2017"
	
	# 2018
	year = "2018"
	file_name = "210721_FakeTemplate_closure_renew"
	
	file_path = "results/" + file_name
	sample_list = ["Fake", "Real", "data"]
	
	dict_ = Particle_Info_DB.DB
	lumi_factor = dict_[year]["Lumi"]
	GenDict = dict_[year]["Gen"]
	xsecDict = dict_[year]["xsec"]
	
	# --- Arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("hist_name", type=str, help="PT_1_eta_1")
	parser.add_argument("--save", type=bool, help="--save True : save npy",default=False)
	args = parser.parse_args()
	
	histname = args.hist_name
	
	hist_info = Hist_fake_dict.hist_info
	xmin = hist_info[histname]['xmin']
	xmax = hist_info[histname]['xmax']
	ymin = hist_info[histname]['ymin']
	ymax = hist_info[histname]['ymax']
	name = hist_info[histname]['name']
	bins = hist_info[histname]['bins']
	
	


	# --- Get histogram
	h1 = reduce(file_path, sample_list, histname)
	
	# --- Noramlize	
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
	h1 = h1.rebin(histname,hist.Bin(histname,name,bins, xmin,xmax))
	

	# -- Loop need here	
	Closure_bin  = "from_4_to_10"


	# --- Plotting
	draw(h1,Closure_bin)

	print('Calculating known fake fraction....')
	out_dict = {}
	in_dict = {}
	for Closure_bin in tqdm(Closure_bins):
		# --- Get Fake fraction	ff = get_fake_fraction(h1,Closure_bin,histname)
		in_dict[Closure_bin] = get_fake_fraction(h1,Closure_bin,histname)
		out_dict['known_fake_fraction'] = in_dict

	if args.save:
		# -- Extract bin, contents and save npy
		print("Start.. Data extracting.")
		
		# Extractor
		#out_dict = {}
		in_dict = {}
		for Closure_bin in tqdm(Closure_bins):
		
			length, bins, contents = Hist_to_array(h1.integrate('Closure_bin',Closure_bin).sum('dataset'),histname).bin_content()
			in_dict[Closure_bin] =  {'length': length, 'bins': bins, 'contents':contents}
			out_dict['Fake_template'] = in_dict
		print(out_dict.keys())
		print(out_dict['Fake_template'].keys())
		print(out_dict['known_fake_fraction'])
		
		# Save array
		from pathlib import Path
		output_directory = "Fitting_2018_ClosureTest"
		Path(output_directory + '/FakeTemplate').mkdir(exist_ok=True,parents=True) # exist_ok = True -> if file not exist -> mkdir file
		out_npy_name = output_directory + '/FakeTemplate/'+ histname + '.npy'
		np.save(out_npy_name,out_dict)
