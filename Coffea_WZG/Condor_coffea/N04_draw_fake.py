import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time
import sys


sys.path.append("util")
import Particle_Info_DB
import Hist_fake_dict

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


# Parameter set
isData = True
isFake = True
isReal = True


year = "2018"

file_name = "210427_FakeTemplate_v2"

file_path = "results/" + file_name


sample_list = ["Fake", "Real", "data"]

dict_ = Particle_Info_DB.DB
lumi_factor = dict_[year]["Lumi"]
GenDict = dict_[year]["Gen"]
xsecDict = dict_[year]["xsec"]



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("hist_name", type=str, help="PT_1_eta_1")
args = parser.parse_args()



histname = args.hist_name

hist_info = Hist_fake_dict.hist_info
xmin = hist_info[histname]['xmin']
xmax = hist_info[histname]['xmax']
ymin = hist_info[histname]['ymin']
ymax = hist_info[histname]['ymax']
name = hist_info[histname]['name']
bins = hist_info[histname]['bins']



#histname = "PT_1_eta_1";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+6;
#histname = "PT_1_eta_2";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+5;
#histname = "PT_1_eta_3";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+5;
#histname = "PT_1_eta_4";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+5;

#histname = "PT_2_eta_1";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+6;
#histname = "PT_2_eta_2";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+5;
#histname = "PT_2_eta_3";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+4;
#histname = "PT_2_eta_4";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+4;

#histname = "PT_3_eta_1";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+5;
#histname = "PT_3_eta_2";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+5;
#histname = "PT_3_eta_3";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+4;
#histname = "PT_3_eta_4";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+4;

#histname = "PT_4_eta_1";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+5;
#histname = "PT_4_eta_2";xmin=0; xmax=0.02; ymin=0.1; ymax=1e+5;
#histname = "PT_4_eta_3";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+4;
#histname = "PT_4_eta_4";xmin=0; xmax=0.05; ymin=0.1; ymax=1e+4;

h1 = reduce(file_path, sample_list, histname)

## --Noramlize
scales = {
	"Real_template": lumi_factor * 1000 * xsecDict["WZG"] / GenDict["WZG"],
}
#h1.scale(scales,axis='dataset')
## --Rebin


#h1 = h1.rebin(histname, hist.Bin("PT_1_eta_1", "20 < pt <30 & |eta| < 1", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_1_eta_2","20 < pt <30 & 1 < |eta| < 1.5", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_1_eta_3","20 < pt <30 & 1.5 < |eta| < 2", 200, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_1_eta_4","20 < pt <30 & 2 < |eta| < 2.5", 200, 0, 0.05))

#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_1","30 < pt <40 & |eta| < 1", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_2","30 < pt <40 & 1 < |eta| < 1.5", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_3","30 < pt <40 & 1.5 < |eta| < 2", 200, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_2_eta_4","30 < pt <40 & 2 < |eta| < 2.5", 200, 0, 0.05))

#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_1","40 < pt <50 & |eta| < 1", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_2","40 < pt <50 & 1 < |eta| < 1.5", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_3","40 < pt <50 & 1.5 < |eta| < 2", 200, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_3_eta_4","40 < pt <50 & 2 < |eta| < 2.5", 200, 0, 0.05))

#h1 = h1.rebin(histname,hist.Bin("PT_4_eta_1","50 < pt & |eta| < 1", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_4_eta_2","50 <pt  & 1 < |eta| < 1.5", 200, 0, 0.02))
#h1 = h1.rebin(histname,hist.Bin("PT_4_eta_3","50 < pt  & 1.5 < |eta| < 2", 200, 0, 0.05))
#h1 = h1.rebin(histname,hist.Bin("PT_4_eta_4","50 < pt  & 2 < |eta| < 2.5", 200, 0, 0.05))

h1 = h1.rebin(histname,hist.Bin(histname,name,bins, xmin,xmax))


# ----> Plotting
print("End processing.. make plot")
print(" ")


import mplhep as hep

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

real_error_opts = {
	"linestyle": "none",
	"marker": "+",
	"markersize": 10.0,
	"color": "darkorange",
	"elinewidth": 1,
}

data_error_opts = {
	"linestyle": "none",
	"marker": "+",
	"markersize": 10.0,
	"color": "black",
	"elinewidth": 1,
}

print("###" * 10)

a = h1['Fake_template'].values()
b = h1['Real_template'].values()
c = h1['data_template'].values()
print(a)
print(b)
print(c)


# Fake template
if isFake:
	hist.plot1d(
		h1["Fake_template"],
		ax=ax,
		clear=False,
		error_opts=fake_error_opts,
		#density=True
	)



# Real template
if isReal:
	hist.plot1d(
		h1["Real_template"],
		ax=ax,
		clear=False,
		error_opts=real_error_opts,
		#density=True
	)

# Data template
if isData:
	hist.plot1d(
		h1["data_template"],
		ax=ax,
		clear=False,
		error_opts=data_error_opts,
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
#plt.show()


print("Start.. Data extracting.")

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

	def Extract_data(self):
		exr_arr = np.array([])

		_, bin_, content_ = self.bin_content()
		for x, y in zip(bin_, content_):
			if y == 0:
				continue

			sub_arr = np.ones(int(y)) * x
			exr_arr = np.append(exr_arr, sub_arr)
		return exr_arr



tp_names=['data_template', 'Fake_template', 'Real_template']
out_dict = {}

for tp_name in tp_names:
	length, bins, contents = Hist_to_array(h1[tp_name],histname).bin_content()
	in_dict = {}
	in_dict['length'] = length
	in_dict['bins'] = bins
	in_dict['contents'] = contents
	
	out_dict[tp_name] =  in_dict
	

print(out_dict)


for i in range(len(out_dict['data_template']['contents'])):
	print(\
	out_dict['data_template']['bins'][i]," ",
	out_dict['data_template']['contents'][i],
	out_dict['Fake_template']['contents'][i],
	out_dict['Real_template']['contents'][i]
	)


out_npy_name = "Fitting_v2/" + histname + '.npy'
np.save(out_npy_name,out_dict)
