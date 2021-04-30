import os
import numpy as np
from coffea.util import load, save
import matplotlib.pyplot as plt
import coffea.hist as hist
import time
import Particle_Info_DB


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

file_path = "210427_FakeTemplate_v2"
sample_list = ["Fake", "Real", "data"]

dict_ = Particle_Info_DB.DB
lumi_factor = dict_[year]["Lumi"]
GenDict = dict_[year]["Gen"]
xsecDict = dict_[year]["xsec"]


histname = "PT_1_eta_1"
xmin = 0
xmax = 0.02
ymin = 0.0
ymax = 4000
# histname = "PT_1_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=1000;
# histname = "PT_1_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=500;
# histname = "PT_1_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=300;
# histname = "PT_2_eta_1";xmin=0; xmax=0.02; ymin=0.; ymax=1500;
# histname = "PT_2_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=500;
# histname = "PT_2_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=200;
# histname = "PT_2_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=150;
# histname = "PT_3_eta_1";xmin=0; xmax=0.02; ymin=0.; ymax=500;
# histname = "PT_3_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=200;
# histname = "PT_3_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=100;
# histname = "PT_3_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=60;
# histname = "PT_4_eta_1";xmin=0; xmax=0.02; ymin=0.; ymax=1000;
# histname = "PT_4_eta_2";xmin=0; xmax=0.02; ymin=0.; ymax=300;
# histname = "PT_4_eta_3";xmin=0; xmax=0.05; ymin=0.; ymax=200;
# histname = "PT_4_eta_4";xmin=0; xmax=0.05; ymin=0.; ymax=100;

h1 = reduce(file_path, sample_list, histname)

## --Noramlize
scales = {
    "WZG": lumi_factor * 1000 * xsecDict["WZG"] / GenDict["WZG"],
    "DY": lumi_factor * 1000 * xsecDict["DY"] / GenDict["DY"],
    "WZ": lumi_factor * 1000 * xsecDict["WZ"] / GenDict["WZ"],
    "ZZ": lumi_factor * 1000 * xsecDict["ZZ"] / GenDict["ZZ"],
    "TTWJets": lumi_factor * 1000 * xsecDict["TTWJets"] / GenDict["TTWJets"],
    "TTZtoLL": lumi_factor * 1000 * xsecDict["TTZtoLL"] / GenDict["TTZtoLL"],
    "tZq": lumi_factor * 1000 * xsecDict["tZq"] / GenDict["tZq"],
    "ZGToLLG": lumi_factor * 1000 * xsecDict["ZGToLLG"] / GenDict["ZGToLLG"],
    "TTGJets": lumi_factor * 1000 * xsecDict["TTGJets"] / GenDict["TTGJets"],
    "WGToLNuG": lumi_factor * 1000 * xsecDict["WGToLNuG"] / GenDict["WGToLNuG"],
}
# h1.scale(scales,axis='dataset')
## --Rebin


h1 = h1.rebin(histname, hist.Bin("PT_1_eta_1", "20 < pt <30 & |eta| < 1", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_1_eta_2","20 < pt <30 & 1 < |eta| < 1.5", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_1_eta_3","20 < pt <30 & 1.5 < |eta| < 2", 50, 0, 0.05))
# h1 = h1.rebin(histname,hist.Bin("PT_1_eta_4","20 < pt <30 & 2 < |eta| < 2.5", 50, 0, 0.05))
# h1 = h1.rebin(histname,hist.Bin("PT_2_eta_1","30 < pt <40 & |eta| < 1", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_2_eta_2","30 < pt <40 & 1 < |eta| < 1.5", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_2_eta_3","30 < pt <40 & 1.5 < |eta| < 2", 50, 0, 0.05))
# h1 = h1.rebin(histname,hist.Bin("PT_2_eta_4","30 < pt <40 & 2 < |eta| < 2.5", 50, 0, 0.05))
# h1 = h1.rebin(histname,hist.Bin("PT_3_eta_1","40 < pt <50 & |eta| < 1", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_3_eta_2","40 < pt <50 & 1 < |eta| < 1.5", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_3_eta_3","40 < pt <50 & 1.5 < |eta| < 2", 50, 0, 0.05))
# h1 = h1.rebin(histname,hist.Bin("PT_3_eta_4","40 < pt <50 & 2 < |eta| < 2.5", 50, 0, 0.05))
# h1 = h1.rebin(histname,hist.Bin("PT_4_eta_1","50 < pt & |eta| < 1", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_4_eta_2","50 <pt  & 1 < |eta| < 1.5", 50, 0, 0.02))
# h1 = h1.rebin(histname,hist.Bin("PT_4_eta_3","50 < pt  & 1.5 < |eta| < 2", 50, 0, 0.05))
# h1 = h1.rebin(histname,hist.Bin("PT_4_eta_4","50 < pt  & 2 < |eta| < 2.5", 50, 0, 0.05))


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
    "marker": ".",
    "markersize": 10.0,
    "color": "royalblue",
    "elinewidth": 1,
}

real_error_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.0,
    "color": "darkorange",
    "elinewidth": 1,
}

data_error_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.0,
    "color": "black",
    "elinewidth": 1,
}


# Fake template
if isFake:
    hist.plot1d(
        h1["Fake_template"],
        ax=ax,
        clear=False,
        error_opts=fake_error_opts,
    )

# Real template
if isReal:
    hist.plot1d(
        h1["Real_template"],
        ax=ax,
        clear=False,
        error_opts=real_error_opts,
    )

# Data template
if isData:
    hist.plot1d(
        h1["data_template"],
        ax=ax,
        clear=False,
        error_opts=data_error_opts,
    )

np.set_printoptions(suppress=True)

ax.autoscale(axis="x", tight=True)
ax.set_ylim(ymin, ymax)
ax.set_xlim(xmin, xmax)
# ax.set_xlabel('')
# ax.set_yscale('log')

lum = plt.text(
    1.0,
    1.0,
    r"%.2f fb$^{-1}$ (13 TeV)" % (lumi_factor),
    fontsize=16,
    horizontalalignment="right",
    verticalalignment="bottom",
    transform=ax.transAxes,
)


outname = histname + "_" + file_path + ".png"

plt.savefig(outname)
plt.show()


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


## -- save function! -- on going
def make_data(templates):

    dict_ = {}
    for tp_name in templates:
        length, bins, contents = Hist_to_array(h1, histname).bin_content()
        arr = Hist_to_array(h1, histname).Extract_data()

        dict_[tp_name]["length"] = length
        dict_[tp_name]["bins"] = bins
        dict_[tp_name]["contents"] = contents
        dict_[tp_name]["data"] = arr

    return dict


templates = ["data_template", "Fake_template", "Real_template"]


length, bins, contents = Hist_to_array(h1, histname).bin_content()
arr = Hist_to_array(h1, histname).Extract_data()

plt.hist(arr, bins=bins)
plt.savefig("reco.png")
plt.show()


print(arr)
