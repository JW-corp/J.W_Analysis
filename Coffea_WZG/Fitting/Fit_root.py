import ROOT
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ctypes import *


label_name = {"PT_1_eta_1": "20 < pt <30 & |eta| < 1"
,"PT_1_eta_2":"20 < pt <30 & 1 < |eta| < 1.5"
,"PT_1_eta_3":"20 < pt <30 & 1.5 < |eta| < 2"
,"PT_1_eta_4":"20 < pt <30 & 2 < |eta| < 2.5"
,"PT_2_eta_1":"30 < pt <40 & |eta| < 1"
,"PT_2_eta_2":"30 < pt <40 & 1 < |eta| < 1.5"
,"PT_2_eta_3":"30 < pt <40 & 1.5 < |eta| < 2"
,"PT_2_eta_4":"30 < pt <40 & 2 < |eta| < 2.5"
,"PT_3_eta_1":"40 < pt <50 & |eta| < 1"
,"PT_3_eta_2":"40 < pt <50 & 1 < |eta| < 1.5"
,"PT_3_eta_3":"40 < pt <50 & 1.5 < |eta| < 2"
,"PT_3_eta_4":"40 < pt <50 & 2 < |eta| < 2.5"
,"PT_4_eta_1":"50 < pt & |eta| < 1"
,"PT_4_eta_2":"50 <pt  & 1 < |eta| < 1.5"
,"PT_4_eta_3":"50 < pt  & 1.5 < |eta| < 2"
,"PT_4_eta_4":"50 < pt  & 2 < |eta| < 2.5"}



def read_data(infile):
	indict = np.load(infile, allow_pickle=True)[()]
	return indict["data_template"], indict["Fake_template"], indict["Real_template"]


def Extract_data(bin_, content_):
	exr_arr = np.array([])

	for x, y in zip(bin_, content_):
		if y == 0:
			continue

		sub_arr = np.ones(int(y)) * x
		exr_arr = np.append(exr_arr, sub_arr)
	return exr_arr


def HisttoRoot(Extract_data, bins, data_template, Fake_template, Real_template,Eta_index):

	# Extract data from numpy hist
	data_arr = Extract_data(data_template["bins"], data_template["contents"])
	fake_arr = Extract_data(Fake_template["bins"], Fake_template["contents"])
	real_arr = Extract_data(Real_template["bins"], Real_template["contents"])

	# Set limit
	if (Eta_index == 1) or (Eta_index == 2):
		print("is Barrel!")
		isbarrel = True
	else:
		print("is Endcap!")
		isbarrel = False

	bins = bins
	start = 0
	if isbarrel:
		end = 0.02
	else:
		end = 0.05
	# Make ROOT-hist
	h_data = ROOT.TH1D("Data", "Data template", bins, start, end)
	h_fake = ROOT.TH1D("Fake", "Fake template", bins, start, end)
	h_real = ROOT.TH1D("Real", "Real template", bins, start, end)

	for i in data_arr:
		h_data.Fill(i)

	for i in fake_arr:
		h_fake.Fill(i)

	for i in real_arr:
		h_real.Fill(i)

	return h_data, h_fake, h_real


def set_limit(Eta_index):
	isEB_sieie = 0.01015
	isEE_sieie = 0.0326

	if (Eta_index == 1) or (Eta_index == 2):
		return isEB_sieie
	elif (Eta_index == 3) or (Eta_index == 4):
		return isEE_sieie
	else:
		raise ValueError



def fit(h_data,mc):
	ffitter = ROOT.TFractionFitter(h_data, mc)
	fit_status = ffitter.Fit()
	print("Fit status: ",fit_status)

	# pre-defined c-type value to store parameter
	frac_fake, frac_real = c_double(-1),c_double(-1)
	error_fake, error_real = c_double(-1), c_double(-1)

	# get information about parameters
	ffitter.GetResult(0, frac_fake, error_fake) # Fake SF
	ffitter.GetResult(1, frac_real, error_real) # Real SF

	# type change c-type -> python
	SF_fake  = frac_fake.value
	SF_real  = frac_real.value
	err_fake = frac_fake.value
	err_real = frac_real.value

	return ffitter,SF_fake,SF_real,err_fake,err_real


if __name__ == "__main__":

	# Multi-thread On
	ROOT.ROOT.EnableImplicitMT()


	# --Read data
	parser = argparse.ArgumentParser()
	parser.add_argument("infile_name", type=str, help="python Fit.py PT_1_eta_1.npy")
	parser.add_argument("bins", type=int, help="python Fit.py PT_1_eta_1.npy 200")
	args = parser.parse_args()

	infile_name = args.infile_name
	name = infile_name.split('.')[0]
	Eta_index = int(infile_name.split(".")[0].split("_")[-1])

	data_template, Fake_template, Real_template = read_data(infile_name)

	# --Numpy hist to ROOT hist
	h_data, h_fake, h_real = HisttoRoot(
		Extract_data, args.bins, data_template, Fake_template, Real_template,Eta_index
	)


	# --Draw hist and Fit
	mc = ROOT.TObjArray(2)
	mc.Add(h_fake)  # from data-driven
	mc.Add(h_real)

	# Draw "Before fit"
	c1 = ROOT.TCanvas("c1", "c1", 5, 50, 500, 500)
	ROOT.gStyle.SetOptStat(0)
	h_data.SetTitle(label_name[name])
	h_data.GetXaxis().SetTitle("Photon #sigma_{i#eta i#eta}")
	h_data.SetLineColor(1)
	h_fake.SetLineColor(9)
	h_real.SetLineColor(800)
	h_data.Draw("Ep")
	h_real.Draw("same hist")
	h_fake.Draw("same hist")


	legend = ROOT.TLegend(0.65, 0.89, 0.90, 0.65)
	legend.SetFillStyle(0)
	legend.SetBorderSize(0)
	legend.SetTextSize(0)
	l01 = ROOT.TLegendEntry
	l01 = legend.AddEntry(h_data, "data template", "l")
	l01.SetTextColor(h_data.GetLineColor())
	l01 = legend.AddEntry(h_fake, "fake template", "l")
	l01.SetTextColor(h_fake.GetLineColor())
	l01 = legend.AddEntry(h_real, "real template", "l")
	l01.SetTextColor(h_real.GetLineColor())
	legend.Draw()
	c1.SaveAs(name + "_before_fit_" + ".png")


	# Fitting
	ffitter,frac_fake,frac_real,err_fake,err_real = fit(h_data,mc)

	# Scale factor
	SF_fake = frac_fake * h_data.Integral()	/ h_fake.Integral()
	SF_real = frac_real * h_data.Integral()	/ h_real.Integral()
	print("SF_fake = {0:.2f}, SF_real = {1:.2f}".format(SF_fake,SF_real))

	# Weight hist
	h_fake.Scale(SF_fake)
	h_real.Scale(SF_real)

	# Calculate the Fake fraction
	limit = set_limit(Eta_index)
	fake_fraction = (
		SF_fake
		* h_fake.Integral(0, h_fake.GetXaxis().FindFixBin(limit))
		/ h_data.Integral(0, h_data.GetXaxis().FindFixBin(limit))
	)

	print("Fake fraction in medium cut range: ", fake_fraction)

	# Draw "After fit"
	h_data.Draw("Ep")

	h_fit = ffitter.GetPlot()
	h_fit.SetLineColor(2)
	h_fit.Draw("same hist")
	h_fake.Draw("same hist")
	h_real.Draw("same hist")
	
	legend = ROOT.TLegend(0.65, 0.89, 0.90, 0.65)
	legend.SetFillStyle(0)
	legend.SetBorderSize(0)
	legend.SetTextSize(0)
	l01 = ROOT.TLegendEntry
	l01 = legend.AddEntry(h_data, "data template", "l")
	l01.SetTextColor(h_data.GetLineColor())
	l01 = legend.AddEntry(h_fit, "data fitted", "l")
	l01.SetTextColor(h_fit.GetLineColor())
	l01 = legend.AddEntry(h_fake, "fake component", "l")
	l01.SetTextColor(h_fake.GetLineColor())
	l01 = legend.AddEntry(h_real, "real component", "l")
	l01.SetTextColor(h_real.GetLineColor())
	legend.Draw()


	c1.ForceUpdate()
	c1.Modified()
	c1.SaveAs(name + '.png')
