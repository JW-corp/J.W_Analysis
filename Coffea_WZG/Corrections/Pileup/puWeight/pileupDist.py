#!/usr/bin/env python

import argparse
import sys, os
import fnmatch
import time
import ROOT
import numpy as np


def FindFiles(directory, pattern):
	lfiles = []
	for root, dirs, files in os.walk(directory):
		for basename in files:
			if fnmatch.fnmatch(basename, pattern):
				filename = os.path.join(root, basename)
				lfiles.append(filename)
	return lfiles


class PileupWeight():
	"""
	Usage:
	puWeight = PileupWeight("data.root","mc.root")
	weightFactor = puWeight(Events.Pileup_nTrueInt) 
	"""
	def __init__(self, filenameData = None, filenameMC = None):
		file_data = ROOT.TFile.Open(filenameData,"read")
		file_mc = ROOT.TFile.Open(filenameMC,"read")
		h1_data = file_data.Get("pileup")
		h1_mc = file_mc.Get("h1_Pileup_nTrueInt")
		dataY = [h1_data.GetBinContent(idx) for idx in xrange(1,h1_data.GetNbinsX()+1)]
		mcY = [h1_mc.GetBinContent(idx) for idx in xrange(1,h1_mc.GetNbinsX()+1)]
		file_data.Close()
		file_mc.Close()
		mcY[:] = [x / sum(mcY) * sum(dataY) for x in mcY]
		self.weights = [0 if m == 0 else d / m for d, m in zip(dataY, mcY)]

	def Weight(self, idx):
		return self.weights[idx]

	def WeightList(self):
		return self.weights


def MakeMCPileupDist(inFiles, outName):
	chain = ROOT.TChain("Events")
	nFiles = len(inFiles)
	idx = 0
	for idx, inFile in zip(xrange(0,nFiles), inFiles):
		print "Adding file : ", idx, "/", nFiles, inFile
		chain.Add(inFile)

	nEvents = chain.GetEntries()
	outFile = ROOT.TFile.Open(outName, 'create')
	h1_Pileup_nTrueInt = ROOT.TH1D("h1_Pileup_nTrueInt","h1_Pileup_nTrueInt",100,0,100)
	chain.Draw("Pileup_nTrueInt >> h1_Pileup_nTrueInt")
	outFile.Write()
	print "Done ", nFiles, " files " , nEvents , " " , h1_Pileup_nTrueInt.GetEntries(), " events", "to ", outName  



def TestPileupWeight(filenameData, filenameMC):
	puWeight = PileupWeight(filenameData,filenameMC)	

	file_data = ROOT.TFile.Open(filenameData,"read")
	file_mc = ROOT.TFile.Open(filenameMC,"read")
	h1_data = file_data.Get("pileup")
	h1_mc = file_mc.Get("h1_Pileup_nTrueInt")

	h1_test_data = ROOT.TH1D("h1_test_data","h1_test_data",100,0,100)
	h1_test_data.SetMarkerStyle(7)
	h1_test_mc = ROOT.TH1D("h1_test_mc","h1_test_mc",100,0,100)
	h1_test_mc.SetLineWidth(1)
	h1_test_mc.SetLineColor(2)
	h1_test_mcPU = ROOT.TH1D("h1_test_mcPU","h1_test_mcPU",100,0,100)
	h1_test_mcPU.SetLineWidth(1)
	h1_test_mcPU.SetLineColor(3)
	for i in range(10000000):
		h1_test_data.Fill(int(h1_data.GetRandom()))
		nTrueInt = int(h1_mc.GetRandom())
		h1_test_mc.Fill(nTrueInt)
		h1_test_mcPU.Fill(nTrueInt, puWeight.Weight(nTrueInt))
		

	c1 = ROOT.TCanvas()
	c1.Divide(2,1)
	c1.GetPad(2).SetLogy()
	c1.cd(1)
	h1_test_mc.Draw("Hist")
	h1_test_mcPU.Draw("same Hist")
	h1_test_data.Draw("same EP")
	c1.cd(2)
	h1_test_mc.Draw("Hist")
	h1_test_mcPU.Draw("same Hist")
	h1_test_data.Draw("same EP")
	c1.Print("testPUweight.png")
	return  puWeight.WeightList()





if __name__ == '__main__':
	
#	# -- For making MC hist
#	parser = argparse.ArgumentParser()
#
#	parser.add_argument('--infile', type=str)
#	parser.add_argument('--outfile', type=str)
#	args = parser.parse_args()
#	
#	datadir = args.infile
#	outname = args.outfile
#
#	inFiles = FindFiles(datadir, "*.root")  # from DAS file structure
#	#inFiles = [datadir] # Ntuple 
#
#	
#	MakeMCPileupDist(inFiles, outname)

# ------------------------------------------------------#


	# -- For makine SF
	infile = sys.argv[1]
	startTime = time.time()

	infile_path = "mc_pure/2017/" + infile
	outfile_name = infile.split('.')[0] + '.npy'
	puWeight_arr = TestPileupWeight("../2017/data_2018_pileup_out.root",infile_path)	
	print(puWeight_arr)
	print(len(puWeight_arr))
	

	np.save(outfile_name,np.array(puWeight_arr))

	print("RunningTime : ", time.time() - startTime)
