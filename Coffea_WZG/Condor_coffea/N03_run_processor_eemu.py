import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import time
from coffea import processor, hist
from coffea.util import load, save
import json
import glob
import os
import argparse
import numpy as np
from coffea import lumi_tools
import numba
import pandas as pd


# -- Coffea 0.8.0 --> Must fix!!
import warnings

warnings.filterwarnings("ignore")


# ---> Class JW Processor
class JW_Processor(processor.ProcessorABC):

	# -- Initializer
	def __init__(self, year, sample_name, puweight_arr, corrections,isFake):

		# Parameter set
		self._year = year
		self._isFake = isFake


		# Trigger set
		self._doubleelectron_triggers = {
			"2018": [
				"Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",  # Recomended
			],
			"2017": [
				"Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",  # Recomended
			],
		}

		self._singleelectron_triggers = (
			{  # 2017 and 2018 from monojet, applying dedicated trigger weights
				"2016": ["Ele27_WPTight_Gsf", "Ele105_CaloIdVT_GsfTrkIdT"],
				"2017": ["Ele35_WPTight_Gsf", "Ele115_CaloIdVT_GsfTrkIdT", "Photon200"],
				"2018": [
					"Ele32_WPTight_Gsf",  # Recomended
				],
			}
		)

		# Corrrection set
		self._corrections = corrections
		self._puweight_arr = puweight_arr

		# hist set
		self._accumulator = processor.dict_accumulator(
			{
				"sumw": processor.defaultdict_accumulator(float),
				"cutflow": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("cutflow", "Cut index", [0, 1, 2, 3, 4, 5, 6, 7,8]),
				),
				"nPV": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("nPV", "Number of Primary vertex", 100, 0, 100),
				),
				"nPV_nw": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("nPV_nw", "Number of Primary vertex", 100, 0, 100),
				),
				# -- Kinematics -- #
				"mass": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("mass", "$m_{e+e-}$ [GeV]", 100, 0, 200),
				),
				"mass_eea": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("mass_eea", "$m_{e+e-\gamma}$ [GeV]", 300, 0, 600),
				),
				"MT": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("MT", "W MT [GeV]", 100, 0, 200),
				),
				"met": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("met", "met [GeV]", 300, 0, 600),
				),

				# -- Muon -- #
		
				"mupt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("mupt","Leading Muon  $P_{T}$ [GeV]", 300, 0, 600),
				),
				"mueta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("mueta", "Leading Muon $\eta$ [GeV]", 20, -5, 5),
				),
				"muphi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"muphi", "Leading Muon $\phi$ [GeV]", 20, -3.15, 3.15
					),
				),


				# -- Electron -- #
				"ele1pt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("ele1pt", "Leading Electron $P_{T}$ [GeV]", 300, 0, 600),
				),
				"ele2pt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"ele2pt", "Subleading $Electron P_{T}$ [GeV]", 300, 0, 600
					),
				),
				"ele1eta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("ele1eta", "Leading Electron $\eta$ [GeV]", 20, -5, 5),
				),
				"ele2eta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("ele2eta", "Subleading Electron $\eta$ [GeV]", 20, -5, 5),
				),
				"ele1phi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"ele1phi", "Leading Electron $\phi$ [GeV]", 20, -3.15, 3.15
					),
				),
				"ele2phi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"ele2phi", "Subleading Electron $\phi$ [GeV]", 20, -3.15, 3.15
					),
				),
				# -- Photon -- #
				"phopt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("phopt", "Leading Photon $P_{T}$ [GeV]", 300, 0, 600),
				),
				"phoeta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("phoeta", "Photon  $\eta$ ", 50, -5, 5),
				),
				"phophi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("phophi", "Photon $\phi$ ", 50, -3.15, 3.15),
				),
				# -- Photon Endcap -- #
				"pho_EE_pt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EE_pt", "Photon EE $P_{T}$ [GeV]", 300, 0, 600),
				),
				"pho_EE_eta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EE_eta", "Photon EE $\eta$ ", 50, -5, 5),
				),
				"pho_EE_phi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EE_phi", "Photon EE $\phi$ ", 50, -3.15, 3.15),
				),
				"pho_EE_hoe": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EE_hoe", "Photon EE HoverE", 100, 0, 0.6),
				),
				"pho_EE_sieie": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EE_sieie", "Photon EE sieie", 100, 0, 0.03),
				),
				"pho_EE_Iso_all": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EE_Iso_all", "Photon EE pfReoIso03_all", 100, 0, 0.3),
				),
				"pho_EE_Iso_chg": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"pho_EE_Iso_chg", "Photon EE pfReoIso03_charge", 200, 0, 1
					),
				),
				# -- Photon Barrel -- #
				"pho_EB_pt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EB_pt", "Photon EB $P_{T}$ [GeV]", 300, 0, 600),
				),
				"pho_EB_eta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EB_eta", "Photon EB $\eta$ ", 50, -5, 5),
				),
				"pho_EB_phi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EB_phi", "Photon EB $\phi$ ", 50, -3.15, 3.15),
				),
				"pho_EB_hoe": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EB_hoe", "Photon EB HoverE", 100, 0, 0.6),
				),
				"pho_EB_sieie": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EB_sieie", "Photon EB sieie", 100, 0, 0.012),
				),
				"pho_EB_Iso_all": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"pho_EB_Iso_all", "Photon EB pfReoIso03_all", 100, 0, 0.15
					),
				),
				"pho_EB_Iso_chg": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"pho_EB_Iso_chg", "Photon EB pfReoIso03_charge", 100, 0, 0.03
					),
				),
			}
		)

	# -- Accumulator: accumulate histograms
	@property
	def accumulator(self):
		return self._accumulator

	# -- Main function : Process events
	def process(self, events):

		# Initialize accumulator
		out = self.accumulator.identity()
		dataset = sample_name
		# events.metadata['dataset']

		# Data or MC
		isData = "genWeight" not in events.fields
		isFake = self._isFake


		# Stop processing if there is no event remain
		if len(events) == 0:
			return out

		# Golden Json file
		if (self._year == "2018") and isData:
			injson = "/x5/cms/jwkim/gitdir/JWCorp/JW_analysis/Coffea_WZG/Corrections/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt.RunABD"

		if (self._year == "2017") and isData:
			injson = "/x5/cms/jwkim/gitdir/JWCorp/JW_analysis/Coffea_WZG/Corrections/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"

		# <----- Get Scale factors ------>#

		if not isData:

			# Egamma reco ID
			get_ele_reco_above20_sf = self._corrections["get_ele_reco_above20_sf"][
				self._year
			]
			get_ele_medium_id_sf = self._corrections["get_ele_medium_id_sf"][self._year]
			get_pho_medium_id_sf = self._corrections["get_pho_medium_id_sf"][self._year]

			# DoubleEG trigger # 2016, 2017 are not applied yet
			if self._year == "2018":
				get_ele_trig_leg1_SF = self._corrections["get_ele_trig_leg1_SF"][
					self._year
				]
				get_ele_trig_leg1_data_Eff = self._corrections[
					"get_ele_trig_leg1_data_Eff"
				][self._year]
				get_ele_trig_leg1_mc_Eff = self._corrections[
					"get_ele_trig_leg1_mc_Eff"
				][self._year]
				get_ele_trig_leg2_SF = self._corrections["get_ele_trig_leg2_SF"][
					self._year
				]
				get_ele_trig_leg2_data_Eff = self._corrections[
					"get_ele_trig_leg2_data_Eff"
				][self._year]
				get_ele_trig_leg2_mc_Eff = self._corrections[
					"get_ele_trig_leg2_mc_Eff"
				][self._year]
			
			# Muon ID, Iso
			get_mu_tight_id_sf  = self._corrections['get_mu_tight_id_sf'][self._year]
			get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf'][self._year]

			# PU weight with custom made npy and multi-indexing
			pu_weight_idx = ak.values_astype(events.Pileup.nTrueInt, "int64")
			pu = self._puweight_arr[pu_weight_idx]



		# <----- Helper functions ------>#

		#  Sort by PT  helper function
		def sort_by_pt(ele, pho, jet):
			ele = ele[ak.argsort(ele.pt, ascending=False, axis=1)]
			pho = pho[ak.argsort(pho.pt, ascending=False, axis=1)]
			jet = jet[ak.argsort(jet.pt, ascending=False, axis=1)]

			return ele, pho, jet

		# Lorentz vectors
		from coffea.nanoevents.methods import vector

		ak.behavior.update(vector.behavior)

		def TLorentz_vector(vec):
			vec = ak.zip(
				{"x": vec.x, "y": vec.y, "z": vec.z, "t": vec.t},
				with_name="LorentzVector",
			)
			return vec

		def TLorentz_vector_cylinder(vec):

			vec = ak.zip(
				{
					"pt": vec.pt,
					"eta": vec.eta,
					"phi": vec.phi,
					"mass": vec.mass,
				},
				with_name="PtEtaPhiMLorentzVector",
			)

			return vec

		# <----- Selection ------>#

		Initial_events = events
		# Good Run ( Golden Json files )
		from coffea import lumi_tools

		if isData:
			lumi_mask_builder = lumi_tools.LumiMask(injson)
			lumimask = ak.Array(
				lumi_mask_builder.__call__(events.run, events.luminosityBlock)
			)
			events = events[lumimask]
			# print("{0}%  of files pass good-run conditions".format(len(events)/ len(Initial_events)))

		# Stop processing if there is no event remain
		if len(events) == 0:
			return out


		# Cut flow
		cut0 = np.zeros(len(events))
		if not isFake:
			out["cutflow"].fill(dataset=dataset, cutflow=cut0)

		##----------- Cut flow1: Passing Triggers

		# double lepton trigger
		is_double_ele_trigger = True
		if not is_double_ele_trigger:
			double_ele_triggers_arr = np.ones(len(events), dtype=np.bool)
		else:
			double_ele_triggers_arr = np.zeros(len(events), dtype=np.bool)
			for path in self._doubleelectron_triggers[self._year]:
				if path not in events.HLT.fields:
					continue
				double_ele_triggers_arr = double_ele_triggers_arr | events.HLT[path]

		# single lepton trigger
		is_single_ele_trigger = True
		if not is_single_ele_trigger:
			single_ele_triggers_arr = np.ones(len(events), dtype=np.bool)
		else:
			single_ele_triggers_arr = np.zeros(len(events), dtype=np.bool)
			for path in self._singleelectron_triggers[self._year]:
				if path not in events.HLT.fields:
					continue
				single_ele_triggers_arr = single_ele_triggers_arr | events.HLT[path]

		events.Electron, events.Photon, events.Jet = sort_by_pt(
			events.Electron, events.Photon, events.Jet
		)

		# Good Primary vertex
		nPV = events.PV.npvsGood
		nPV_nw = events.PV.npvsGood
		if not isData:
			nPV = nPV * pu

			print(pu)


		# Apply cut1
		events = events[double_ele_triggers_arr]
		if not isData:
			pu = pu[double_ele_triggers_arr]

		# Stop processing if there is no event remain
		if len(events) == 0:
			return out

		cut1 = np.ones(len(events))
		if not isFake:
			out["cutflow"].fill(dataset=dataset, cutflow=cut1)

		# Set Particles
		Electron = events.Electron
		Muon = events.Muon
		Photon = events.Photon
		MET = events.MET
		Jet = events.Jet


		##----------- Cut flow2: Muon Selection
		MuSelmask = (
			(Muon.pt >= 10)
			& (abs(Muon.eta) <= 2.5)
			& (Muon.tightId)
			& (Muon.pfRelIso04_all < 0.15)
		)
		Muon = Muon[MuSelmask]
		

		# Exatly one muon
		Muon_sel_mask = (ak.num(Muon) == 1)
		Electron = Electron[Muon_sel_mask]
		Photon = Photon[Muon_sel_mask]
		Jet = Jet[Muon_sel_mask]
		MET = MET[Muon_sel_mask]
		Muon = Muon[Muon_sel_mask]
		events = events[Muon_sel_mask]
		if not isData:
			pu = pu[Muon_sel_mask]


		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		cut2 = np.ones(len(Photon)) * 2
		if not isFake:
			out["cutflow"].fill(dataset=dataset, cutflow=cut2)

		##----------- Cut flow3: Electron Selection

		EleSelmask = (
			(Electron.pt >= 10)
			& (np.abs(Electron.eta + Electron.deltaEtaSC) < 1.479)
			& (Electron.cutBased > 2)
			& (abs(Electron.dxy) < 0.05)
			& (abs(Electron.dz) < 0.1)
		) | (
			(Electron.pt >= 10)
			& (np.abs(Electron.eta + Electron.deltaEtaSC) > 1.479)
			& (np.abs(Electron.eta + Electron.deltaEtaSC) <= 2.5)
			& (Electron.cutBased > 2)
			& (abs(Electron.dxy) < 0.1)
			& (abs(Electron.dz) < 0.2)
		)

		Electron = Electron[EleSelmask]

		
		# Exactly two electrons
		ee_mask = ak.num(Electron) == 2
		Electron = Electron[ee_mask]
		Photon = Photon[ee_mask]
		Jet = Jet[ee_mask]
		MET = MET[ee_mask]
		Muon = Muon[ee_mask]
		if not isData:
			pu = pu[ee_mask]
		events = events[ee_mask]
		
		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		cut3 = np.ones(len(Photon)) * 3
		if not isFake:
			out["cutflow"].fill(dataset=dataset, cutflow=cut3)


		##----------- Cut flow4: Photon Selection

		# Basic photon selection
		isgap_mask = (abs(Photon.eta) < 1.442) | (
			(abs(Photon.eta) > 1.566) & (abs(Photon.eta) < 2.5)
		)
		Pixel_seed_mask = ~Photon.pixelSeed

		if (dataset == "ZZ") and (self._year == "2017"):
			PT_ID_mask = (Photon.pt >= 20) & (
				Photon.cutBasedBitmap >= 3
			)  # 2^0(Loose) + 2^1(Medium) + 2^2(Tights)
		else:
			PT_ID_mask = (Photon.pt >= 20) & (Photon.cutBased > 1)

		# dR cut with selected Muon and Electrons
		dr_pho_ele_mask = ak.all(
			Photon.metric_table(Electron) >= 0.5, axis=-1
		)  # default metric table: delta_r
		dr_pho_mu_mask = ak.all(Photon.metric_table(Muon) >= 0.5, axis=-1)

		# genPartFlav cut
		"""
		if dataset == "WZG":
			isPrompt = (Photon.genPartFlav == 1) | (Photon.genPartFlav == 11)
			PhoSelmask = PT_ID_mask & isgap_mask &  Pixel_seed_mask & isPrompt & dr_pho_ele_mask & dr_pho_mu_mask

		elif dataset == "WZ":
			isPrompt = (Photon.genPartFlav == 1) 
			PhoSelmask = PT_ID_mask & isgap_mask &  Pixel_seed_mask & ~isPrompt & dr_pho_ele_mask & dr_pho_mu_mask
				
		else:
			PhoSelmask = PT_ID_mask  & isgap_mask &  Pixel_seed_mask & dr_pho_ele_mask & dr_pho_mu_mask
		"""

		PhoSelmask = (
			PT_ID_mask & isgap_mask & Pixel_seed_mask & dr_pho_ele_mask & dr_pho_mu_mask
		)
		Photon = Photon[PhoSelmask]

		# Apply cut 4
		A_photon_mask = ak.num(Photon) > 0
		Electron = Electron[A_photon_mask]
		Photon = Photon[A_photon_mask]
		Jet = Jet[A_photon_mask]
		Muon = Muon[A_photon_mask]
		MET = MET[A_photon_mask]
		if not isData:
			pu = pu[A_photon_mask]
		events = events[A_photon_mask]

		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		def make_leading_pair(target, base):
			return target[ak.argmax(base.pt, axis=1, keepdims=True)]

		leading_pho = make_leading_pair(Photon, Photon)

		# -------------------- Make Fake Photon BKGs---------------------------#
		def make_bins(pt, eta, bin_range_str):

			bin_dict = {
				"PT_1_eta_1": (pt > 20) & (pt < 30) & (eta < 1),
				"PT_1_eta_2": (pt > 20) & (pt < 30) & (eta > 1) & (eta < 1.5),
				"PT_1_eta_3": (pt > 20) & (pt < 30) & (eta > 1.5) & (eta < 2),
				"PT_1_eta_4": (pt > 20) & (pt < 30) & (eta > 2) & (eta < 2.5),
				"PT_2_eta_1": (pt > 30) & (pt < 40) & (eta < 1),
				"PT_2_eta_2": (pt > 30) & (pt < 40) & (eta > 1) & (eta < 1.5),
				"PT_2_eta_3": (pt > 30) & (pt < 40) & (eta > 1.5) & (eta < 2),
				"PT_2_eta_4": (pt > 30) & (pt < 40) & (eta > 2) & (eta < 2.5),
				"PT_3_eta_1": (pt > 40) & (pt < 50) & (eta < 1),
				"PT_3_eta_2": (pt > 40) & (pt < 50) & (eta > 1) & (eta < 1.5),
				"PT_3_eta_3": (pt > 40) & (pt < 50) & (eta > 1.5) & (eta < 2),
				"PT_3_eta_4": (pt > 40) & (pt < 50) & (eta > 2) & (eta < 2.5),
				"PT_4_eta_1": (pt > 50) & (eta < 1),
				"PT_4_eta_2": (pt > 50) & (eta > 1) & (eta < 1.5),
				"PT_4_eta_3": (pt > 50) & (eta > 1.5) & (eta < 2),
				"PT_4_eta_4": (pt > 50) & (eta > 2) & (eta < 2.5),
			}

			binmask = bin_dict[bin_range_str]

			return binmask

		bin_name_list = [
			"PT_1_eta_1",
			"PT_1_eta_2",
			"PT_1_eta_3",
			"PT_1_eta_4",
			"PT_2_eta_1",
			"PT_2_eta_2",
			"PT_2_eta_3",
			"PT_2_eta_4",
			"PT_3_eta_1",
			"PT_3_eta_2",
			"PT_3_eta_3",
			"PT_3_eta_4",
			"PT_4_eta_1",
			"PT_4_eta_2",
			"PT_4_eta_3",
			"PT_4_eta_4",
		]



		## -- Fake-fraction Lookup table --##
		if isFake:
			# Make Bin-range mask
			binned_pteta_mask = {}
			for name in bin_name_list:
				binned_pteta_mask[name] = make_bins(
					ak.flatten(leading_pho.pt),
					ak.flatten(abs(leading_pho.eta)),
					name,
				)
			# Read Fake fraction --> Mapping bin name to int()
			in_dict = np.load('Fitting_v2/results_210517.npy',allow_pickle="True")[()]
			idx=0
			fake_dict ={}
			for i,j in in_dict.items():
				fake_dict[idx] = j
				idx+=1


			# Reconstruct Fake_weight
			fw= 0
			for i,j in binned_pteta_mask.items():
				fw = fw + j*fake_dict[bin_name_list.index(i)]


			# Process 0 weight to 1
			@numba.njit
			def zero_one(x):
				if x == 0:
					x = 1
				return x
			vec_zero_one = np.vectorize(zero_one)
			fw = vec_zero_one(fw)
		else:
			fw = np.ones(len(events))

		cut4 = np.ones(len(Photon)) * 4
		print('Fake fraction weight: ',len(fw),len(cut4),fw)
		out["cutflow"].fill(dataset=dataset, cutflow=cut4,weight=fw)



		##----------- Cut flow5: OSSF
		ossf_mask = Electron.charge[:,0] + Electron.charge[:,1] == 0
		

		# Apply cut 5
		Electron = Electron[ossf_mask]
		Photon = Photon[ossf_mask]
		fw = fw[ossf_mask]
		Jet = Jet[ossf_mask]
		MET = MET[ossf_mask]
		Muon = Muon[ossf_mask]
		if not isData:
			pu = pu[ossf_mask]
		events = events[ossf_mask]

		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		cut5 = np.ones(ak.sum(ak.num(Electron) > 0)) * 5
		out["cutflow"].fill(dataset=dataset, cutflow=cut5,weight=fw)

		# Define Electron Triplet
		Diele = ak.zip(
			{
				"lep1": Electron[:,0],
				"lep2": Electron[:,1],
				"p4": TLorentz_vector(Electron[:,0] + Electron[:,1]),
			}
		)
		
		
		leading_ele	= Diele.lep1
		subleading_ele = Diele.lep2
		
		def make_leading_pair(target, base):
			return target[ak.argmax(base.pt, axis=1, keepdims=True)]
		
		leading_pho = make_leading_pair(Photon, Photon)

		# -- Scale Factor for each electron
		
		# Trigger weight helper function
		def Trigger_Weight(eta1, pt1, eta2, pt2):
			per_ev_MC = (
				get_ele_trig_leg1_mc_Eff(eta1, pt1)
				* get_ele_trig_leg2_mc_Eff(eta2, pt2)
				+ get_ele_trig_leg1_mc_Eff(eta2, pt2)
				* get_ele_trig_leg2_mc_Eff(eta1, pt1)
				- get_ele_trig_leg1_mc_Eff(eta1, pt1)
				* get_ele_trig_leg1_mc_Eff(eta2, pt2)
			)

			per_ev_data = (
				get_ele_trig_leg1_data_Eff(eta1, pt1)
				* get_ele_trig_leg1_SF(eta1, pt1)
				* get_ele_trig_leg2_data_Eff(eta2, pt2)
				* get_ele_trig_leg2_SF(eta2, pt2)
				+ get_ele_trig_leg1_data_Eff(eta2, pt2)
				* get_ele_trig_leg1_SF(eta2, pt2)
				* get_ele_trig_leg2_data_Eff(eta1, pt1)
				* get_ele_trig_leg2_SF(eta1, pt1)
				- get_ele_trig_leg1_data_Eff(eta1, pt1)
				* get_ele_trig_leg1_SF(eta1, pt1)
				* get_ele_trig_leg1_data_Eff(eta2, pt2)
				* get_ele_trig_leg1_SF(eta2, pt2)
			)

			return per_ev_data / per_ev_MC

		if not isData:

			## -------------< Egamma ID and Reco Scale factor > -----------------##
			get_pho_medium_id_sf = get_pho_medium_id_sf(
				ak.flatten(leading_pho.eta), ak.flatten(leading_pho.pt)
			)

			ele_reco_sf = (
				get_ele_reco_above20_sf(
					leading_ele.deltaEtaSC + leading_ele.eta,
					leading_ele.pt,
				)
				* get_ele_reco_above20_sf(
					subleading_ele.deltaEtaSC + subleading_ele.eta,
					subleading_ele.pt,
				)
			)

			ele_medium_id_sf = (
				get_ele_medium_id_sf(
					leading_ele.deltaEtaSC + leading_ele.eta,
					leading_ele.pt,
				)
				* get_ele_medium_id_sf(
					subleading_ele.deltaEtaSC + subleading_ele.eta,
					subleading_ele.pt,
				)
			)

			print("#### Test Muon: ",abs(Muon))


			## -------------< Muon ID and Iso Scale factor > -----------------##
			get_mu_tight_id_sf = get_mu_tight_id_sf(
				ak.flatten(abs(Muon.eta)), ak.flatten(Muon.pt)
			)
			get_mu_tight_iso_sf = get_mu_tight_iso_sf(
				ak.flatten(abs(Muon.eta)), ak.flatten(Muon.pt)
			)

			## -------------< Double Electron Trigger Scale factor > -----------------##
			eta1 = leading_ele.deltaEtaSC + leading_ele.eta
			eta2 = subleading_ele.deltaEtaSC + subleading_ele.eta
			pt1 = leading_ele.pt
			pt2 = subleading_ele.pt

			# -- 2017,2016 are not applied yet
			if self._year == "2018":
				ele_trig_weight = Trigger_Weight(eta1, pt1, eta2, pt2)

		##----------- Cut flow6: Event selection

		# Mee cut
		Mee_cut_mask = Diele.p4.mass > 4

		# Lepton PT cuts
		Leppt_mask = ak.firsts((Diele.lep1.pt >= 25) & (Diele.lep2.pt >= 10) & (Muon.pt >= 25))

		# MET cuts
		MET_mask = MET.pt > 20  # Baseline
		
		# Assemble!!
		Event_sel_mask = Leppt_mask & MET_mask & Mee_cut_mask  # SR,CR

		# Apply cut6
		Diele_sel = Diele[Event_sel_mask]
		leading_pho_sel = leading_pho[Event_sel_mask]
		MET_sel = MET[Event_sel_mask]
		Muon_sel = Muon[Event_sel_mask]
		events = events[Event_sel_mask]
		fw = fw[Event_sel_mask]

		# Photon  EE and EB
		isEE_mask = leading_pho.isScEtaEE
		isEB_mask = leading_pho.isScEtaEB
		Pho_EE = leading_pho[isEE_mask & Event_sel_mask]
		Pho_EB = leading_pho[isEB_mask & Event_sel_mask]


		# Stop processing if there is no event remain
		if len(leading_pho_sel) == 0:
			return out

		cut6 = np.ones(ak.sum(ak.num(leading_pho_sel) > 0)) * 6
		out["cutflow"].fill(dataset=dataset, cutflow=cut6,weight=fw)

		## -------------------- Prepare making hist --------------#

		# Photon
		phoPT = ak.flatten(leading_pho_sel.pt)
		phoEta = ak.flatten(leading_pho_sel.eta)
		phoPhi = ak.flatten(leading_pho_sel.phi)

		# Photon EE
		if len(Pho_EE.pt) != 0:
			Pho_EE_PT = ak.flatten(Pho_EE.pt)
			Pho_EE_Eta = ak.flatten(Pho_EE.eta)
			Pho_EE_Phi = ak.flatten(Pho_EE.phi)
			Pho_EE_sieie = ak.flatten(Pho_EE.sieie)
			Pho_EE_hoe = ak.flatten(Pho_EE.hoe)
			Pho_EE_Iso_all = ak.flatten(Pho_EE.pfRelIso03_all)
			Pho_EE_Iso_charge = ak.flatten(Pho_EE.pfRelIso03_chg)

		# Photon EB
		if len(Pho_EB.pt) != 0:
			Pho_EB_PT = ak.flatten(Pho_EB.pt)
			Pho_EB_Eta = ak.flatten(Pho_EB.eta)
			Pho_EB_Phi = ak.flatten(Pho_EB.phi)
			Pho_EB_sieie = ak.flatten(Pho_EB.sieie)
			Pho_EB_hoe = ak.flatten(Pho_EB.hoe)
			Pho_EB_Iso_all = ak.flatten(Pho_EB.pfRelIso03_all)
			Pho_EB_Iso_charge = ak.flatten(Pho_EB.pfRelIso03_chg)

		# Electrons
		ele1PT  = Diele_sel.lep1.pt
		ele1Eta = Diele_sel.lep1.eta
		ele1Phi = Diele_sel.lep1.phi

		ele2PT  = Diele_sel.lep2.pt
		ele2Eta = Diele_sel.lep2.eta
		ele2Phi = Diele_sel.lep2.phi


		# Muon
		muPT  = ak.flatten(Muon_sel.pt)
		muEta = ak.flatten(Muon_sel.eta)
		muPhi = ak.flatten(Muon_sel.phi)


		# MET
		met = ak.to_numpy(MET_sel.pt)

		# M(eea) M(ee)
		diele = Diele_sel.p4
		eeg_vec = diele + leading_pho_sel
		Meea = ak.flatten(eeg_vec.mass)
		Mee = diele.mass

		# W MT (--> beta)
		MT = np.sqrt(
			2 * Muon_sel.pt * MET_sel.pt * (1 - np.cos(abs(MET_sel.delta_phi(Muon_sel))))
		)
		MT = np.array(ak.firsts(MT))

		# --- Apply weight and hist
		
		if isFake:
			weights = processor.Weights(len(cut6))
		else:
			weights = processor.Weights(len(cut5))
			

		# --- skim cut-weight
		if not isFake:
			def skim_weight(arr):
				mask1 = ~ak.is_none(arr)
				subarr = arr[mask1]
				mask2 = subarr != 0
				return ak.to_numpy(subarr[mask2])
		else:
			def skim_weight(arr):
				return arr


		if not isFake:
			cuts = Event_sel_mask
			cuts_pho_EE = ak.flatten(isEE_mask)
			cuts_pho_EB = ak.flatten(isEB_mask)

		if isFake:
			cuts = np.ones(len(Event_sel_mask))
			cuts_pho_EE = ak.flatten(isEE_mask & Event_sel_mask)
			cuts_pho_EB = ak.flatten(isEB_mask & Event_sel_mask)


		if isFake:
			weights.add("fake_fraction", fw)
			
		# Weight and SF here
		if not (isData | isFake):
			weights.add("pileup", pu)
			weights.add("ele_id", ele_medium_id_sf)
			weights.add("ele_reco", ele_reco_sf)
			weights.add("pho_id", get_pho_medium_id_sf)
			weights.add("mu_id",get_mu_tight_id_sf)
			weights.add("mu_iso",get_mu_tight_id_sf)



			# 2016,2017 are not applied yet
			if self._year == "2018":
				weights.add("ele_trigger", ele_trig_weight)

		# ---------------------------- Fill hist --------------------------------------#

		# Initial events
		out["sumw"][dataset] += len(Initial_events)




		print("cut1: {0},cut2: {1},cut3: {2},cut4: {3},cut5: {4},cut6: {5},cut7: {6}".format(len(cut0), len(cut1), len(cut2), len(cut3), len(cut4), len(cut5),len(cut6)))


		## Cut flow loop
		#for cut in [cut0, cut1, cut2, cut3, cut4, cut5,cut6]:
		#	out["cutflow"].fill(dataset=dataset, cutflow=cut)

		# Primary vertex
		out["nPV"].fill(
			dataset=dataset,
			nPV=nPV,
		)
		out["nPV_nw"].fill(dataset=dataset, nPV_nw=nPV_nw)

		# Fill hist


		# -- met -- #
		out["met"].fill(
			dataset=dataset, met=met, weight=skim_weight(weights.weight() * cuts)
		)

		# --mass -- #
		out["MT"].fill(
			dataset=dataset, MT=MT, weight=skim_weight(weights.weight() * cuts)
		)
		out["mass"].fill(
			dataset=dataset, mass=Mee, weight=skim_weight(weights.weight() * cuts)
		)
		out["mass_eea"].fill(
			dataset=dataset, mass_eea=Meea, weight=skim_weight(weights.weight() * cuts)
		)


		# -- Muon -- #
		out["mupt"].fill(
			dataset=dataset, mupt=muPT, weight=skim_weight(weights.weight() * cuts)
		)
		out["mueta"].fill(
			dataset=dataset,
			mueta=muEta,
			weight=skim_weight(weights.weight() * cuts),
		)
		out["muphi"].fill(
			dataset=dataset,
			muphi=muPhi,
			weight=skim_weight(weights.weight() * cuts),
		)


		# -- Electron -- #
		out["ele1pt"].fill(
			dataset=dataset, ele1pt=ele1PT, weight=skim_weight(weights.weight() * cuts)
		)
		out["ele1eta"].fill(
			dataset=dataset,
			ele1eta=ele1Eta,
			weight=skim_weight(weights.weight() * cuts),
		)
		out["ele1phi"].fill(
			dataset=dataset,
			ele1phi=ele1Phi,
			weight=skim_weight(weights.weight() * cuts),
		)
		out["ele2pt"].fill(
			dataset=dataset, ele2pt=ele2PT, weight=skim_weight(weights.weight() * cuts)
		)
		out["ele2eta"].fill(
			dataset=dataset,
			ele2eta=ele2Eta,
			weight=skim_weight(weights.weight() * cuts),
		)
		out["ele2phi"].fill(
			dataset=dataset,
			ele2phi=ele2Phi,
			weight=skim_weight(weights.weight() * cuts),
		)

		# -- Photon -- #

		out["phopt"].fill(
			dataset=dataset, phopt=phoPT, weight=skim_weight(weights.weight() * cuts)
		)
		out["phoeta"].fill(
			dataset=dataset, phoeta=phoEta, weight=skim_weight(weights.weight() * cuts)
		)
		out["phophi"].fill(
			dataset=dataset, phophi=phoPhi, weight=skim_weight(weights.weight() * cuts)
		)

		if len(Pho_EE.pt) != 0:

			out["pho_EE_pt"].fill(
				dataset=dataset,
				pho_EE_pt=Pho_EE_PT,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EE),
			)
			out["pho_EE_eta"].fill(
				dataset=dataset,
				pho_EE_eta=Pho_EE_Eta,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EE),
			)
			out["pho_EE_phi"].fill(
				dataset=dataset,
				pho_EE_phi=Pho_EE_Phi,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EE),
			)
			out["pho_EE_hoe"].fill(
				dataset=dataset,
				pho_EE_hoe=Pho_EE_hoe,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EE),
			)
			out["pho_EE_sieie"].fill(
				dataset=dataset,
				pho_EE_sieie=Pho_EE_sieie,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EE),
			)
			out["pho_EE_Iso_all"].fill(
				dataset=dataset,
				pho_EE_Iso_all=Pho_EE_Iso_all,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EE),
			)
			out["pho_EE_Iso_chg"].fill(
				dataset=dataset,
				pho_EE_Iso_chg=Pho_EE_Iso_charge,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EE),
			)

		if len(Pho_EB.pt) != 0:
			out["pho_EB_pt"].fill(
				dataset=dataset,
				pho_EB_pt=Pho_EB_PT,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EB),
			)
			out["pho_EB_eta"].fill(
				dataset=dataset,
				pho_EB_eta=Pho_EB_Eta,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EB),
			)
			out["pho_EB_phi"].fill(
				dataset=dataset,
				pho_EB_phi=Pho_EB_Phi,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EB),
			)
			out["pho_EB_hoe"].fill(
				dataset=dataset,
				pho_EB_hoe=Pho_EB_hoe,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EB),
			)
			out["pho_EB_sieie"].fill(
				dataset=dataset,
				pho_EB_sieie=Pho_EB_sieie,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EB),
			)
			out["pho_EB_Iso_all"].fill(
				dataset=dataset,
				pho_EB_Iso_all=Pho_EB_Iso_all,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EB),
			)
			out["pho_EB_Iso_chg"].fill(
				dataset=dataset,
				pho_EB_Iso_chg=Pho_EB_Iso_charge,
				weight=skim_weight(weights.weight() * cuts * cuts_pho_EB),
			)

		return out

	# -- Finally! return accumulator
	def postprocess(self, accumulator):

		return accumulator


# <---- Class JW_Processor


if __name__ == "__main__":

	start = time.time()
	parser = argparse.ArgumentParser()

	parser.add_argument("--nWorker", type=int, help=" --nWorker 2", default=8)
	parser.add_argument("--metadata", type=str, help="--metadata xxx.json")
	parser.add_argument(
		"--dataset", type=str, help="--dataset ex) Egamma_Run2018A_280000"
	)
	parser.add_argument("--year", type=str, help="--year 2018", default="2017")
	parser.add_argument("--isdata", type=bool, help="--isdata True", default=False)
	parser.add_argument("--isFake", type=bool, help="--isFake True", default=False)
	args = parser.parse_args()

	## Prepare files
	N_node = args.nWorker
	metadata = args.metadata
	data_sample = args.dataset
	year = args.year
	isdata = args.isdata
	isFake = args.isFake

	## Json file reader
	with open(metadata) as fin:
		datadict = json.load(fin)

	filelist = glob.glob(datadict[data_sample])

	if isFake:
		sample_name = "Fake_Photon"
	else:
		sample_name = data_sample.split("_")[0]

	corr_file = "../Corrections/corrections.coffea"
	# corr_file = "corrections.coffea" # Condor-batch

	corrections = load(corr_file)

	## Read PU weight file

	if not isdata:
		pu_path_dict = {
			"DY": "mcPileupDist_DYToEE_M-50_NNPDF31_TuneCP5_13TeV-powheg-pythia8.npy",
			"TTWJets": "mcPileupDist_TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8.npy",
			"TTZtoLL": "mcPileupDist_TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8.npy",
			"WW": "mcPileupDist_WW_TuneCP5_DoubleScattering_13TeV-pythia8.npy",
			"WZ": "mcPileupDist_WZ_TuneCP5_13TeV-pythia8.npy",
			"ZZ": "mcPileupDist_ZZ_TuneCP5_13TeV-pythia8.npy",
			"tZq": "mcPileupDist_tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8.npy",
			"WZG": "mcPileupDist_wza_UL18.npy",
			"ZGToLLG": "mcPileupDist_ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8.npy",
			"TTGJets": "mcPileupDist_TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8.npy",
			"WGToLNuG": "mcPileupDist_WGToLNuG_01J_5f_PtG_120_TuneCP5_13TeV-amcatnloFXFX-pythia8.npy",
		}

		if year == "2018":
			pu_path = (
				"../Corrections/Pileup/puWeight/npy_Run2018ABD_pure/"
				+ pu_path_dict[sample_name]
			)  # Local pure 2018
			# pu_path = '../Corrections/Pileup/puWeight/npy_Run2018ABD/'+ pu_path_dict[sample_name] # Local skim 2018

		if year == "2017":
			pu_path = (
				"../Corrections/Pileup/puWeight/npy_Run2017/"
				+ pu_path_dict[sample_name]
			)  # 2017

		print("Use the PU file: ", pu_path)
		with open(pu_path, "rb") as f:
			pu = np.load(f)

	else:
		pu = -1

	print("Processing the sample: ", sample_name)
	samples = {sample_name: filelist}

	# Class -> Object
	JW_Processor_instance = JW_Processor(year, sample_name, pu, corrections,isFake)

	## -->Multi-node Executor
	result = processor.run_uproot_job(
		samples,  # dataset
		"Events",  # Tree name
		JW_Processor_instance,  # Class
		executor=processor.futures_executor,
		executor_args={"schema": NanoAODSchema, "workers": 20},
		# maxchunks=4,
	)

	if isFake:
		outname =sample_name + "_" + data_sample + ".futures"
	else:
		outname = data_sample + ".futures"
	# outname = 'DY_test.futures'
	save(result, outname)

	elapsed_time = time.time() - start
	print("Time: ", elapsed_time)
