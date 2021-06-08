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
import numba

# -- Coffea 0.8.0 --> Must fix!!
import warnings

warnings.filterwarnings("ignore")


# ---> Class JW Processor
class JW_Processor(processor.ProcessorABC):

	# -- Initializer
	def __init__(self, year, sample_name):

		# Parameter set
		self._year = year

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

		# hist set
		self._accumulator = processor.dict_accumulator(
			{
				"sumw": processor.defaultdict_accumulator(float),
				"cutflow": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("cutflow", "Cut index", [0, 1, 2, 3, 4, 5, 6, 7]),
				),
				# -- Leading Electron  -- #
				"ele1pt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("ele1pt", "Leading Electron $P_{T}$ [GeV]", 300, 0, 600),
				),
				"ele1eta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("ele1eta", "Leading Electron $\eta$ [GeV]", 20, -5, 5),
				),
				"ele1phi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"ele1phi", "Leading Electron $\phi$ [GeV]", 20, -3.15, 3.15
					),
				),
				# -- Sub-leading Electron  -- #
				"ele2pt": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"ele2pt", "Subleading $Electron P_{T}$ [GeV]", 300, 0, 600
					),
				),
				"ele2eta": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("ele2eta", "Subleading Electron $\eta$ [GeV]", 20, -5, 5),
				),
				"ele2phi": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"ele2phi", "Subleading Electron $\phi$ [GeV]", 20, -3.15, 3.15
					),
				),
				# -- Photon --#
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
				# -- Photon EE -- #
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
				"pho_EE_sieie": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EE_sieie", "Photon EE sieie", 1000, 0, 0.1),
				),
				"pho_EE_Iso_chg": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"pho_EE_Iso_chg", "Photon EE pfReoIso03_charge", 200, 0, 1
					),
				),
				# -- Photon EB -- #
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
				"pho_EB_Iso_chg": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"pho_EB_Iso_chg", "Photon EB pfReoIso03_charge", 100, 0, 0.03
					),
				),
				"pho_EB_sieie": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("pho_EB_sieie", "Photon EB sieie", 200, 0, 0.1),
				),
				# -- Kinematic variables -- #
				"mass": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("Mee", "M(ee) [GeV]", 100, 0, 200),
				),
				"mass_eea": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("mass_eea", "$m_{e+e-\gamma}$ [GeV]", 300, 0, 600),
				),
				"met": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("met", "met [GeV]", 300, 0, 600),
				),
				"dR_ae1": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("dR_ae1", "dR(ae1)", 100, 0, 4),
				),
				"dR_ae2": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("dR_ae2", "$dR(ae2)$", 100, 0, 4),
				),
				"dR_aj": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("dR_aj", "dR(aj1)", 100, 0, 4),
				),
				# -- Sieie bins -- #
				"PT_1_eta_1": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("PT_1_eta_1", "20 < pt <30 & |eta| < 1", 200, 0, 0.02),
				),
				"PT_1_eta_2": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_1_eta_2", "20 < pt <30 & 1 < |eta| < 1.5", 200, 0, 0.02
					),
				),
				"PT_1_eta_3": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_1_eta_3", "20 < pt <30 & 1.5 < |eta| < 2", 200, 0, 0.05
					),
				),
				"PT_1_eta_4": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_1_eta_4", "20 < pt <30 & 2 < |eta| < 2.5", 200, 0, 0.05
					),
				),
				"PT_2_eta_1": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("PT_2_eta_1", "30 < pt <40 & |eta| < 1", 200, 0, 0.02),
				),
				"PT_2_eta_2": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_2_eta_2", "30 < pt <40 & 1 < |eta| < 1.5", 200, 0, 0.02
					),
				),
				"PT_2_eta_3": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_2_eta_3", "30 < pt <40 & 1.5 < |eta| < 2", 200, 0, 0.05
					),
				),
				"PT_2_eta_4": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_2_eta_4", "30 < pt <40 & 2 < |eta| < 2.5", 200, 0, 0.05
					),
				),
				"PT_3_eta_1": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("PT_3_eta_1", "40 < pt <50 & |eta| < 1", 200, 0, 0.02),
				),
				"PT_3_eta_2": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_3_eta_2", "40 < pt <50 & 1 < |eta| < 1.5", 200, 0, 0.02
					),
				),
				"PT_3_eta_3": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_3_eta_3", "40 < pt <50 & 1.5 < |eta| < 2", 200, 0, 0.05
					),
				),
				"PT_3_eta_4": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin(
						"PT_3_eta_4", "40 < pt <50 & 2 < |eta| < 2.5", 200, 0, 0.05
					),
				),
				"PT_4_eta_1": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("PT_4_eta_1", "50 < pt & |eta| < 1", 200, 0, 0.02),
				),
				"PT_4_eta_2": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("PT_4_eta_2", "50 <pt  & 1 < |eta| < 1.5", 200, 0, 0.02),
				),
				"PT_4_eta_3": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("PT_4_eta_3", "50 < pt  & 1.5 < |eta| < 2", 200, 0, 0.05),
				),
				"PT_4_eta_4": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("PT_4_eta_4", "50 < pt  & 2 < |eta| < 2.5", 200, 0, 0.05),
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

		# Stop processing if there is no event remain
		if len(events) == 0:
			return out

		# Cut flow
		cut0 = np.zeros(len(events))

		# <----- Helper functions ------>#

		# Sort by PT helper function
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

		# Cut-based ID modification
		@numba.njit
		def PhotonVID(vid, idBit):
			rBit = 0
			for x in range(0, 7):
				rBit |= (1 << x) if ((vid >> (x * 2)) & 0b11 >= idBit) else 0
			return rBit

		# Inverse Sieie and upper limit
		@numba.njit
		def make_fake_obj_mask(Pho, builder):

			# for eventIdx,pho in enumerate(tqdm(Pho)):   # --Event Loop
			for eventIdx, pho in enumerate(Pho):
				builder.begin_list()
				if len(pho) < 1:
					continue

				for phoIdx, _ in enumerate(pho):  # --Photon Loop

					vid = Pho[eventIdx][phoIdx].vidNestedWPBitmap
					vid_cuts1 = PhotonVID(vid, 1)  # Loose photon
					vid_cuts2 = PhotonVID(vid, 2)  # Medium photon
					vid_cuts3 = PhotonVID(vid, 3)  # Tight photon

					# Field name
					# |0|0|0|0|0|0|0|
					# |IsoPho|IsoNeu|IsoChg|Sieie|hoe|scEta|PT|

					# 1. Turn off cut (ex turn off Sieie
					# |1|1|1|0|1|1|1| = |1|1|1|0|1|1|1|

					# 2. Inverse cut (ex inverse Sieie)
					# |1|1|1|1|1|1|1| = |1|1|1|0|1|1|1|

					# if (vid_cuts2 & 0b1111111 == 0b1111111): # Cut applied
					#if vid_cuts2 & 0b1111111 == 0b1110111:  # Inverse Sieie
					if (vid_cuts2 & 0b1100111 == 0b1100111): # Without Sieie and IsoChg
						isochg = (
							Pho[eventIdx][phoIdx].pfRelIso03_chg
							* Pho[eventIdx][phoIdx].pt
						)
						if (isochg >= 4) & (isochg <= 10):
							builder.boolean(True)
						else:
							builder.boolean(False)
						

					else:

						builder.boolean(False)

				builder.end_list()

			return builder


		# Golden Json file
		if self._year == "2018":
			injson = "/x5/cms/jwkim/gitdir/JWCorp/JW_analysis/Coffea_WZG/Corrections/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt.RunABD"

		if self._year == "2017":
			injson = "/x5/cms/jwkim/gitdir/JWCorp/JW_analysis/Coffea_WZG/Corrections/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"

		# --- Selection
		Initial_events = events
		# Good Run ( Golden Json files )
		from coffea import lumi_tools

		lumi_mask_builder = lumi_tools.LumiMask(injson)
		lumimask = ak.Array(
			lumi_mask_builder.__call__(events.run, events.luminosityBlock)
		)
		events = events[lumimask]
		# print("{0}%  of files pass good-run conditions".format(len(events)/ len(Initial_events)))

		# Stop processing if there is no event remain
		if len(events) == 0:
			return out

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

		# Apply cut1
		Initial_events = events
		# events = events[single_ele_triggers_arr | double_ele_triggers_arr]
		events = events[double_ele_triggers_arr]

		cut1 = np.ones(len(events))

		# Set Particles
		Electron = events.Electron
		Muon = events.Muon
		Photon = events.Photon
		MET = events.MET
		Jet = events.Jet

		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		#  --Muon ( only used to calculate dR )
		MuSelmask = (
			(Muon.pt >= 10)
			& (abs(Muon.eta) <= 2.5)
			& (Muon.tightId)
			& (Muon.pfRelIso04_all < 0.15)
		)
		# Muon = ak.mask(Muon,MuSelmask)
		Muon = Muon[MuSelmask]


		##----------- Cut flow2: Electron Selection

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

		# apply cut 2
		Tri_electron_mask = ak.num(Electron) >= 2
		Electron = Electron[Tri_electron_mask]
		Photon = Photon[Tri_electron_mask]
		Jet = Jet[Tri_electron_mask]
		MET = MET[Tri_electron_mask]
		Muon = Muon[Tri_electron_mask]
		events = events[Tri_electron_mask]

		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		cut2 = np.ones(len(Photon)) * 2


		##----------- Cut flow3: Photon Selection

		# Basic photon selection
		isgap_mask = (abs(Photon.eta) < 1.442) | (
			(abs(Photon.eta) > 1.566) & (abs(Photon.eta) < 2.5)
		)
		Pixel_seed_mask = ~Photon.pixelSeed
		PT_mask = Photon.pt >= 20

		# dR cut with selected Muon and Electrons
		dr_pho_ele_mask = ak.all(
			Photon.metric_table(Electron) >= 0.5, axis=-1
		)  # default metric table: delta_r
		dr_pho_mu_mask = ak.all(Photon.metric_table(Muon) >= 0.5, axis=-1)


		PhoSelmask = (
			PT_mask
			& isgap_mask
			& Pixel_seed_mask
			& dr_pho_ele_mask
			& dr_pho_mu_mask
		)
		Photon = Photon[PhoSelmask]

		# Apply cut 3
		A_photon_mask = ak.num(Photon) > 0
		Electron = Electron[A_photon_mask]
		Photon = Photon[A_photon_mask]
		Jet = Jet[A_photon_mask]
		Muon = Muon[A_photon_mask]
		MET = MET[A_photon_mask]
		events = events[A_photon_mask]

		# ID for fake photon
		Photon_template_mask = make_fake_obj_mask(Photon, ak.ArrayBuilder()).snapshot()

		Photon = Photon[Photon_template_mask]
		# Apply cut -Fake Photon -
		A_photon_mask = ak.num(Photon) > 0
		Electron = Electron[A_photon_mask]
		Photon = Photon[A_photon_mask]
		Jet = Jet[A_photon_mask]
		Muon = Muon[A_photon_mask]
		MET = MET[A_photon_mask]
		events = events[A_photon_mask]


		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		cut3 = np.ones(len(Photon)) *  3

		##-----------  Cut flow4:  Select 2 OSSF electrons from Z
		@numba.njit
		def find_2lep(events_leptons, builder):
			for leptons in events_leptons:

				builder.begin_list()
				nlep = len(leptons)
				for i0 in range(nlep):
					for i1 in range(i0 + 1, nlep):
						if leptons[i0].charge + leptons[i1].charge != 0:
							continue

						if nlep == 2:
							builder.begin_tuple(2)
							builder.index(0).integer(i0)
							builder.index(1).integer(i1)
							builder.end_tuple()

						else:
							for i2 in range(nlep):
								if len({i0, i1, i2}) < 3:
									continue
								builder.begin_tuple(3)
								builder.index(0).integer(i0)
								builder.index(1).integer(i1)
								builder.index(2).integer(i2)
								builder.end_tuple()
				builder.end_list()
			return builder

		ossf_idx = find_2lep(Electron, ak.ArrayBuilder()).snapshot()

		# OSSF cut
		ossf_mask = ak.num(ossf_idx) >= 1
		ossf_idx = ossf_idx[ossf_mask]
		Electron = Electron[ossf_mask]
		Photon = Photon[ossf_mask]
		Jet = Jet[ossf_mask]
		MET = MET[ossf_mask]

		Double_electron = [Electron[ossf_idx[idx]] for idx in "01"]
		from coffea.nanoevents.methods import vector

		ak.behavior.update(vector.behavior)

		Diele = ak.zip(
			{
				"lep1": Double_electron[0],
				"lep2": Double_electron[1],
				"p4": TLorentz_vector(Double_electron[0] + Double_electron[1]),
			}
		)

		bestZ_idx = ak.singletons(ak.argmin(abs(Diele.p4.mass - 91.1876), axis=1))
		Diele = Diele[bestZ_idx]

		cut4 = np.ones(len(Electron)) * 4

		##-----------  Cut flow 5: Event Selection

		def make_leading_pair(target, base):
			return target[ak.argmax(base.pt, axis=1, keepdims=True)]

		leading_pho = make_leading_pair(Photon, Photon)

		# Mee cut
		Mee_cut_mask = ak.firsts(Diele.p4.mass) > 4

		# Electron PT cuts
		Elept_mask = ak.firsts((Diele.lep1.pt >= 25) & (Diele.lep2.pt >= 20))

		# MET cuts
		MET_mask = MET.pt > 20

		# --------Mask -------#
		Event_sel_mask = Mee_cut_mask & Elept_mask & MET_mask
		Diele_sel = Diele[Event_sel_mask]
		leading_pho_sel = leading_pho[Event_sel_mask]
		Jet_sel = Jet[Event_sel_mask]
		MET_sel = MET[Event_sel_mask]

		cut5 = np.ones(len(Diele)) * 5

		# Photon  EE and EB
		isEE_mask = leading_pho.isScEtaEE
		isEB_mask = leading_pho.isScEtaEB
		Pho_EE = leading_pho[isEE_mask & Event_sel_mask]
		Pho_EB = leading_pho[isEB_mask & Event_sel_mask]

		# -------------------- Flatten variables ---------------------------#

		# -- Ele1 --#
		Ele1_PT = ak.flatten(Diele_sel.lep1.pt)
		Ele1_Eta = ak.flatten(Diele_sel.lep1.eta)
		Ele1_Phi = ak.flatten(Diele_sel.lep1.phi)

		# -- Ele2 --#
		Ele2_PT = ak.flatten(Diele_sel.lep2.pt)
		Ele2_Eta = ak.flatten(Diele_sel.lep2.eta)
		Ele2_Phi = ak.flatten(Diele_sel.lep2.phi)

		# -- Pho -- #
		Pho_PT = ak.flatten(leading_pho_sel.pt)
		Pho_Eta = ak.flatten(leading_pho_sel.eta)
		Pho_Phi = ak.flatten(leading_pho_sel.phi)


		# -- Pho EB --#
		Pho_EB_PT = ak.flatten(Pho_EB.pt)
		Pho_EB_Eta = ak.flatten(Pho_EB.eta)
		Pho_EB_Phi = ak.flatten(Pho_EB.phi)
		Pho_EB_Isochg = ak.flatten(Pho_EE.pfRelIso03_chg)
		Pho_EB_Sieie = ak.flatten(Pho_EE.sieie)

		# -- Pho EE --#
		Pho_EE_PT = ak.flatten(Pho_EE.pt)
		Pho_EE_Eta = ak.flatten(Pho_EE.eta)
		Pho_EE_Phi = ak.flatten(Pho_EE.phi)
		Pho_EE_Isochg = ak.flatten(Pho_EE.pfRelIso03_chg)
		Pho_EE_Sieie = ak.flatten(Pho_EE.sieie)

		# --Kinematics --#
		Diele_mass = ak.flatten(Diele_sel.p4.mass)
		eeg_vec = Diele_sel.p4 + leading_pho_sel
		eeg_mass = ak.flatten(eeg_vec.mass)

		leading_ele, subleading_ele = ak.flatten(
			TLorentz_vector_cylinder(Diele_sel.lep1)
		), ak.flatten(TLorentz_vector_cylinder(Diele_sel.lep2))
		dR_e1pho = ak.flatten(leading_ele.delta_r(leading_pho_sel))  # dR pho,ele1
		dR_e2pho = ak.flatten(subleading_ele.delta_r(leading_pho_sel))  # dR pho,ele2
		dR_jpho = ak.flatten(Jet_sel[:, 0].delta_r(leading_pho_sel))

		MET_PT = ak.to_numpy(MET_sel.pt)

		# -------------------- Sieie bins---------------------------#
		def make_bins(pt, eta, sieie, bin_range_str):

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

			return ak.to_numpy(sieie[binmask])

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

		binned_sieie_hist = {}
		for name in bin_name_list:
			binned_sieie_hist[name] = make_bins(
				ak.flatten(leading_pho_sel.pt),
				ak.flatten(abs(leading_pho_sel.eta)),
				ak.flatten(leading_pho_sel.sieie),
				name,
			)

		# -------------------- Fill hist ---------------------------#

		# Initial events
		out["sumw"][dataset] += len(Initial_events)

		# print("cut5: ",len(cut5))

		# Cut flow loop
		for cut in [cut0, cut1, cut2, cut3, cut4, cut5]:
			out["cutflow"].fill(dataset=dataset, cutflow=cut)

		# --Ele1 -- #
		out["ele1pt"].fill(dataset=dataset, ele1pt=Ele1_PT)
		out["ele1eta"].fill(dataset=dataset, ele1eta=Ele1_Eta)
		out["ele1phi"].fill(dataset=dataset, ele1phi=Ele1_Phi)

		# --Ele2 -- #
		out["ele2pt"].fill(dataset=dataset, ele2pt=Ele2_PT)
		out["ele2eta"].fill(dataset=dataset, ele2eta=Ele2_Eta)
		out["ele2phi"].fill(dataset=dataset, ele2phi=Ele2_Phi)

		# --Photon-- #

		out["phopt"].fill(dataset=dataset, phopt=Pho_PT)

		out["phoeta"].fill(dataset=dataset, phoeta=Pho_Eta)
		out["phophi"].fill(dataset=dataset, phophi=Pho_Phi)

		# --Photon EB --#
		out["pho_EB_pt"].fill(
			dataset=dataset,
			pho_EB_pt=Pho_EB_PT,
		)
		out["pho_EB_eta"].fill(
			dataset=dataset,
			pho_EB_eta=Pho_EB_Eta,
		)
		out["pho_EB_phi"].fill(
			dataset=dataset,
			pho_EB_phi=Pho_EB_Phi,
		)
		out["pho_EB_sieie"].fill(
			dataset=dataset,
			pho_EB_sieie=Pho_EB_Sieie,
		)
		out["pho_EB_Iso_chg"].fill(dataset=dataset, pho_EB_Iso_chg=Pho_EB_Isochg)

		# --Photon EE --#
		out["pho_EE_pt"].fill(
			dataset=dataset,
			pho_EE_pt=Pho_EE_PT,
		)
		out["pho_EE_eta"].fill(
			dataset=dataset,
			pho_EE_eta=Pho_EE_Eta,
		)
		out["pho_EE_phi"].fill(
			dataset=dataset,
			pho_EE_phi=Pho_EE_Phi,
		)
		out["pho_EE_sieie"].fill(
			dataset=dataset,
			pho_EE_sieie=Pho_EE_Sieie,
		)
		out["pho_EE_Iso_chg"].fill(dataset=dataset, pho_EE_Iso_chg=Pho_EE_Isochg)


		# -- Kinematic variables -- #
		out["mass"].fill(dataset=dataset, Mee=Diele_mass)
		out["mass_eea"].fill(dataset=dataset, mass_eea=eeg_mass)
		out["met"].fill(dataset=dataset, met=MET_PT)
		out["dR_ae1"].fill(dataset=dataset, dR_ae1=dR_e1pho)
		out["dR_ae2"].fill(dataset=dataset, dR_ae2=dR_e2pho)
		out["dR_aj"].fill(dataset=dataset, dR_aj=dR_jpho)

		# test_target = binned_sieie_hist['PT_1_eta_1']
		# print("CheckAAA: ",test_target[test_target > 0.05])
		# print("CheckBBB: ",test_target[test_target > 0])

		# -- Binned sieie hist -- #

		if len(binned_sieie_hist["PT_1_eta_1"] > 0):
			out["PT_1_eta_1"].fill(
				dataset=dataset, PT_1_eta_1=binned_sieie_hist["PT_1_eta_1"]
			)
		if len(binned_sieie_hist["PT_1_eta_2"] > 0):
			out["PT_1_eta_2"].fill(
				dataset=dataset, PT_1_eta_2=binned_sieie_hist["PT_1_eta_2"]
			)
		if len(binned_sieie_hist["PT_1_eta_3"] > 0):
			out["PT_1_eta_3"].fill(
				dataset=dataset, PT_1_eta_3=binned_sieie_hist["PT_1_eta_3"]
			)
		if len(binned_sieie_hist["PT_1_eta_4"] > 0):
			out["PT_1_eta_4"].fill(
				dataset=dataset, PT_1_eta_4=binned_sieie_hist["PT_1_eta_4"]
			)
		if len(binned_sieie_hist["PT_2_eta_1"] > 0):
			out["PT_2_eta_1"].fill(
				dataset=dataset, PT_2_eta_1=binned_sieie_hist["PT_2_eta_1"]
			)
		if len(binned_sieie_hist["PT_2_eta_2"] > 0):
			out["PT_2_eta_2"].fill(
				dataset=dataset, PT_2_eta_2=binned_sieie_hist["PT_2_eta_2"]
			)
		if len(binned_sieie_hist["PT_2_eta_3"] > 0):
			out["PT_2_eta_3"].fill(
				dataset=dataset, PT_2_eta_3=binned_sieie_hist["PT_2_eta_3"]
			)
		if len(binned_sieie_hist["PT_2_eta_4"] > 0):
			out["PT_2_eta_4"].fill(
				dataset=dataset, PT_2_eta_4=binned_sieie_hist["PT_2_eta_4"]
			)
		if len(binned_sieie_hist["PT_3_eta_1"] > 0):
			out["PT_3_eta_1"].fill(
				dataset=dataset, PT_3_eta_1=binned_sieie_hist["PT_3_eta_1"]
			)
		if len(binned_sieie_hist["PT_3_eta_2"] > 0):
			out["PT_3_eta_2"].fill(
				dataset=dataset, PT_3_eta_2=binned_sieie_hist["PT_3_eta_2"]
			)
		if len(binned_sieie_hist["PT_3_eta_3"] > 0):
			out["PT_3_eta_3"].fill(
				dataset=dataset, PT_3_eta_3=binned_sieie_hist["PT_3_eta_3"]
			)
		if len(binned_sieie_hist["PT_3_eta_4"] > 0):
			out["PT_3_eta_4"].fill(
				dataset=dataset, PT_3_eta_4=binned_sieie_hist["PT_3_eta_4"]
			)
		if len(binned_sieie_hist["PT_4_eta_1"] > 0):
			out["PT_4_eta_1"].fill(
				dataset=dataset, PT_4_eta_1=binned_sieie_hist["PT_4_eta_1"]
			)
		if len(binned_sieie_hist["PT_4_eta_2"] > 0):
			out["PT_4_eta_2"].fill(
				dataset=dataset, PT_4_eta_2=binned_sieie_hist["PT_4_eta_2"]
			)
		if len(binned_sieie_hist["PT_4_eta_3"] > 0):
			out["PT_4_eta_3"].fill(
				dataset=dataset, PT_4_eta_3=binned_sieie_hist["PT_4_eta_3"]
			)
		if len(binned_sieie_hist["PT_4_eta_4"] > 0):
			#print("## show me the last bin: ", binned_sieie_hist["PT_4_eta_4"])
			#print("## show me the first bin: ", binned_sieie_hist["PT_1_eta_1"])
			out["PT_4_eta_4"].fill(
				dataset=dataset, PT_4_eta_4=binned_sieie_hist["PT_4_eta_4"]
			)

		return out

	# -- Finally! return accumulator
	def postprocess(self, accumulator):
		return accumulator


# <---- Class JW_Processor


if __name__ == "__main__":

	start = time.time()
	parser = argparse.ArgumentParser()

	parser.add_argument("--nWorker", type=int, help=" --nWorker 2", default=20)
	parser.add_argument("--metadata", type=str, help="--metadata xxx.json")
	parser.add_argument(
		"--dataset", type=str, help="--dataset ex) Egamma_Run2018A_280000"
	)
	parser.add_argument("--year", type=str, help="--year 2018", default="2017")
	parser.add_argument("--isdata", type=bool, help="--isdata False", default=False)
	args = parser.parse_args()

	## Prepare files
	N_node = args.nWorker
	metadata = args.metadata
	data_sample = args.dataset
	year = args.year

	## Json file reader
	with open(metadata) as fin:
		datadict = json.load(fin)

	filelist = glob.glob(datadict[data_sample])
	sample_name = "Fake_template"
	# sample_name = data_sample.split('_')[0]

	print(sample_name)
	samples = {sample_name: filelist}

	# Class -> Object
	JW_Processor_instance = JW_Processor(year, sample_name)

	## -->Multi-node Executor
	result = processor.run_uproot_job(
		samples,  # dataset
		"Events",  # Tree name
		JW_Processor_instance,  # Class
		executor=processor.futures_executor,
		executor_args={"schema": NanoAODSchema, "workers": 20},
		# maxchunks=4,
	)

	# outname = data_sample + '.futures'
	outname = sample_name + "_" + data_sample + ".futures"
	save(result, outname)

	elapsed_time = time.time() - start
