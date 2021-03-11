import awkward1 as ak
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

# ---> Class JW Processor
class JW_Processor(processor.ProcessorABC):

	# -- Initializer
	def __init__(self,year,sample_name,xsec,puweight_arr,corrections):


		lumis = { #Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable													  
		#'2016': 35.92,
		#'2017': 41.53,
		'2018': 21.1
		}	

		

		# Parameter set
		self._year = year
		self._lumi = lumis[self._year] * 1000

		self._xsec = xsec
		
		

		# Trigger set
		self._doubleelectron_triggers  ={
			'2018': [
					"Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", # Recomended
					]
		}
	
	
	
		self._singleelectron_triggers = { #2017 and 2018 from monojet, applying dedicated trigger weights
				'2016': [
					'Ele27_WPTight_Gsf',
					'Ele105_CaloIdVT_GsfTrkIdT'
				],
				'2017': [
					'Ele35_WPTight_Gsf',
					'Ele115_CaloIdVT_GsfTrkIdT',
					'Photon200'
				],
				'2018': [
					'Ele32_WPTight_Gsf',   # Recomended
				]
			}
		

		# Corrrection set 
		self._corrections = corrections
		self._puweight_arr = puweight_arr

		



		# hist set
		self._accumulator = processor.dict_accumulator({

			"sumw": processor.defaultdict_accumulator(float),
	
			'cutflow': hist.Hist(
				'Events',
				hist.Cat('dataset', 'Dataset'),
				hist.Bin('cutflow', 'Cut index', [0, 1, 2, 3, 4,5,6])
			),

			"nPV": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("nPV","Number of Primary vertex",100, 0, 100),
			),
			"nPV_nw": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("nPV_nw","Number of Primary vertex",100, 0, 100),
			),

			"mass": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("mass","$m_{e+e-}$ [GeV]", 100, 0, 200),
			),
			"charge": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("charge","charge sum of electrons", 6, -3, 3),
			),
			"mass": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("mass","$m_{e+e-}$ [GeV]", 100, 0, 200),
			),
			"met": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("met","met [GeV]", 300, 0, 600),
			),
			"ele1pt": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele1pt","Leading Electron $P_{T}$ [GeV]", 300, 0, 600),
			),

			"ele2pt": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele2pt","Subleading $Electron P_{T}$ [GeV]", 300, 0, 600),
			),
			"ele3pt": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele3pt","Third $Electron P_{T}$ [GeV]", 300, 0, 600),
			),
			"ele1eta": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele1eta","Leading Electron $\eta$ [GeV]", 20, -5, 5),
			),

			"ele2eta": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele2eta","Subleading Electron $\eta$ [GeV]", 20, -5, 5),
			),
			"ele1phi": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele1phi","Leading Electron $\phi$ [GeV]", 20, -3.15, 3.15),
			),

			"ele2phi": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele2phi","Subleading Electron $\phi$ [GeV]", 20, -3.15, 3.15),
			),

			"pho_EE_pt": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_pt","Photon EE $P_{T}$ [GeV]", 300, 0, 600),
			),
			"pho_EE_eta": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_eta","Photon EE $\eta$ ", 50, -5, 5),
			),
			"pho_EE_phi": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_phi","Photon EE $\phi$ ", 50, -3.15, 3.15),
			),
			"pho_EE_hoe": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_hoe","Photon EE HoverE", 100, 0, 0.6),
			),

			"pho_EE_sieie": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_sieie","Photon EE sieie", 100, 0, 0.03),
			),
			"pho_EE_Iso_all": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_Iso_all","Photon EE pfReoIso03_all", 100, 0, 0.3),
			),
			"pho_EE_Iso_chg": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_Iso_chg","Photon EE pfReoIso03_charge", 200, 0, 1),
			),
			"pho_EB_pt": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_pt","Photon EB $P_{T}$ [GeV]", 300, 0, 600),
			),
			"pho_EB_eta": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_eta","Photon EB $\eta$ ", 50, -5, 5),
			),
			"pho_EB_phi": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_phi","Photon EB $\phi$ ", 50, -3.15, 3.15),
			),
			"pho_EB_hoe": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_hoe","Photon EB HoverE", 100, 0, 0.6),
			),

			"pho_EB_sieie": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_sieie","Photon EB sieie", 100, 0, 0.012),
			),
			"pho_EB_Iso_all": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_Iso_all","Photon EB pfReoIso03_all", 100, 0, 0.15),
			),
			"pho_EB_Iso_chg": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_Iso_chg","Photon EB pfReoIso03_charge", 100, 0, 0.03),
			)

			})
		
	# -- Accumulator: accumulate histograms
	@property
	def accumulator(self):
		return self._accumulator

	# -- Main function : Process events
	def process(self, events):

		# Initialize accumulator
		out = self.accumulator.identity()
		dataset = sample_name
		#events.metadata['dataset']
		

		isData = 'genWeight' not in events.fields
		

		selection = processor.PackedSelection()

		# Cut flow
		cut0 = np.zeros(len(events))
		

		# --- Selection

		# << flat dim helper function >>
		def flat_dim(arr):
			
		#	if len(ak.to_numpy(arr).shape) > 1:
		#		sub_arr = ak.flatten(arr)
		#	else:
		#		sub_arr = arr
			sub_arr = ak.flatten(arr)
			mask = ~ak.is_none(sub_arr)

			return ak.to_numpy(sub_arr[mask])
		# << drop na helper function >>
		def drop_na(arr):

			mask = ~ak.is_none(arr)

			return arr[mask]
		# << drop na helper function >>
		def drop_na_np(arr):

			mask = ~np.isnan(arr)

			return arr[mask]
		# << Sort by PT  helper function >>
		def sort_by_pt(ele,pho,jet):
			ele = ele[ak.argsort(ele.pt,ascending=False,axis=1)]
			pho = pho[ak.argsort(pho.pt,ascending=False,axis=1)]
			jet = jet[ak.argsort(jet.pt,ascending=False,axis=1)]

			return ele,pho,jet
		


		# double lepton trigger
		is_double_ele_trigger=True
		if not is_double_ele_trigger:
			double_ele_triggers_arr=np.ones(len(events), dtype=np.bool)
		else:
			double_ele_triggers_arr = np.zeros(len(events), dtype=np.bool)
			for path in self._doubleelectron_triggers[self._year]:
				if path not in events.HLT.fields: continue
				double_ele_triggers_arr = double_ele_triggers_arr | events.HLT[path]


		# single lepton trigger
		is_single_ele_trigger=True
		if not is_single_ele_trigger:
			single_ele_triggers_arr=np.ones(len(events), dtype=np.bool)
		else:
			single_ele_triggers_arr = np.zeros(len(events), dtype=np.bool)
			for path in self._singleelectron_triggers[self._year]:
				if path not in events.HLT.fields: continue
				single_ele_triggers_arr = single_ele_triggers_arr | events.HLT[path]

		# Sort particle order by PT
		events.Electron,events.Photon,events.Jet = sort_by_pt(events.Electron,events.Photon,events.Jet)
		
		Initial_events = events
		#events = events[single_ele_triggers_arr | double_ele_triggers_arr]
		events = events[double_ele_triggers_arr]
		
		##----------- Cut flow1: Passing Triggers
		cut1 = np.ones(len(events))
		# Particle Identification
		Electron = events.Electron
		Photon = events.Photon
		Jet = events.Jet
		MET = events.MET



		def Particle_selection(ele,pho,jet):
			# Electron selection
			EleSelmask = (ele.pt > 25) & (np.abs(ele.eta) < 2.5) & (ele.cutBased > 2)
			
			# Photon selection
			PhoSelmask = (pho.pt > 25) & (pho.cutBased > 1)

			# Jet selection for dR cut
			bJet_selmask = (jet.btagCMVA > -0.5844)
			
			return EleSelmask,PhoSelmask,bJet_selmask

		# Event Selection
		Electron_mask, Photon_mask, bjet_selmask  = Particle_selection(Electron,Photon,Jet)
		Ele_channel_mask = ak.num(Electron[Electron_mask])  == 3 
		Pho_channel_mask = ak.num(Photon[Photon_mask]) > 0
		bJet_veto_mask = ak.num(Jet[bjet_selmask]) == 0

		Ele_channel_events = events[Ele_channel_mask & Pho_channel_mask &  bJet_veto_mask]
		
		

		##-----------  Cut flow2: Electron channel with one photon ( ID cut )
		cut2 = np.ones(len(Ele_channel_events)) * 2
		
		# --- Get scale factors
		if not isData:
	
			# Egamma reco ID
			get_ele_reco_sf		 = self._corrections['get_ele_reco_sf'][self._year]
			get_ele_medium_id_sf = self._corrections['get_ele_medium_id_sf'][self._year]
			get_pho_medium_id_sf = self._corrections['get_pho_medium_id_sf'][self._year]
	
			

			# DoubleEG trigger
			get_ele_trig_leg1_SF		= self._corrections['get_ele_trig_leg1_SF'][self._year]
			get_ele_trig_leg1_data_Eff	= self._corrections['get_ele_trig_leg1_data_Eff'][self._year]
			get_ele_trig_leg1_mc_Eff	= self._corrections['get_ele_trig_leg1_mc_Eff'][self._year]
			get_ele_trig_leg2_SF		= self._corrections['get_ele_trig_leg2_SF'][self._year]
			get_ele_trig_leg2_data_Eff  = self._corrections['get_ele_trig_leg2_data_Eff'][self._year]
			get_ele_trig_leg2_mc_Eff	= self._corrections['get_ele_trig_leg2_mc_Eff'][self._year]
			
			# PU weight with custom made npy and multi-indexing
			pu_weight_idx = ak.values_astype(Ele_channel_events.Pileup.nTrueInt,"int64")
			pu = self._puweight_arr[pu_weight_idx]
			


		# Particle array
		Electron = Ele_channel_events.Electron
		Photon   = Ele_channel_events.Photon
		Jet      = Ele_channel_events.Jet
		nPV		 = Ele_channel_events.PV.npvsGood
		MET		 = Ele_channel_events.MET

		Electron_mask,Photon_mask,_ = Particle_selection(Electron,Photon,Jet)	
		Electron = Electron[Electron_mask]	
		Photon   = Photon[Photon_mask]

		# OSSF
		@numba.njit
		def find_3lep(events_leptons,builder):
		    
		    for leptons in events_leptons:
		        builder.begin_list()    
		        nlep = len(leptons)
		        
		        for i0 in range(nlep):
		            for i1 in range(i0+1,nlep):
		                if leptons[i0].charge + leptons[i1].charge != 0: continue;
		                
		                for i2 in range(nlep):
		                    if len({i0,i1,i2}) < 3: continue;
		                    builder.begin_tuple(3)
		                    builder.index(0).integer(i0)
		                    builder.index(1).integer(i1)
		                    builder.index(2).integer(i2)
		                    builder.end_tuple()
		        builder.end_list()
		    return builder

		eee_triplet_idx = find_3lep(Electron,ak.ArrayBuilder()).snapshot()
		ossf_mask = ak.num(eee_triplet_idx) == 2
		eee_triplet_idx = eee_triplet_idx[ossf_mask]


		Electron= Electron[ossf_mask]
		Photon= Photon[ossf_mask]
		Jet= Jet[ossf_mask]
		MET = MET[ossf_mask]
		if not isData: pu = pu[ossf_mask]


		# Stop processing if there is no event remain
		if len(Electron) == 0:
			return out

		Triple_electron = [Electron[eee_triplet_idx[idx]] for idx in "012"]


		##-----------  Cut flow3: OSSF
		cut3 = np.ones(ak.sum(ak.num(Electron) > 0)) * 3


		from coffea.nanoevents.methods import vector
		ak.behavior.update(vector.behavior)
		
		def TLorentz_vector(vec):
		    vec = ak.zip(
		    {
		            "x":vec.x,
		            "y":vec.y,
		            "z":vec.z,
		            "t":vec.t
		    },
		    with_name = "LorentzVector"
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


		
		Triple_eee = ak.zip({"lep1":Triple_electron[0],
		                     "lep2":Triple_electron[1],
		                   "lep3":Triple_electron[2],
		                   "p4":TLorentz_vector(Triple_electron[0]+Triple_electron[1])})


		bestZ_idx = ak.singletons(ak.argmin(abs(Triple_eee.p4.mass - 91.1876), axis=1))
		Triple_eee = Triple_eee[bestZ_idx]
		
		leading_ele, subleading_ele = ak.flatten(TLorentz_vector_cylinder(Triple_eee.lep1)),ak.flatten(TLorentz_vector_cylinder(Triple_eee.lep2))

		# dR cut
		def make_DR(ele1,ele2,pho,jet):
		      
		    dR_e1pho  = ele1.delta_r(pho)
		    dR_e2pho  = ele2.delta_r(pho)
		    dR_phojet = jet[:,0].delta_r(pho)
		    dR_mask   = (dR_e1pho > 0.4) & (dR_e2pho > 0.4) & (dR_phojet > 0.4)
		    
		    return dR_mask,dR_e1pho,dR_e2pho,dR_phojet

		dR_mask,dR_e1pho,dR_e2pho,dR_phojet = make_DR(leading_ele,subleading_ele,Photon,Jet)
		Photon = Photon[dR_mask]
		
		
		Ele_channel_with_dR = Ele_channel_events[ak.num(Photon) > 0]
		Triple_eee = Triple_eee[ak.num(Photon) > 0]
		Jet      = Jet[ak.num(Photon) > 0]
		nPV = nPV[ak.num(Photon) > 0]
		if not isData: pu = pu[ak.num(Photon) > 0]
		MET = MET[ak.num(Photon) > 0]

		Photon   = Photon[ak.num(Photon) > 0] # Beware the order! Photon must be located last!


		


		#Stop processing if there is no event remain
		if len(Photon) == 0:
			return out

		##-----------  Cut flow4: Photon dR
		cut4 = np.ones(ak.sum(ak.num(Photon) > 0)) * 4


		leading_ele		= Triple_eee.lep1
		subleading_ele  = Triple_eee.lep2
		
		def make_leading_pair(target,base):
			return target[ak.argmax(base.pt,axis=1,keepdims=True)]

		leading_pho		= make_leading_pair(Photon,Photon)
		


		# Photon  EE and EB
		isEE_mask = leading_pho.isScEtaEE
		isEB_mask = leading_pho.isScEtaEB

		# -- Scale Factor for each electron


		def Trigger_Weight(eta1,pt1,eta2,pt2):
			per_ev_MC =\
			get_ele_trig_leg1_mc_Eff(eta1,pt1) * get_ele_trig_leg2_mc_Eff(eta2,pt2) +\
			get_ele_trig_leg1_mc_Eff(eta2,pt2) * get_ele_trig_leg2_mc_Eff(eta1,pt1) -\
			get_ele_trig_leg1_mc_Eff(eta1,pt1) * get_ele_trig_leg1_mc_Eff(eta2,pt2)

			per_ev_data =\
			get_ele_trig_leg1_data_Eff(eta1,pt1) * get_ele_trig_leg1_SF(eta1,pt1) * get_ele_trig_leg2_data_Eff(eta2,pt2) * get_ele_trig_leg2_SF(eta2,pt2) +\
			get_ele_trig_leg1_data_Eff(eta2,pt2) * get_ele_trig_leg1_SF(eta2,pt2) * get_ele_trig_leg2_data_Eff(eta1,pt1) * get_ele_trig_leg2_SF(eta1,pt1) -\
			get_ele_trig_leg1_data_Eff(eta1,pt1) * get_ele_trig_leg1_SF(eta1,pt1) * get_ele_trig_leg1_data_Eff(eta2,pt2) * get_ele_trig_leg1_SF(eta2,pt2)

			return per_ev_data/per_ev_MC
			

		if not isData:


			## -------------< Egamma ID and Reco Scale factor > -----------------##
			get_pho_medium_id_sf = get_pho_medium_id_sf(ak.flatten(leading_pho.eta),ak.flatten(leading_pho.pt))

			ele_reco_sf = get_ele_reco_sf(ak.flatten(leading_ele.deltaEtaSC + leading_ele.eta),ak.flatten(leading_ele.pt))* get_ele_reco_sf(ak.flatten(subleading_ele.deltaEtaSC + subleading_ele.eta),ak.flatten(subleading_ele.pt))
		
			ele_medium_id_sf = get_ele_medium_id_sf(ak.flatten(leading_ele.deltaEtaSC + leading_ele.eta),ak.flatten(leading_ele.pt))* get_ele_medium_id_sf(ak.flatten(subleading_ele.deltaEtaSC + subleading_ele.eta),ak.flatten(subleading_ele.pt))
				
			
			## -------------< Double Electron Trigger Scale factor > -----------------##
			eta1 = ak.flatten(leading_ele.deltaEtaSC + leading_ele.eta)
			eta2 = ak.flatten(subleading_ele.deltaEtaSC + subleading_ele.eta)
			pt1  = ak.flatten(leading_ele.pt)	
			pt2  = ak.flatten(subleading_ele.pt)
			ele_trig_weight = Trigger_Weight(eta1,pt1,eta2,pt2)

		

		# for masking

		diele = Triple_eee.p4
		zmass_window_mask = ak.firsts((diele.mass) > 60 | (diele.mass < 120))
		eeg_vec = diele + Photon[:,0]
		Meeg_mask = ak.firsts(eeg_vec.mass > 120)
		Event_sel_mask = zmass_window_mask & Meeg_mask

		Triple_eee_sel = Triple_eee[Event_sel_mask]
		Photon_sel     = Photon[Event_sel_mask]
		Jet_sel        = Jet[Event_sel_mask]
		MET_sel		   = MET[Event_sel_mask]

		Pho_EE = leading_pho[Event_sel_mask & isEE_mask]
		Pho_EB = leading_pho[Event_sel_mask & isEB_mask]

		##-----------  Cut flow5: Event Selection
		cut5 = np.ones(ak.sum(ak.num(Photon_sel) > 0)) * 5


		# Photon EE
		if len(Pho_EE.pt) != 0:
			Pho_EE_PT		 = flat_dim(Pho_EE.pt)
			Pho_EE_Eta	   	= flat_dim(Pho_EE.eta)
			Pho_EE_Phi		= flat_dim(Pho_EE.phi)
			Pho_EE_sieie	  = flat_dim(Pho_EE.sieie)
			Pho_EE_hoe		= flat_dim(Pho_EE.hoe)
			Pho_EE_Iso_all	= flat_dim(Pho_EE.pfRelIso03_all)
			Pho_EE_Iso_charge = flat_dim(Pho_EE.pfRelIso03_chg)

		# Photon EB
		if len(Pho_EB.pt) !=0:
			Pho_EB_PT		 = flat_dim(Pho_EB.pt)
			Pho_EB_Eta		= flat_dim(Pho_EB.eta)
			Pho_EB_Phi		= flat_dim(Pho_EB.phi)
			Pho_EB_sieie	  = flat_dim(Pho_EB.sieie)
			Pho_EB_hoe		= flat_dim(Pho_EB.hoe)
			Pho_EB_Iso_all	= flat_dim(Pho_EB.pfRelIso03_all)
			Pho_EB_Iso_charge = flat_dim(Pho_EB.pfRelIso03_chg)

		# Electrons
		ele1PT  = flat_dim(Triple_eee_sel.lep1.pt)
		ele1Eta = flat_dim(Triple_eee_sel.lep1.eta)
		ele1Phi = flat_dim(Triple_eee_sel.lep1.phi)
	
		ele2PT  = flat_dim(Triple_eee_sel.lep2.pt)
		ele2Eta = flat_dim(Triple_eee_sel.lep2.eta)
		ele2Phi = flat_dim(Triple_eee_sel.lep2.phi)
	
		ele3PT  = flat_dim(Triple_eee_sel.lep3.pt)
		ele3Eta = flat_dim(Triple_eee_sel.lep3.eta)
		ele3Phi = flat_dim(Triple_eee_sel.lep3.phi)

		Mee	 = flat_dim(Triple_eee_sel.p4.mass)
		charge  = flat_dim(Triple_eee.lep1.charge +Triple_eee.lep2.charge)
	

		# MET
		met = ak.to_numpy(MET_sel.pt)

	
		# --- Apply weight and hist  
		weights = processor.Weights(len(cut4))


		# --- skim cut-weight 
		def skim_weight(arr):
			mask1 = ~ak.is_none(arr)
			subarr = arr[mask1]
			mask2 = subarr !=0
			return ak.to_numpy(subarr[mask2])


		cuts = Event_sel_mask
		cuts_pho_EE = flat_dim(isEE_mask)
		cuts_pho_EB = flat_dim(isEB_mask)
		
		print("cut0: {0}, cut1: {1}, cut2: {2}, cut3: {3}, cut4: {4}, cut5: {5},Mee: {6}".format(len(Initial_events),len(cut1),len(cut2),len(cut3),len(cut4),len(cut5),len(Mee)))

		if not isData:
			weights.add('pileup',pu)		
			weights.add('ele_id',ele_medium_id_sf)		
			weights.add('pho_id',get_pho_medium_id_sf)		
			weights.add('ele_reco',ele_reco_sf)		
			weights.add('ele_trigger',ele_trig_weight)		
			print("#### Weight: ",weights.weight())


		# Initial events
		out["sumw"][dataset] += len(Initial_events)


		# Cut flow loop
		for cut in [cut0,cut1,cut2,cut3,cut4,cut5]:
			out["cutflow"].fill(
				dataset = dataset,
				cutflow=cut
			)
		
		
		# Primary vertex
		out['nPV'].fill(
			dataset=dataset,
			nPV = nPV,
			weight = weights.weight()
		)
		out['nPV_nw'].fill(
			dataset=dataset,
			nPV_nw = nPV
		)

		# Fill hist


			# -- met -- #
		out["met"].fill(
			dataset=dataset,
			met=met,
			weight = skim_weight(weights.weight() * cuts)
		)
			# -- Electron -- #
		out["mass"].fill(
			dataset=dataset,
			mass=Mee,
			weight = skim_weight(weights.weight() * cuts)
		)
		out["ele1pt"].fill(
			dataset=dataset,
			ele1pt=ele1PT,
			weight = skim_weight(weights.weight() * cuts)
		)
		out["ele1eta"].fill(
			dataset=dataset,
			ele1eta=ele1Eta,
			weight = skim_weight(weights.weight() * cuts)
		)
		out["ele1phi"].fill(
			dataset=dataset,
			ele1phi=ele1Phi,
			weight = skim_weight(weights.weight() * cuts)
		)
		out["ele2pt"].fill(
			dataset=dataset,
			ele2pt=ele2PT,
			weight = skim_weight(weights.weight() * cuts)
		)
		out["ele2eta"].fill(
			dataset=dataset,
			ele2eta=ele2Eta,
			weight = skim_weight(weights.weight() * cuts)
		)
		out["ele2phi"].fill(
			dataset=dataset,
			ele2phi=ele2Phi,
			weight = skim_weight(weights.weight() * cuts)
		)
		out["ele3pt"].fill(
			dataset=dataset,
			ele3pt=ele3PT,
			weight = skim_weight(weights.weight() * cuts)
		)

			# -- Photon -- #

		if len(Pho_EE.pt) != 0:





			out["pho_EE_pt"].fill(
				dataset=dataset,
				pho_EE_pt=Pho_EE_PT,
				weight = skim_weight(weights.weight() * cuts * cuts_pho_EE)
			)
			out["pho_EE_eta"].fill(
				dataset=dataset,
				pho_EE_eta=Pho_EE_Eta,
				weight = skim_weight(weights.weight() * cuts * cuts_pho_EE)
			)
			out["pho_EE_phi"].fill(
				dataset=dataset,
				pho_EE_phi=Pho_EE_Phi,
				weight = skim_weight(weights.weight() * cuts * cuts_pho_EE)
			)
			out["pho_EE_hoe"].fill(
				dataset=dataset,
				pho_EE_hoe=Pho_EE_hoe,
				weight = skim_weight(weights.weight() * cuts * cuts_pho_EE)
			)
			out["pho_EE_sieie"].fill(
				dataset=dataset,
				pho_EE_sieie=Pho_EE_sieie,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EE)
			)
			out["pho_EE_Iso_all"].fill(
				dataset=dataset,
				pho_EE_Iso_all=Pho_EE_Iso_all,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EE)
			)
			out["pho_EE_Iso_chg"].fill(
				dataset=dataset,
				pho_EE_Iso_chg=Pho_EE_Iso_charge,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EE)
			)


		if len(Pho_EB.pt) != 0:
			out["pho_EB_pt"].fill(
				dataset=dataset,
				pho_EB_pt=Pho_EB_PT,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EB)
			)
			out["pho_EB_eta"].fill(
				dataset=dataset,
				pho_EB_eta=Pho_EB_Eta,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EB)
			)
			out["pho_EB_phi"].fill(
				dataset=dataset,
				pho_EB_phi=Pho_EB_Phi,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EB)
			)
			out["pho_EB_hoe"].fill(
				dataset=dataset,
				pho_EB_hoe=Pho_EB_hoe,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EB)
			)
			out["pho_EB_sieie"].fill(
				dataset=dataset,
				pho_EB_sieie=Pho_EB_sieie,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EB)
			)
			out["pho_EB_Iso_all"].fill(
				dataset=dataset,
				pho_EB_Iso_all=Pho_EB_Iso_all,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EB)
			)
			out["pho_EB_Iso_chg"].fill(
				dataset=dataset,
				pho_EB_Iso_chg=Pho_EB_Iso_charge,
				weight = skim_weight(weights.weight() *cuts *  cuts_pho_EB)
			)


		return out

	# -- Finally! return accumulator
	def postprocess(self,accumulator):

#		scale={}
#		for d in accumulator['nPV'].identifiers('dataset'):
#			print('Scaling:',d.name)
#			dset = d.name
#			print('Xsec:',self._xsec)
#			scale[d.name] = self._lumi * self._xsec
#			
#
#
#		for histname,h in accumulator.items():
#			if histname == 'sumw' : continue
#			if isinstance(h,hist.Hist):
#				h.scale(scale,axis='dataset')

		return accumulator
# <---- Class JW_Processor


if __name__ == '__main__':

	start = time.time()
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--nWorker', type=int,
				help=" --nWorker 2", default=8)
	parser.add_argument('--metadata', type=str,
				help="--metadata xxx.json")
	parser.add_argument('--dataset', type=str,
				help="--dataset ex) Egamma_Run2018A_280000")
	args = parser.parse_args()
	
	
	
	## Prepare files
	N_node = args.nWorker
	metadata = args.metadata
	data_sample = args.dataset
	year='2018'
	xsecDY=2137.0

	## Json file reader
	with open(metadata) as fin:
		datadict = json.load(fin)



	filelist = glob.glob(datadict[data_sample])

	sample_name = data_sample.split('_')[0]
	
	
	## Read Correction file <-- on developing -->
	corr_file = "../Corrections/corrections.coffea"
	#corr_file = "corrections.coffea"
	corrections = load(corr_file)

	## Read PU weight file


	isdata=False

	if not isdata:
		pu_path_dict = {
		"DY":"mcPileupDist_DYToEE_M-50_NNPDF31_TuneCP5_13TeV-powheg-pythia8.npy",
		"TTWJets":"mcPileupDist_TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8.npy",
		"TTZtoLL":"mcPileupDist_TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8.npy",
		"WW":"mcPileupDist_WW_TuneCP5_DoubleScattering_13TeV-pythia8.npy",
		"WZ":"mcPileupDist_WZ_TuneCP5_13TeV-pythia8.npy",
		"ZZ":"mcPileupDist_ZZ_TuneCP5_13TeV-pythia8.npy",
		"tZq":"mcPileupDist_tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8.npy"
		}
		pu_path = '../Corrections/Pileup/puWeight/npy_Run2018ABD/'+ pu_path_dict[sample_name]

		print("Use the PU file: ",pu_path)
		with open(pu_path,'rb') as f:
			pu = np.load(f)

	else:
		pu=-1

#	# test one file 
#	sample_name="DY"
#	sample_name="DY"
#	filelist=["/x6/cms/store_skim_2ElIdPt20/mc/RunIISummer19UL18NanoAODv2/DYToEE_M-50_NNPDF31_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/280000/59AB328B-F0E3-F544-98BB-E5E55577C649_skim_2ElIdPt20.root"]
#



	print("Processing the sample: ",sample_name)
	samples = {
		sample_name : filelist
	}
	
	
	# Class -> Object
	JW_Processor_instance = JW_Processor(year,sample_name,xsecDY,pu,corrections)
	
	
	## -->Multi-node Executor
	result = processor.run_uproot_job(
		samples,  #dataset
		"Events", # Tree name
		JW_Processor_instance, # Class
		executor=processor.futures_executor,
		executor_args={"schema": NanoAODSchema, "workers": 16},
	#maxchunks=4,
	)
	
	outname = data_sample + '.futures'
	#outname = 'DY_test.futures'
	save(result,outname)
	
	elapsed_time = time.time() - start
	print("Time: ",elapsed_time)
