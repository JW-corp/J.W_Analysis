import awkward1 as ak
import uproot3
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
	def __init__(self,year,setname,xsec,puweight_arr,corrections):


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
		

		# Corrrection set <-- on developing -->
		self._corrections = corrections
		self._puweight_arr = puweight_arr

		



		# hist set
		self._accumulator = processor.dict_accumulator({

			"sumw": processor.defaultdict_accumulator(float),
	
			# -- Leading Electron  -- #

			"ele1pt": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele1pt","Leading Electron $P_{T}$ [GeV]", 300, 0, 600),
			),

			"ele1eta": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele1eta","Leading Electron $\eta$ [GeV]", 20, -5, 5),
			),

			"ele1phi": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele1phi","Leading Electron $\phi$ [GeV]", 20, -3.15, 3.15),
			),

			# -- Sub-leading Electron  -- #

			"ele2pt": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele2pt","Subleading $Electron P_{T}$ [GeV]", 300, 0, 600),
			),
			
			"ele2eta": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele2eta","Subleading Electron $\eta$ [GeV]", 20, -5, 5),
			),

			"ele2phi": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("ele2phi","Subleading Electron $\phi$ [GeV]", 20, -3.15, 3.15),
			),


			# -- Photon --#
	
			"phopt": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("phopt","Leading Photon $P_{T}$ [GeV]", 300, 0, 600),
            ),
            "phoeta": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("phoeta","Photon  $\eta$ ", 50, -5, 5),
            ),
            "phophi": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("phophi","Photon $\phi$ ", 50, -3.15, 3.15),
            ),



			# -- Photon EE -- #

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
			"pho_EE_sieie": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_sieie","Photon EE sieie", 100, 0, 0.03),
			),
			"pho_EE_Iso_chg": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EE_Iso_chg","Photon EE pfReoIso03_charge", 200, 0, 1),
			),

			# -- Photon EB -- #

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

			"pho_EB_Iso_chg": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_Iso_chg","Photon EB pfReoIso03_charge", 100, 0, 0.03),
			),

			"pho_EB_sieie": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("pho_EB_sieie","Photon EB sieie", 100, 0, 0.1),
			)



			# -- Kinematic variables -- #
			"mass": hist.Hist(
				"Events",
				hist.Cat("dataset","Dataset"),
				hist.Bin("Mee","M(ee) [GeV]", 100, 0, 200),
			),

			"mass_eea": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("mass_eea","$m_{e+e-\gamma}$ [GeV]", 300, 0, 600),
            ),
			
			"met": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("met","met [GeV]", 300, 0, 600),
            ),

			"dR_ae1": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("dR_ae1","dR(ae1)", 100, 0, 4),
            ),
            "dR_ae2": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("dR_ae2","$dR(ae2)$", 100, 0, 4),
            ),
			
			"dR_aj": hist.Hist(
                "Events",
                hist.Cat("dataset","Dataset"),
                hist.Bin("dR_aj","dR(aj1)", 100, 0, 4),
            ),			




			})
		
	# -- Accumulator: accumulate histograms
	@property
	def accumulator(self):
		return self._accumulator

	# -- Main function : Process events
	def process(self, events):

		# Initialize accumulator
		out = self.accumulator.identity()
		dataset = setname
		#events.metadata['dataset']
		

		isData = 'genWeight' not in events.fields
		

		selection = processor.PackedSelection()

		# Cut flow
		cut0 = np.zeros(len(events))
		

		# --- Selection

		# << flat dim helper function >>
		def flat_dim(arr):

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


		
		##---------------  Cut flow1: Passing Triggers ------------------------#
		Initial_events = events
		print("#### Initial events: ",Initial_events)
		#events = events[single_ele_triggers_arr | double_ele_triggers_arr]
		events = events[double_ele_triggers_arr]
		
		cut1 = np.ones(len(events))
		print("#### cut1: ",len(cut1))
		# Particle Identification
		Electron = events.Electron
		Photon   = events.Photon


		##-----------  Cut flow2: Events contain 2 electron and 1 photon are selected  ( Basic cut applied : No Photon ID)
        def Particle_selection(ele,pho):


            EleSelmask = EleSelmask = ((ele.pt > 10) & (np.abs(ele.eta + ele.deltaEtaSC) < 1.4442) & (ele.cutBased > 2) & (abs(ele.dxy) < 0.05) & (abs(ele.dz) < 0.1)) | \
                ((ele.pt > 10) & (np.abs(ele.eta + ele.deltaEtaSC) > 1.5660) & (np.abs(ele.eta + ele.deltaEtaSC) < 2.5) & (ele.cutBased > 2) & (abs(ele.dxy) < 0.1) & (abs(ele.dz) < 0.2))


            # Photon selection
            isgap_mask = (abs(pho.eta) < 1.442)  |  ((abs(pho.eta) > 1.566) & (abs(pho.eta) < 2.5))
            Pixel_seed_mask = ~pho.pixelSeed
            PhoSelmask = (pho.pt > 20)  & isgap_mask &  Pixel_seed_mask


            return EleSelmask,PhoSelmask

		
		Electron_mask, Photon_mask  = Particle_selection(Electron,Photon)
        Ele_channel_mask = ak.num(Electron[Electron_mask])  > 1
        Pho_channel_mask = ak.num(Photon[Photon_mask]) > 0
        Ele_channel_events = events[Ele_channel_mask & Pho_channel_mask]	
		
		cut2 = np.ones(len(Ele_channel_events)) * 2


			
		# Particle array
		Electron = Ele_channel_events.Electron
		Photon  = Ele_channel_events.Photon
		Jet     = Ele_channel_events.Jet
		MET      = Ele_channel_events.MET
		
		Electron_mask,Photon_mask = Particle_selection(Electron,Photon)
		Electron = Electron[Electron_mask]
		Photon  = Photon[Photon_mask]

		@numba.njit
		def find_2lep(events_leptons,builder):
		    for leptons in events_leptons:
		
		        builder.begin_list()
		        nlep = len(leptons)
		        for i0 in range(nlep):
		            for i1 in range(i0+1,nlep):
		                if leptons[i0].charge + leptons[i1].charge != 0: continue;
		                
		                if nlep == 2:
		                    builder.begin_tuple(2)
		                    builder.index(0).integer(i0)
		                    builder.index(1).integer(i1)
		                    builder.end_tuple()  
		
		            
		                else:
		                    for i2 in range(nlep):
		                        if len({i0,i1,i2}) < 3: continue;
		                        builder.begin_tuple(3)
		                        builder.index(0).integer(i0)
		                        builder.index(1).integer(i1)
		                        builder.index(2).integer(i2)
		                        builder.end_tuple()
		        builder.end_list()
		    return builder

		ossf_idx = find_2lep(Electron,ak.ArrayBuilder()).snapshot()


		# OSSF cut
		ossf_mask = ak.num(ossf_idx) >= 1
		ossf_idx = ossf_idx[ossf_mask]
		
		Ele_channel_events = Ele_channel_events[ossf_mask]
		Electron= Electron[ossf_mask]
		Photon= Photon[ossf_mask]
		Jet= Jet[ossf_mask]
		MET = MET[ossf_mask]

		Double_electron = [Electron[ossf_idx[idx]] for idx in "01"]
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
	
		
		Diele    = ak.zip({"lep1":Double_electron[0],
                   "lep2":Double_electron[1],
                     "p4":TLorentz_vector(Double_electron[0]+Double_electron[1])})


		bestZ_idx = ak.singletons(ak.argmin(abs(Diele.p4.mass - 91.1876), axis=1))
		Diele = Diele[bestZ_idx]

		leading_ele, subleading_ele  = ak.flatten(TLorentz_vector_cylinder(Diele.lep1)),ak.flatten(TLorentz_vector_cylinder(Diele.lep2))

		##-----------  Cut flow4: Photon "cleaning"  -- Basis of array-dimension
		def make_DR(ele1,ele2,pho,jet):
		
		    dR_e1pho  = ele1.delta_r(pho) # dR pho,ele1
		    dR_e2pho  = ele2.delta_r(pho) # dR pho,ele2
		    dR_phojet = jet[:,0].delta_r(pho) # dR pho,jet # #--> Need check
		
		    #dR_mask    = (dR_e1pho > 0.4) & (dR_e2pho > 0.4)&  (dR_e3pho > 0.4) & (dR_phojet > 0.4) #--> Need check
		    dR_mask = (dR_e1pho > 0.4) & (dR_e2pho > 0.4)
		
		    #return dR_mask,dR_e1pho,dR_e2pho,dR_e3pho,dR_phojet #--> Need check
		    return dR_mask,dR_e1pho,dR_e2pho,dR_phojet


		dR_mask,dR_e1pho,dR_e2pho,dR_phojet  = make_DR(leading_ele,subleading_ele,Photon,Jet)
		Photon = Photon[dR_mask]


		Ele_channel_events  = Ele_channel_events[ak.num(Photon) > 0]
		Diele               = Diele[ak.num(Photon) > 0]
		Jet                 = Jet[ak.num(Photon) > 0]
		MET                 = MET[ak.num(Photon) > 0]
		Photon              = Photon[ak.num(Photon) > 0] # Beware the order! Photon must be located last!

		

		#@numba.njit ## Numba compile -- Boost!
		def PhotonVID(vid, idBit):
			rBit = 0
			for x in range(0, 7):
				rBit |= (1 << x) if ((vid >> (x * 2)) & 0b11 >= idBit) else 0
			return rBit


		#@numba.njit ## Numba compile -- Boost!
		def make_fake_obj_mask(Pho,builder):
		
		    for eventIdx,pho in enumerate(tqdm(Pho)):   # --Event Loop
		        builder.begin_list()
		        if len(pho) < 1: continue;
		    
		            
		        for phoIdx,_ in enumerate(pho):# --Photon Loop
		        
		            vid = Pho[eventIdx][phoIdx].vidNestedWPBitmap
		            vid_cuts1 = PhotonVID(vid,1) # Loose photon
		            vid_cuts2 = PhotonVID(vid,2) # Medium photon
		            vid_cuts3 = PhotonVID(vid,3) # Tight photon
		
		            # Field name
		            # |0|0|0|0|0|0|0| 
		            # |IsoPho|IsoNeu|IsoChg|Sieie|hoe|scEta|PT|
		
		            # 1. Turn off cut (ex turn off Sieie
		            # |1|1|1|0|1|1|1| = |1|1|1|0|1|1|1|
		
		            # 2. Inverse cut (ex inverse Sieie)
		            # |1|1|1|1|1|1|1| = |1|1|1|0|1|1|1|
		
		            
		                
		            #if (vid_cuts2 & 0b1111111 == 0b1111111): # Cut applied
		            if (vid_cuts2 & 0b1111111 == 0b1110111): # Inverse Sieie
		            #if (vid_cuts2 & 0b1110111 == 0b1110111): # Without Sieie
		            
		                if (Pho[eventIdx][phoIdx].isScEtaEB) & (Pho[eventIdx][phoIdx].sieie < 0.01015 * 1.75):
		                    builder.boolean(True)
		            
		                elif (Pho[eventIdx][phoIdx].isScEtaEE) & (Pho[eventIdx][phoIdx].sieie < 0.0272 * 1.75):
		                    builder.boolean(True)
		                else: builder.boolean(False)
		
		                
		            else:
		
		                builder.boolean(False)
		
		        builder.end_list()
		                
		    return builder
			# - Sieie-> EB: 0.01015 EE: 0.0272

		#@numba.njit ## Numba compile -- Boost!
		def make_IsochgSide_mask(Pho,builder):
		
		
		    for eventIdx,pho in enumerate(tqdm(Pho)): # --Event Loop
		        builder.begin_list()
		        if len(pho) < 1: continue;
		        
		        for phoIdx,_ in enumerate(pho): # --Photon Loop
		            
		            vid = Pho[eventIdx][phoIdx].vidNestedWPBitmap
		            vid_cuts1 = PhotonVID(vid,1) # Loose photon
		            vid_cuts2 = PhotonVID(vid,2) # Medium photon
		            vid_cuts3 = PhotonVID(vid,3) # Tight photon
		
		            #if (vid_cuts2 & 0b1111111 == 0b1111111): # Cut applied
		            if (vid_cuts2 & 0b1111111 == 0b1101111): # Inverse Isochg
		            #if (vid_cuts2 & 0b1101111 == 0b1101111): # Withtou Isochg
		                isochg = Pho[eventIdx][phoIdx].pfRelIso03_chg * Pho[eventIdx][phoIdx].pt
		                
		                if (isochg >= 4) & (isochg <= 10):               
		                    builder.boolean(True)
		                else: 
		                    builder.boolean(False)
		
		                
		            else:
		                #builder.begin_list()
		                builder.boolean(False)
		                #builder.end_list()      
		        builder.end_list()
		                
		    return builder
			# - IsoChg-> EB: 1.141 EE: 1.051

		is_photon_sieie   = make_fake_obj_mask(Photon, ak.ArrayBuilder()).snapshot()
		is_photon_Isochg  = make_IsochgSide_mask(Photon,ak.ArrayBuilder()).snapshot()
		
		# -- wether the cuts are applied or not ------> #
		Photon_invsieie  = Photon[is_photon_sieie]   
		Photon_invIsochg = Photon[is_photon_Isochg]
		
		Photon_invsieieEB = Photon_invsieie[Photon_invsieie.isScEtaEB]
		Photon_invsieieEE = Photon_invsieie[Photon_invsieie.isScEtaEE]
		
		Photon_invIsochgEB = Photon_invIsochg[Photon_invIsochg.isScEtaEB]
		Photon_invIsochgEE = Photon_invIsochg[Photon_invIsochg.isScEtaEE]
		
		
		hist_sieie_EB = flat_dim(Photon_invsieieEB.sieie)
		hist_sieie_EE = flat_dim(Photon_invsieieEE.sieie)
		
		hist_Isochg_EB =  flat_dim(Photon_invIsochgEB.pfRelIso03_chg * Photon_invIsochgEB.pt)
		hist_Isochg_EE =  flat_dim(Photon_invIsochgEE.pfRelIso03_chg * Photon_invIsochgEE.pt)
		# <------------------------- #
			
		Photon_template_mask = (is_photon_sieie) | (is_photon_Isochg)
		Photon = Photon[Photon_template_mask]
		Ele_channel_fake_template = Ele_channel_events[ak.num(Photon) == 1]
		Diele  = Diele[ak.num(Photon) == 1]
		Jet    = Jet[ak.num(Photon) == 1]
		MET    = MET[ak.num(Photon) == 1]
		Photon = Photon[ak.num(Photon) == 1]

		##-----------  Cut flow5 and 6: Event Selection


		leading_pho     = Photon

		# bjet veto
		bJet_selmask = (Jet.btagCMVA > -0.5844)
		bJet_veto    = ak.num(Jet[bJet_selmask])==0
		cut5 = np.ones(ak.sum(ak.num(leading_pho[bJet_veto] > 0 ))) * 5
		
		# Z mass window
		zmass_window_mask = ak.firsts((Diele.p4.mass) > 60 | (Diele.p4.mass < 120))
		
		# M(eea) cuts 
		eeg_vec           = Diele.p4 + leading_pho
		Meeg_mask         = ak.firsts(eeg_vec.mass > 120)


		
		# --------Mask -------#
		Event_sel_mask   = bJet_veto & zmass_window_mask & Meeg_mask
		Jet_sel			 = Jet[Event_sel_mask]
		Diele_sel		 = Diele[Event_sel_mask]
		leading_pho_sel	 = leading_pho[Event_sel_mask]
		MET_sel			 = MET[Event_sel_mask]
				

		# Photon  EE and EB
		isEE_mask = leading_pho.isScEtaEE
		isEB_mask = leading_pho.isScEtaEB
		Pho_EE = leading_pho[isEE_mask & Event_sel_mask]
		Pho_EB = leading_pho[isEB_mask & Event_sel_mask]
		


		# -------------------- Fill hist ---------------------------#	

				# -- Ele1 --#
		Ele1_pt  = flat_dim(Diele_sel.lep1.pt)
		Ele1_eta = flat_dim(Diele_sel.lep1.eta)
		Ele1_phi = flat_dim(Diele_sel.lep1.phi)
		
				# -- Ele2 --#
		Ele2_pt  = flat_dim(Diele_sel.lep2.pt)
		Ele2_eta = flat_dim(Diele_sel.lep2.eta)
		Ele2_phi = flat_dim(Diele_sel.lep2.phi)
		

				# -- Pho -- #
		Pho_PT  = flat_dim(leading_pho_sel.pt)
		Pho_Eta = flat_dim(leading_pho_sel.eta)
		Pho_Phi = flat_dim(leading_pho_sel.phi)

				# -- Pho EB --#
		Pho_EB_pt  = flat_dim(Pho_EB.pt)
		Pho_EB_eta = flat_dim(Pho_EB.eta)
		Pho_EB_phi = flat_dim(Pho_EB.phi)
		Pho_EB_Isochg = flat_dim(Pho_EE.pfRelIso03_chg)
		Pho_EB_Sieie  = flat_dim(Pho_EE.sieie)

				# -- Pho EE --#
		Pho_EE_pt     = flat_dim(Pho_EE.pt)
		Pho_EE_eta    = flat_dim(Pho_EE.eta)
		Pho_EE_phi    = flat_dim(Pho_EE.phi)
		Pho_EE_Isochg = flat_dim(Pho_EE.pfRelIso03_chg)
		Pho_EE_Sieie  = flat_dim(Pho_EE.sieie)

				# --Kinematics --#
		Diele_mass = flat_dim(Diele_sel.p4.mass)

		eeg_vec = Diele_sel.p4 + leading_pho_sel
		eeg_mass = flat_dim(eeg_vec.mass)
	
		leading_ele, subleading_ele = ak.flatten(TLorentz_vector_cylinder(Diele_sel.lep1)),ak.flatten(TLorentz_vector_cylinder(Diele_sel.lep2))
		dR_e1pho  = flat_dim(leading_ele.delta_r(leading_pho_sel)) # dR pho,ele1
		dR_e2pho  = flat_dim(subleading_ele.delta_r(leading_pho_sel)) # dR pho,ele2
		dR_jpho   = flat_dim(Jet_sel[:,0].delta_r(leading_pho_sel))



		# Initial events
		out["sumw"][dataset] += len(Initial_events)

		out['ele1pt'].fill(
			dataset=dataset,
			ele1pt=Ele1_pt
		)


		out["pho_EE_pt"].fill(
			dataset=dataset,
			pho_EE_pt=Pho_EE_PT,
		)
		out["pho_EE_sieie"].fill(
			dataset=dataset,
			pho_EE_sieie=Pho_EE_sieie,
		)
		out["pho_EB_pt"].fill(
			dataset=dataset,
			pho_EB_pt=Pho_EB_PT,
		)
		out["pho_EB_sieie"].fill(
			dataset=dataset,
			pho_EB_sieie=Pho_EB_sieie,
		)

		return out

	# -- Finally! return accumulator
	def postprocess(self,accumulator):
		return accumulator
# <---- Class JW_Processor


if __name__ == '__main__':

	start = time.time()
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--nWorker', type=int,
				help=" --nWorker 2", default=20)
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
	setname = metadata.split('.')[0].split('/')[1]
	
	
	## Read Correction file <-- on developing -->
	#corr_file = "../Corrections/corrections.coffea"
	corr_file = "corrections.coffea"
	corrections = load(corr_file)

	## Read PU weight file
	#with open('../Corrections/Pileup/puWeight/pu_weight_RunAB.npy','rb') as f:
	with open('pu_weight_RunAB.npy','rb') as f:
		pu = np.load(f)


#	# test one file 
#	sample_name="DY"
#	setname="DY"
#	filelist=["/x6/cms/store_skim_2ElIdPt20/mc/RunIISummer19UL18NanoAODv2/DYToEE_M-50_NNPDF31_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/280000/59AB328B-F0E3-F544-98BB-E5E55577C649_skim_2ElIdPt20.root"]
#



	print(sample_name)
	samples = {
		sample_name : filelist
	}
	
	
	# Class -> Object
	#JW_Processor_instance = JW_Processor(year,setname,corrections,xsecDY)  <--on developing-->
	JW_Processor_instance = JW_Processor(year,setname,xsecDY,pu,corrections)
	
	
	## -->Multi-node Executor
	result = processor.run_uproot_job(
		samples,  #dataset
		"Events", # Tree name
		JW_Processor_instance, # Class
		executor=processor.futures_executor,
		executor_args={"schema": NanoAODSchema, "workers": 20},
	#maxchunks=4,
	)
	
	outname = data_sample + '.futures'
	#outname = 'DY_test.futures'
	save(result,outname)
	
	elapsed_time = time.time() - start
	print("Time: ",elapsed_time)
