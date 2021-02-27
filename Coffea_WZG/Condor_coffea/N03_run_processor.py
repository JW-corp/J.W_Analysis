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
	
			'cutflow': hist.Hist(
				'Events',
				hist.Cat('dataset', 'Dataset'),
				hist.Bin('cutflow', 'Cut index', [0, 1, 2, 3, 4,5])
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


		
		Initial_events = events
		print("#### Initial events: ",Initial_events)
		#events = events[single_ele_triggers_arr | double_ele_triggers_arr]
		events = events[double_ele_triggers_arr]
		
		##----------- Cut flow1: Passing Triggers
		cut1 = np.ones(len(events))
		print("#### cut1: ",len(cut1))
		# Particle Identification
		Electron = events.Electron

		def Electron_selection(ele):
			return(ele.pt > 25) & (np.abs(ele.eta) < 2.5) & (ele.cutBased > 2)
		

		# Electron channel
		Electron_mask = Electron_selection(Electron)
		Ele_channel_mask = ak.num(Electron[Electron_mask]) > 1
		Ele_channel_events = events[Ele_channel_mask]


		##-----------  Cut flow2: Electron channel
		cut2 = np.ones(len(Ele_channel_events)) * 2
		print("#### cut2: ",len(cut2))
		
		# --- Calculate Scale factor weight
		
		if not isData:
			# PU weight with lookup table <-- On developing -->
			#get_pu_weight = self._corrections['get_pu_weight'][self._year]
			#pu = get_pu_weight(events.Pileup.nTrueInt)
	
			get_ele_reco_sf = self._corrections['get_ele_reco_sf'][self._year]
			get_ele_medium_id_sf = self._corrections['get_ele_medium_id_sf'][self._year]


			get_ele_trig_leg1_SF		= self._corrections['get_ele_trig_leg1_SF'][self._year]
			get_ele_trig_leg1_data_Eff	= self._corrections['get_ele_trig_leg1_data_Eff'][self._year]
			get_ele_trig_leg1_mc_Eff	= self._corrections['get_ele_trig_leg1_mc_Eff'][self._year]
			get_ele_trig_leg2_SF		= self._corrections['get_ele_trig_leg2_SF'][self._year]
			get_ele_trig_leg2_data_Eff  = self._corrections['get_ele_trig_leg2_data_Eff'][self._year]
			get_ele_trig_leg2_mc_Eff	= self._corrections['get_ele_trig_leg2_mc_Eff'][self._year]





			# PU weight with custom made npy and multi-indexing
			pu_weight_idx = ak.values_astype(Ele_channel_events.Pileup.nTrueInt,"int64")
			pu = self._puweight_arr[pu_weight_idx]
			nPV = Ele_channel_events.PV.npvsGood
		
		else:
			nPV = Ele_channel_events.PV.npvsGood


		# Electron array
		Ele = Ele_channel_events.Electron
		Electron_mask = Electron_selection(Ele)	
		Ele_sel = Ele[Electron_mask]	



		# Electron pair
		ele_pairs = ak.combinations(Ele_sel,2,axis=1)
		ele_left, ele_right = ak.unzip(ele_pairs)
		diele = ele_left + ele_right

		# OS
		os_mask		 = diele.charge == 0 
		os_diele	 = diele[os_mask]
		os_ele_left  = ele_left[os_mask]
		os_ele_right = ele_right[os_mask]
		os_event_mask = ak.num(os_diele) > 0
		Ele_os_channel_events = Ele_channel_events[os_event_mask]
		#selection.add('ossf',os_event_mask)


		# Helper function: High PT argmax
		def make_leading_pair(target,base):

			return target[ak.argmax(base.pt,axis=1,keepdims=True)]


		# -- Only Leading pair --
		leading_diele = make_leading_pair(diele,diele)
		leading_ele   = make_leading_pair(ele_left,diele)
		subleading_ele= make_leading_pair(ele_right,diele)

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
			ele_mediun_id_sf = get_ele_medium_id_sf(ak.flatten(leading_ele.deltaEtaSC + leading_ele.eta),ak.flatten(leading_ele.pt))* get_ele_medium_id_sf(ak.flatten(subleading_ele.deltaEtaSC + subleading_ele.eta),ak.flatten(subleading_ele.pt))
			#print("Ele ID SC---->",ele_loose_id_sf)
			
			ele_reco_sf = get_ele_reco_sf(ak.flatten(leading_ele.deltaEtaSC + leading_ele.eta),ak.flatten(leading_ele.pt))* get_ele_reco_sf(ak.flatten(subleading_ele.deltaEtaSC + subleading_ele.eta),ak.flatten(subleading_ele.pt))
			#print("Ele RECO SC---->",ele_reco_sf)
		
		
			eta1 = ak.flatten(leading_ele.deltaEtaSC + leading_ele.eta)
			eta2 = ak.flatten(subleading_ele.deltaEtaSC + subleading_ele.eta)
			pt1  = ak.flatten(leading_ele.pt)	
			pt2  = ak.flatten(subleading_ele.pt)

			ele_trig_weight = Trigger_Weight(eta1,pt1,eta2,pt2)
			print("#### Test print trigger weight ####")
			print(ele_trig_weight)

		# --OS and Leading pair --
		leading_os_diele = make_leading_pair(os_diele,os_diele)
		leading_os_ele   = make_leading_pair(os_ele_left,os_diele)
		subleading_os_ele= make_leading_pair(os_ele_right,os_diele)

		##-----------  Cut flow3: OSSF
		cut3 = np.ones(len(flat_dim(leading_os_diele))) * 3
		print("#### cut3: ",len(cut3))

		# Helper function: Zmass window
		def makeZmass_window_mask(dielecs,start=60,end=120):
			mask = (dielecs.mass >= start) & (dielecs.mass <= end)	
			return mask

		# -- OS and Leading pair --
		Zmass_mask_os = makeZmass_window_mask(leading_os_diele)
		leading_os_Zwindow_ele = leading_os_ele[Zmass_mask_os]
		subleading_os_Zwindow_ele = subleading_os_ele[Zmass_mask_os]
		leading_os_Zwindow_diele = leading_os_diele[Zmass_mask_os]
		

		# for masking
		Zmass_event_mask = makeZmass_window_mask(leading_diele)
		Zmass_os_event_mask= ak.flatten(os_event_mask * Zmass_event_mask)
		

		Ele_Zmass_os_events = Ele_channel_events[Zmass_os_event_mask]

		##-----------  Cut flow4: Zmass
		cut4 = np.ones(len(flat_dim(leading_os_Zwindow_diele))) * 4
		print("#### cut4: ",len(cut4))


		
		## << Selection method -- Need validation >>
		#print("a--->",len(Ele_channel_events))
		#print("b--->",len(Ele_os_channel_events))
		#print("b2--->",len(cut3))
		#print("c--->",len(Ele_Zmass_os_events))
		#print("c2--->",len(cut4))


		ele1PT  = flat_dim(leading_os_Zwindow_ele.pt)
		ele1Eta = flat_dim(leading_os_Zwindow_ele.eta)
		ele1Phi = flat_dim(leading_os_Zwindow_ele.phi)
		ele2PT  = flat_dim(subleading_os_Zwindow_ele.pt)
		ele2Eta = flat_dim(subleading_os_Zwindow_ele.eta)
		ele2Phi = flat_dim(subleading_os_Zwindow_ele.phi)
		Mee	 = flat_dim(leading_os_Zwindow_diele.mass)
		charge  = flat_dim(leading_os_Zwindow_diele.charge)
		
		# --- Apply weight and hist  
		weights = processor.Weights(len(cut2))


		# --- skim cut-weight 
		def skim_weight(arr):
			mask1 = ~ak.is_none(arr)
			subarr = arr[mask1]
			mask2 = subarr !=0
			return ak.to_numpy(subarr[mask2])

		cuts = ak.flatten(Zmass_mask_os)
		if not isData:
			weights.add('pileup',pu)		
			weights.add('ele_id',ele_medium_id_sf)		
			weights.add('ele_reco',ele_reco_sf)		
			weights.add('ele_trigger',ele_trig_weight)		

		# Initial events
		out["sumw"][dataset] += len(Initial_events)


		# Cut flow loop
		for cut in [cut0,cut1,cut2,cut3,cut4]:
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

		# Physics varibles passing Zwindow
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
