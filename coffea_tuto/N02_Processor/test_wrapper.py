from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor, hist
import numpy as np
import matplotlib.pyplot as plt
from coffea import lumi_tools
import glob
from coffea.util import load, save
import awkward as ak

import warnings

warnings.filterwarnings("ignore")


class JW_Processor(processor.ProcessorABC):

	# -- Initializer
	def __init__(self, year, sample_name):

		self._year = year
		self._sample_name = sample_name
		# Trigger set
		self._doubleelectron_triggers = {
			"2018": [
				"Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",  # Recomended
			]
		}

		# hist set
		self._accumulator = processor.dict_accumulator(
			{
				"sumw": processor.defaultdict_accumulator(float),
				# -- Kinematics -- #
				"mass": hist.Hist(
					"Events",
					hist.Cat("dataset", "Dataset"),
					hist.Bin("mass", "$m_{e+e-}$ [GeV]", 100, 0, 200),
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
		dataset = self._sample_name

		# Data or MC
		isData = "genWeight" not in events.fields

		# << Sort by PT  helper function >>
		def sort_by_pt(ele, pho, jet):
			ele = ele[ak.argsort(ele.pt, ascending=False, axis=1)]
			pho = pho[ak.argsort(pho.pt, ascending=False, axis=1)]
			jet = jet[ak.argsort(jet.pt, ascending=False, axis=1)]

			return ele, pho, jet

		events.Electron, events.Photon, events.Jet = sort_by_pt(
			events.Electron, events.Photon, events.Jet
		)

		# Sort particle order by PT  # RunD --> has problem
		events.Electron, events.Photon, events.Jet = sort_by_pt(
			events.Electron, events.Photon, events.Jet
		)

		Initial_events = len(events)
		# Electron selection
		Electron = events.Electron

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

		# Select events with two electrons
		Diele_mask = ak.num(Electron) == 2
		Electron = Electron[Diele_mask]
		events = events[Diele_mask]

		# OSSF
		diele = Electron[:, 0] + Electron[:, 1]

		# OSSF
		ossf_mask = diele.charge == 0

		# Z mass window
		Mee_cut_mask = diele.mass > 10

		# Electron PT cuts
		Elept_mask = (Electron[:, 0].pt >= 25) & (Electron[:, 1].pt >= 10)

		# Mask
		Event_sel_mask = Elept_mask & ossf_mask & Mee_cut_mask

		diele = diele[Event_sel_mask]
		Electron = Electron[Event_sel_mask]
		events = events[Event_sel_mask]

		# Fill hist
		Mee_arr = diele.mass
		out["sumw"][dataset] +=Initial_events
		out["mass"].fill(dataset=dataset, mass=Mee_arr)
		return out

	# -- Finally! return accumulator
	def postprocess(self, accumulator):
		return accumulator
