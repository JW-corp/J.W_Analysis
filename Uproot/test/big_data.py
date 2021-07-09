import uproot3 as uproot
from uproot3_methods import TVector2Array, TLorentzVectorArray
import glob
import awkward as ak
from numba import jit
import numpy as np

dir_path = "/x4/cms/dylee/Delphes/data/root/signal/*/*.root"
file_list = glob.glob(dir_path)


#@jit
def Loop(file_list):

	# define array
	histo={}



	# --Start File Loop
	for path, file, start, stop, arrays in uproot.iterate(file_list,'Delphes',['Electron*','Photon*','Missin*'],reportpath=True, reportfile=True, reportentries=True):
		print(file,start,stop, stop-start+1)


		# ------------- Define particle objects
		Electron = ak.zip(
			{
				"PT": arrays[b"Electron.PT"],
				"Eta": arrays[b"Electron.Eta"],
				"Phi": arrays[b"Electron.Phi"],
				"T": arrays[b"Electron.T"],
				"Charge": arrays[b"Electron.Charge"],
			}
		)
		
		
		# ------------- Baisc Electron Selection
		# Minimum PT cut >= 10 GeV
		Ele_PT_mask = Electron.PT >= 10
		
		# Tracker coverage |eta| <= 2.5
		Ele_Eta_mask = abs(Electron.Eta) <= 2.5
		
		# Combine cut1 and cut2 using AND(&) operator
		Electron_selection_mask = (Ele_PT_mask) & (Ele_Eta_mask)
		
		# Event mask
		Electron = Electron[Electron_selection_mask]
		Electron_event_mask = ak.num(Electron) >= 2
		
		# Apply event selection
		Electron = Electron[Electron_event_mask]
		


		# ------------- OSSF
		def find_3lep(events_leptons, builder):
		
			for leptons in events_leptons:
				builder.begin_list()
				nlep = len(leptons)
				for i0 in range(nlep):
					for i1 in range(i0+1, nlep):
						if leptons[i0].Charge + leptons[i1].Charge != 0: continue;
		
						for i2 in range(nlep):
							if len({i0, i1, i2}) < 3: continue;
							builder.begin_tuple(3)
							builder.index(0).integer(i0)
							builder.index(1).integer(i1)
							builder.index(2).integer(i2)
							builder.end_tuple()
				builder.end_list()
			return builder
		
		eee_triplet_idx = find_3lep(Electron,ak.ArrayBuilder()).snapshot()
		ossf_mask = ak.num(eee_triplet_idx) == 2
		
		eee_Electron = Electron[ossf_mask]
		eee_triplet_idx = eee_triplet_idx[ossf_mask]
		
		Triple_electron = [eee_Electron[eee_triplet_idx[idx]] for idx in "012"]
		def make_TLV(eee):
			eee = ak.to_awkward0(eee)
			TLV = TLorentzVectorArray.from_ptetaphim(eee.PT, eee.Eta, eee.Phi, eee.T*0)
			return TLV
		Triple_eee = ak.zip({
			"lep1" : Triple_electron[0],
			"lep2" : Triple_electron[1],
			"lep3" : Triple_electron[2]
		})


		lepton1 = make_TLV(Triple_eee.lep1)
		lepton2 = make_TLV(Triple_eee.lep2)
		diele = lepton1 + lepton2
		diele_mass_ak0 = ak.from_awkward0(diele.mass)
		bestZ_idx = ak.singletons(ak.argmin(abs(diele_mass_ak0 - 91.1876), axis=1))

		Triple_eee = Triple_eee[bestZ_idx]

		
		# ------------- Make array
		diele = make_TLV(Triple_eee.lep1) + make_TLV(Triple_eee.lep2)
		print(diele.mass)


		if len(histo) == 0 :
			histo["lep1Flav"] = ak.to_numpy(ak.flatten(Triple_eee.lep1.Charge * 1))
			histo["lep1PT"]   = ak.to_numpy(ak.flatten(Triple_eee.lep1.PT))
			histo["lep1Eta"]  = ak.to_numpy(ak.flatten(Triple_eee.lep1.Eta))
			histo["lep1Phi"]  = ak.to_numpy(ak.flatten(Triple_eee.lep1.Phi))
			histo["Mll"]	  = ak.to_numpy(ak.flatten(diele.mass))
		else:
			histo["lep1Flav"] = np.concatenate((histo["lep1Flav"],ak.to_numpy(ak.flatten(Triple_eee.lep1.Charge * 1))))
			histo["lep1PT"] = np.concatenate((histo["lep1PT"]  ,ak.to_numpy(ak.flatten(Triple_eee.lep1.PT))))
			histo["lep1Eta"] = np.concatenate((histo["lep1Eta"] ,ak.to_numpy(ak.flatten(Triple_eee.lep1.Eta))))
			histo["lep1Phi"] = np.concatenate((histo["lep1Phi"] ,ak.to_numpy(ak.flatten(Triple_eee.lep1.Phi))))
			histo["Mll"] = np.concatenate((histo["Mll"]		,ak.to_numpy(ak.flatten(diele.mass))))

		print(len(histo['lep1Flav']))

	print(histo)

				
	return histo



histo = Loop(file_list)
outname = str(file).split("'")[1].split('.')[0] # Looke like "DelPy_run_01_"
outfile = outname + '.npy'
np.save(outfile.histo)
