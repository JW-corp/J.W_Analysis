## -- uproot3 or 4
#import uproot3 as uproot # for uproot3
import uproot # for uproot4



import glob
import awkward as ak
from numba import jit
import numpy as np
from tqdm import tqdm

import time
start_time = time.time()

# using file list
f = open('file_list')
file_list = [line.rstrip() for line in f]

# using glob
#dir_path = "/x4/cms/dylee/Delphes/data/Storage/Second_data/root/signal/condorDelPyOut/*.root"
#file_list = glob.glob(dir_path)



# for uproot4
flist=[]
for f in file_list:
	flist.append(f + ':Delphes')

# for uproot3
#flist = file_list


branches = ['Electron.PT' ,'Electron.Eta' ,'Electron.Phi' ,'Electron.Charge' ,'PhotonLoose.PT','PhotonLoose.Eta','PhotonLoose.Phi','MuonLoose.PT' ,'MuonLoose.Eta' ,'MuonLoose.Phi' ,'MuonLoose.Charge' ,'PuppiMissingET.MET' ,'PuppiMissingET.Phi']



# -- 54s for 470000 events for uproot3
# -- 22s for 470000 events for uproot4
#@jit # for future developer...
def Loop(file_list):

	# define array
	histo={}

	# --Start File Loop
	for arrays in uproot.iterate(flist,branches): #  for Uproot4
	#for arrays in uproot.iterate(flist,'Delphes',branches): # for Uproot3

		print(len(arrays))
		Electron = ak.zip(
		{
		 "PT": arrays[b"Electron.PT"],
	   	 "Eta": arrays[b"Electron.Eta"],
	   	 "Phi": arrays[b"Electron.Phi"],
	   	 "Charge": arrays[b"Electron.Charge"],
		})

		Muon = ak.zip(
		{
		 "PT": arrays[b"MuonLoose.PT"],
	   	 "Eta": arrays[b"MuonLoose.Eta"],
	   	 "Phi": arrays[b"MuonLoose.Phi"],
	   	 "Charge": arrays[b"MuonLoose.Charge"],
		})

		Photon = ak.zip(
		{
		 "PT": arrays[b"PhotonLoose.PT"],
	   	 "Eta": arrays[b"PhotonLoose.Eta"],
	   	 "Phi": arrays[b"PhotonLoose.Phi"],
		})

		MET = ak.zip(
		{
		 "PT": arrays[b"PuppiMissingET.MET"],
	   	 "Phi": arrays[b"PuppiMissingET.Phi"],
		})

		print(Muon.PT)


# -- 153s for 470000 events
def Lazy(file_list,branches):

	cache = uproot.ArrayCache("2 GB")
	tree = uproot.lazyarrays(
		file_list,
		"Delphes",
		branches,
		cache=cache,
	)

	print(len(tree))

	Electron = ak.zip(
	{
	 "PT": tree["Electron.PT"],
   	 "Eta": tree["Electron.Eta"],
   	 "Phi": tree["Electron.Phi"],
   	 "Charge": tree["Electron.Charge"],
	})

	Muon = ak.zip(
	{
	 "PT": tree["MuonLoose.PT"],
   	 "Eta": tree["MuonLoose.Eta"],
   	 "Phi": tree["MuonLoose.Phi"],
   	 "Charge": tree["MuonLoose.Charge"],
	})

	Photon = ak.zip(
	{
	 "PT": tree["PhotonLoose.PT"],
   	 "Eta": tree["PhotonLoose.Eta"],
   	 "Phi": tree["PhotonLoose.Phi"],
	})

	MET = ak.zip(
	{
	 "PT": tree["PuppiMissingET.MET"],
   	 "Phi": tree["PuppiMissingET.Phi"],
	})

	

histo = Loop(file_list)
#histo = Lazy(file_list,branches)

print("--- %s seconds ---" % (time.time() - start_time))
