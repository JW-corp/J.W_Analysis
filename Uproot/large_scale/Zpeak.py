import uproot # for uproot4
import glob
import awkward as ak
import numpy as np
import vector
import time

import matplotlib.pyplot as plt
import mplhep as hep




# Please customize this function for your convenience
def read_data():

	# using file list
	f = open('file_list')
	file_list = [line.rstrip() for line in f]

	# using glob
	#dir_path = "/x4/cms/dylee/Delphes/data/Storage/Second_data/root/signal/condorDelPyOut/*.root"
	#file_list = glob.glob(dir_path)

	# using argparse or argv
	
	
	flist=[]
	for f in file_list:
		flist.append(f + ':Delphes')
	branches = ['Electron.PT' ,'Electron.Eta' ,'Electron.Phi' ,'Electron.Charge' ,'PhotonLoose.PT','PhotonLoose.Eta','PhotonLoose.Phi','MuonLoose.PT' ,'MuonLoose.Eta' ,'MuonLoose.Phi' ,'MuonLoose.Charge' ,'PuppiMissingET.MET' ,'PuppiMissingET.Phi']

	return flist, branches


def Loop(file_list):

	# define array
	histo={}

	# --Start File Loop
	for arrays,doc in uproot.iterate(flist,branches,report=True): #  for Uproot4

		
		print("from: {0}, to: {1} -- Entries: {2}".format(doc.start,doc.stop,len(arrays)))
		
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


		## --- Electron Selection
		cut = (Electron.PT > 20) & (abs(Electron.Eta) < 2.5)
		Electron = Electron[cut]

		# Apply Electron Selection
		cut = ak.num(Electron) >=2
		Electron = Electron[cut]
		Photon	 = Photon[cut]
		Muon	 = Muon[cut]
		MET		 = MET[cut]


		## --- Event Selection

		# Basics of OSSF
		os_cut = (Electron[:,0].Charge + Electron[:,1].Charge == 0)
		Electron = Electron[os_cut]
		Photon	 = Photon[os_cut]
		Muon	 = Muon[os_cut]
		MET		 = MET[os_cut]
		

		Ele1vec = vector.obj(pt=Electron[:,0].PT,eta=Electron[:,0].Eta,phi=Electron[:,0].Phi,mass=0)
		Ele2vec = vector.obj(pt=Electron[:,1].PT,eta=Electron[:,1].Eta,phi=Electron[:,1].Phi,mass=0)

		diele = Ele1vec + Ele2vec



		## --- Flatten and Convert to numpy array
		diele_mass = ak.to_numpy(diele.mass)



		## --- Fill Ntuple
		if len(histo) == 0:
			histo['diele_mass'] = diele_mass
		else:
			histo['diele_mass'] = np.concatenate([histo['diele_mass'],diele_mass])

		print("size of output array: ",len(histo['diele_mass']))

	return histo

def draw(arr, title, start, end, bin):  # Array, Name, x_min, x_max, bin-number


	# ROOT-like format
	plt.style.use(hep.style.ROOT)

	plt.figure(figsize=(8, 5))  # Figure size
	bins = np.linspace(start, end, bin)  # divide start-end range with 'bin' number
	binwidth = (end - start) / bin  # width of one bin

	# Draw histogram
	plt.hist(arr, bins=bins, alpha=0.7, label=title)  # label is needed to draw legend

	plt.xticks(fontsize=16)  # xtick size
	plt.xlabel(title, fontsize=16)  # X-label
	# plt.xlabel('$e_{PT}',fontsize=16) # X-label (If you want LateX format)

	plt.ylabel("Number of Events/(%d GeV)" % binwidth, fontsize=16)  # Y-label
	# plt.ylabel('Number of Events',fontsize=16) # Y-label withou bin-width
	plt.yticks(fontsize=16)  # ytick size

	plt.grid(alpha=0.5)  # grid
	plt.legend(prop={"size": 15})  # show legend
	# plt.yscale('log')	# log scale

	outname_fig = title + ".png"
	#plt.savefig(outname_fig)
	plt.show()  # show histogram
	plt.close()


if __name__ == "__main__":

	start_time = time.time()

	# read data list
	#flist, branches = read_data()

	# start process
	#histo = Loop(flist)
	
	# output ntuple
	#out_name = 'out.npy' # --> Please automate it using str.split() method
	#np.save(out_name,histo,allow_pickle=True)
	
	# read ntuple and draw hist
	ntuple = np.load('out.npy',allow_pickle=True)[()]
	draw(ntuple['diele_mass'],"Z mass",60,120,100)


	
	
	print("--- %s seconds ---" % (time.time() - start_time))
