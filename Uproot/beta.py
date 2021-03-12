import uproot3 as up
from uproot_methods import TVector2Array, TLorentzVectorArray
import numpy as np
import awkward1 as ak


# Input files
infiles = ["/home/dylee/workspace/WZG_ratio/Wdec.root"] # Fix me -- > infiles = glob.glob("FILE_PATH/*.root")

# Ooutput names

sample_names = [ f.split('/')[-1].split('.')[0]  for f in infiles ]


# ---> Processing ....
print("Begin processing.....")



# Read All branch with memory optimization
# Test it using Iterator or uproot3.numentries 
tree = up.lazyarrays(
	infiles,
	"Delphes",
	"*",
)


print("Total events: ",len(tree))



print("Branch to array....")
Electron = ak.zip({
	"PT"		: tree["Electron.PT"],
	"Eta"		: tree["Electron.Eta"],
	"Phi"		: tree["Electron.Phi"],
	"T"			: tree["Electron.T"],
	"Charge"	: tree["Electron.Charge"]
})

Photon = ak.zip({
	"PT"		: tree["Photon.PT"],
	"Eta"		: tree["Photon.Eta"],
	"Phi"		: tree["Photon.Phi"],
	"T"			: tree["Photon.T"],
})


MET = ak.zip({
	"MET"		: tree["MissingET.MET"],
	"Eta"		: tree["MissingET.Eta"],
	"Phi"		: tree["MissingET.Phi"],
})


print("Ele: ",Electron,len(Electron))
print("Pho: ",Photon,len(Photon))
print("MET: ",MET,len(MET))

## -- Event Selection: at least 1 Electron -- ##

Electron_evt_mask = ak.num(Electron.PT) > 0
Evt_mask = Electron_evt_mask

Electron	=  Electron[Evt_mask]
MET	        =  MET[Evt_mask]
print("Selected Events: ",len(Electron))


## -- <Use Ak0 method>
def Make_T2Vector(pt,phi):
	pt  = ak.to_awkward0(pt)
	phi = ak.to_awkward0(phi)
	return TVector2Array.from_polar(pt,phi)



## -- Leading Electron physics variables -- ##
Electron_T2vec = Make_T2Vector(Electron.PT,Electron.Phi)
MET_T2vec	   = Make_T2Vector(MET.MET,MET.Phi)

Leading_Electron		   = Electron[ak.argmax(Electron.PT,axis=1,keepdims=True)]
Leading_Electron_T2Vec	   = Electron_T2vec[ak.argmax(Electron.PT,axis=1)] 


## -- <Use Ak0 method>
MT_e = np.sqrt(2*Leading_Electron.PT * MET.MET)*(1-np.cos(MET_T2vec.delta_phi(Leading_Electron_T2Vec)))

print("W MT: ", MT_e)


## -- Output hist & Ntuple(will be added)  -- ##
#import matplotlib.pyplot as plt
#h1_MT = plt.hist(ak.flatten(MT_e)),bins=200)
#plt.xlim(0,200)
#plt.show()

histo = {}
histo['MT'] = ak.flatten(MT_e)


outname = sample_names[0] + ".npy" # Just one file

print(histo.keys())

np.save(outname,histo)

print("End processing.... Bye Bye!")

