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



## -- Event Selection: at least 1 Electron -- ##

Electron_evt_mask = ak.num(Electron.PT) > 0
Evt_mask = Electron_evt_mask

Electron_sel	=  Electron[Evt_mask]
MET_sel	        =  MET[Evt_mask]
print("Selected Events: ",len(Electron))


## W --MT calculation     <Use Ak0 donwn-grade ...  ak1 with vector method is on developing..>
# ! -------Warning ------>  ak.zip modify the an array type ak0-based jagged-array to ak1-based High level-array
# You should condiser this when you try the ak0-based method eg. uproot method

def Make_T2Vector(pt,phi):
	pt  = ak.to_awkward0(pt)
	phi = ak.to_awkward0(phi)
	return TVector2Array.from_polar(pt,phi)


Electron_T2vec = Make_T2Vector(Electron_sel.PT,Electron_sel.Phi)
MET_T2vec	   = Make_T2Vector(MET_sel.MET,MET_sel.Phi)

Leading_Electron		   = Electron_sel[ak.argmax(Electron_sel.PT,axis=1,keepdims=True)]
Leading_Electron_T2Vec	   = Make_T2Vector(Leading_Electron.PT,Leading_Electron.Phi)

# !--------- Warning ---------! 
#  ak0-based jagged array has shape of (N,) but ak1-based high level array has shape of (N,N)
#  Do not match this dimension with ak.flatten() or array.sum() 
#  This difference is from the different version
#  Therefore, please use the ak.to_awkward0 method to match the difference dimensions from difference ak version
MT_e = np.sqrt(2*ak.to_awkward0(Leading_Electron.PT) *ak.to_awkward0(MET_sel.MET)*(1-np.cos(MET_T2vec.delta_phi(Leading_Electron_T2Vec))))



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

