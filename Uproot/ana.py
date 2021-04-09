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


## W --MT calculation   < Trick is used .. not stable.. ak1 with vector method is on developing..>
			# --------------- Warning --------------#
#	ak.zip modify the an array type ak0-based jagged-array to ak1-based High level-array
#	In principle, you cannot use the ak0-based method including uproot_method
#	But, there is trick: You can use one-dimensional array with uproot_method
#
#	If you realy want to use multi-dimensional array with uproot_method
#	Please use the ak.to_awkward0 method like below:
#
#	def Make_T2Vector(pt,phi):
#		pt  = ak.to_awkward0(pt)
#		phi = ak.to_awkward0(phi)
#		return TVector2Array.from_polar(pt,phi)
#
#	Beware ths fact that ak0 array is shown as 1-dimensional array even if is is actually 2-dimentional array
#	ak0arr.shape -> (N,)



MET_T2vec				   = TVector2Array.from_polar(MET_sel.MET,MET_sel.Phi) 

Leading_Electron		   = Electron_sel[ak.argmax(Electron_sel.PT,axis=1,keepdims=True)]
Leading_Electron_T2Vec	   = TVector2Array.from_polar(Leading_Electron.PT,Leading_Electron.Phi)

MT_e = np.sqrt(2*Leading_Electron.PT *MET_sel.MET * (1-np.cos(abs(MET_T2vec.delta_phi(Leading_Electron_T2Vec)))))



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
