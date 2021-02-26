import uproot as up
from uproot_methods import TVector2Array, TLorentzVectorArray
import numpy as np
import awkward as ak

## -- Input: Tree and Branch  ( Need memory optimization )
infile = "/home/dylee/workspace/WZG_ratio_many/Wdec.root"
sample_name = infile.split('/')[-1].split('.')[0]

print("Begin processing.....{0}".format(sample_name))


f = up.open(infile)
tree = f['Delphes']

Electron_branch =  tree['Electron']
MET_branch = tree['MissingET']

Electron_pt  = Electron_branch['Electron.PT'].array()
Electron_phi = Electron_branch['Electron.Phi'].array()
Electron_T2vec   = TVector2Array.from_polar(Electron_pt,Electron_phi)

MET_met  = MET_branch['MissingET.MET'].array()
MET_phi = MET_branch['MissingET.Phi'].array()
MET_T2vec   = TVector2Array.from_polar(MET_met,MET_phi)


## -- Event Selection: at least 1 Electron -- ##

Electron_evt_mask = Electron_pt.counts > 0
Evt_mask = Electron_evt_mask 
 
Electron_sel_pt    =  Electron_pt[Evt_mask]
Electron_sel_phi   =  Electron_phi[Evt_mask]
Electron_sel_T2vec =  Electron_T2vec[Evt_mask]

MET_sel_met		 = MET_met[Evt_mask]
MET_sel_phi		 = MET_phi[Evt_mask]
MET_sel_T2vec	 = MET_T2vec[Evt_mask]


## -- Leading Electron physics variables -- ##

Leading_Electron_pt    = Electron_sel_pt[Electron_sel_pt.argmax()]
Leading_Electron_phi   = Electron_sel_phi[Electron_sel_pt.argmax()]
Leading_Electron_T2vec = Electron_sel_T2vec[Electron_sel_pt.argmax()]

## -- MT calculation -- ##
MT_e = np.sqrt(2*Leading_Electron_pt * MET_sel_met*(1-np.cos(MET_sel_T2vec.delta_phi(Leading_Electron_T2vec))))


## -- Output hist & Ntuple(will be added)  -- ##
#import matplotlib.pyplot as plt
#h1_MT = plt.hist(MT_e.sum(),bins=200)
#plt.xlim(0,200)
#plt.show()



outname = sample_name + ".npy"

np.save(outname,MT_e.sum())

print("End processing.... Bye Bye!")
