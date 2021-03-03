import uproot3 as up
from uproot_methods import TVector2Array, TLorentzVectorArray
import numpy as np


# Input files
infiles = ["/home/dylee/workspace/WZG_ratio_many/Wdec.root"] # Fix me -- > infiles = glob.glob("FILE_PATH/*.root")

# Ooutput names

sample_names = [ f.split('/')[-1].split(',')[0]  for f in infiles ]


# ---> Processing ....
print("Begin processing.....")



# Read All branch with memory optimization
# Test it using Iterator or uproot3.numentries 
tree = up.lazyarrays(
	infiles,
	"Delphes",
	"*"
)


print("Total events: ",len(tree))



# Get 4vector
def get_4vec(Particle,out_dict):
	PT   = str(Particle) + ".PT"
	Eta  = str(Particle) + ".Eta"
	Phi  = str(Particle) + ".Phi"
	T    = str(Particle) + ".T"

	if str(Particle) != "MissingET":
		out_dict["PT"]  = tree[PT]
		out_dict["Eta"] = tree[Eta]
		out_dict["Phi"] = tree[Phi]
		out_dict["T"]   = tree[T]
	else:
		MET = str(Particle) + ".MET"
		out_dict["MET"] = tree[MET]
		out_dict["Eta"] = tree[Eta]
		out_dict["Phi"] = tree[Phi]
	


Electron_dict = {}
MET_dict = {}

get_4vec("Electron",Electron_dict)
get_4vec("MissingET",MET_dict)
	

print(Electron_dict['PT'])
print(Electron_dict['Eta'])
print(Electron_dict['Phi'])

print(MET_dict['MET'])
print(MET_dict['Eta'])
print(MET_dict['Phi'])
