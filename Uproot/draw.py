import numpy as np

Wdec = 'Wdec.npy'

Wdec_Ntuple = np.load(Wdec,allow_pickle=True)
Ntuples = Wdec_Ntuple[()]



MT_arr = Ntuples['MT']


import matplotlib.pyplot as plt

plt.hist(MT_arr,bins=200)
plt.grid()
plt.xlabel("MT [GeV]")
plt.xlim(0,200)
plt.show()



np.close(Wedc)
	
