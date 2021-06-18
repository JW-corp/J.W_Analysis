import numpy as np
filename = 'Fit_results_2017.npy'
fake_fraction_dict = np.load(filename, allow_pickle="True")[()]

for bins, fr in fake_fraction_dict.items():
    print("{0} : {1}".format(bins,fr))



#Number of FakePhoton  = Number of Data * Fake fraction



'''
PT_1_eta_1 : 20 < pt <30 & |eta| < 1
PT_1_eta_2 : 20 < pt <30 & 1 < |eta| < 1.5
PT_1_eta_3 : 20 < pt <30 & 1.5 < |eta| < 2
PT_1_eta_4 : 20 < pt <30 & 2 < |eta| < 2.5
PT_2_eta_1 : 30 < pt <40 & |eta| < 1
PT_2_eta_2 : 30 < pt <40 & 1 < |eta| < 1.5
PT_2_eta_3 : 30 < pt <40 & 1.5 < |eta| < 2
PT_2_eta_4 : 30 < pt <40 & 2 < |eta| < 2.5
PT_3_eta_1 : 40 < pt <50 & |eta| < 1
PT_3_eta_2 : 40 < pt <50 & 1 < |eta| < 1.5
PT_3_eta_3 : 40 < pt <50 & 1.5 < |eta| < 2
PT_3_eta_4 : 40 < pt <50 & 2 < |eta| < 2.5
PT_4_eta_1 : 50 < pt & |eta| < 1
PT_4_eta_2 : 50 < pt  & 1 < |eta| < 1.5
PT_4_eta_3 : 50 < pt  & 1.5 < |eta| < 2
PT_4_eta_4 : 50 < pt  & 2 < |eta| < 2.5
'''
