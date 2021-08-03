import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



basedir = '/x5/cms/jwkim/gitdir/JWCorp/JW_analysis/Coffea_WZG/Condor_coffea/Fitting_2018_ClosureTest'




PT_eta_bin = [
"PT_1_eta_1"
,"PT_1_eta_2"
,"PT_1_eta_3"
,"PT_1_eta_4"
,"PT_2_eta_1"
,"PT_2_eta_2"
,"PT_2_eta_3"
,"PT_2_eta_4"
,"PT_3_eta_1"
,"PT_3_eta_2"
,"PT_3_eta_3"
,"PT_4_eta_1"
,"PT_4_eta_2"
,"PT_4_eta_3"
,"PT_4_eta_4"
]



for PT_eta_bin in PT_eta_bin:

	print("### start: ",PT_eta_bin)
	# path for known fake fraction
	npy_path = basedir+'/FakeTemplate/'   + PT_eta_bin + '.npy'
	npy = np.load(npy_path,allow_pickle=True)[()]
	keys   = list(npy['known_fake_fraction'].keys())
	mcTrue = list(npy['known_fake_fraction'].values())
	
	
	# path for fitted fake fraction
	df_path = basedir + '/' + PT_eta_bin+ '/log.csv.new'
	df = pd.read_csv(df_path,delimiter='\s+')
	
	
	# make X and Y bins
	xlow=[]
	yhigh=[]
	for i,name in enumerate(keys):
		xlow.append(int(name.split('_')[1]))
		yhigh.append(int(name.split('_')[3]))
	
	# make mcTrue fackefrac dict
	mcTrue_dict={}	
	for key,val in zip(keys,mcTrue):
		mcTrue_dict[key] = val

	# make Fitted fakefrac dict
	fitted_dict={}	
	for key,val in zip(df['SB'],df['Fake_fraction']):
		fitted_dict[key] = val


	# sync together and make Unc dict
	unc_dict={}
	for key,val in mcTrue_dict.items():
		try:
			if (fitted_dict[key] < 0) or (fitted_dict[key] > 1):
				fitted_dict[key] = np.nan

		except:
			fitted_dict[key] = np.nan
	
		unc_dict[key] = abs(val - fitted_dict[key])/val

	bin_mcTrue=[]
	print("######  mcTrue_dict", len(mcTrue_dict))
	for key in keys:
		#print(key, mcTrue_dict[key])
		bin_mcTrue.append(mcTrue_dict[key])

	bin_fitted=[]
	print("######  fitted_dicta", len(fitted_dict))
	for key in keys:
		#print(key,fitted_dict[key])
		bin_fitted.append(fitted_dict[key])

	bin_unc=[]
	print("######  Unc arr",len(unc_dict))
	for key in keys:
		#print(key,unc_dict[key])
		bin_unc.append(unc_dict[key])

	

	# Build data frame
	df = pd.DataFrame.from_dict(np.array([xlow,yhigh,bin_unc]).T)
	df.columns = ['low IsoChg','high IsoChg','unc_closure']
	
	# replace Inf to Nan
	df = df.replace([np.inf, -np.inf], np.nan)
	
	# dataframe for print-out
	df_for_show = df.copy()
	df_for_show['Fake fraction true'] = np.array(bin_mcTrue)
	df_for_show['Fake fraction fitted'] = np.array(bin_fitted)
	print(df_for_show)
	
	
	# df format for heatmap
	df_pivoted = df.pivot('low IsoChg','high IsoChg','unc_closure')
	
	
	# draw heat,ap
	plt.rcParams["figure.figsize"] = (8,8)
	plt.rcParams.update({'font.size': 15})
	sns.heatmap(df_pivoted,annot=True,linewidths=0.5,cmap='Blues',fmt='.4f')
	
	
	# save image
	outname = 'fake_fraction_2d_' + PT_eta_bin + '.png'
	plt.savefig(outname)
	plt.close()
	#plt.show()
