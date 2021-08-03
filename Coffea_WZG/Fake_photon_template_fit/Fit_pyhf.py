import pyhf
import numpy as np
import matplotlib.pyplot as plt


label_name = {"PT_1_eta_1": "20 < pt <30 & |eta| < 1"
,"PT_1_eta_2":"20 < pt <30 & 1 < |eta| < 1.5"
,"PT_1_eta_3":"20 < pt <30 & 1.5 < |eta| < 2"
,"PT_1_eta_4":"20 < pt <30 & 2 < |eta| < 2.5"
,"PT_2_eta_1":"30 < pt <40 & |eta| < 1"
,"PT_2_eta_2":"30 < pt <40 & 1 < |eta| < 1.5"
,"PT_2_eta_3":"30 < pt <40 & 1.5 < |eta| < 2"
,"PT_2_eta_4":"30 < pt <40 & 2 < |eta| < 2.5"
,"PT_3_eta_1":"40 < pt <50 & |eta| < 1"
,"PT_3_eta_2":"40 < pt <50 & 1 < |eta| < 1.5"
,"PT_3_eta_3":"40 < pt <50 & 1.5 < |eta| < 2"
,"PT_3_eta_4":"40 < pt <50 & 2 < |eta| < 2.5"
,"PT_4_eta_1":"50 < pt & |eta| < 1"
,"PT_4_eta_2":"50 <pt  & 1 < |eta| < 1.5"
,"PT_4_eta_3":"50 < pt  & 1.5 < |eta| < 2"
,"PT_4_eta_4":"50 < pt  & 2 < |eta| < 2.5"}





def read_data(infile):
	indict = np.load(infile, allow_pickle=True)[()]
	return indict["data_template"], indict["Fake_template"], indict["Real_template"]

# Do not use normalize
def normalize(data_template, template):
	Inte_data = np.sum(data_template["contents"])
	Inte_template = np.sum(template["contents"])
	return template["contents"] * Inte_data / Inte_template


def draw(infile_name, data, fake, real, fit="",fake_frac=""):

	width = abs(real["bins"][0] - real["bins"][1])
	plt.bar(
		real["bins"],
		real["contents"],
		width,
		fill=False,
		edgecolor="darkorange",
		linewidth=2,
		label="Real",
	)
	plt.bar(
		fake["bins"],
		fake["contents"],
		width,
		fill=False,
		edgecolor="royalblue",
		linewidth=2,
		label="Fake",
	)
	plt.scatter(data["bins"], data["contents"], color="black", label="data")

	if not fit == "":
		plt.bar(
			data["bins"],
			fit,
			width,
			fill=False,
			edgecolor="red",
			linewidth=1,
			linestyle='dashed',
			label="fit",
		)
		
		plt.plot([], [], ' ', label="Fake Rate: %.3f" % fake_frac)


	name = infile_name.split('.')[0]
	plt.title(label_name[name])
	plt.xlabel("Photon $\sigma_{i\eta i\eta}$")
	plt.legend()
	plt.grid(alpha=0.5)
	#plt.yscale('log')
	plt.savefig(name + '.png')
	plt.show()


def preproc(data, fake, real):
	index = []
	for i in range(len(data["contents"])):
		# 	print(\
		# 	i,
		# 	data['contents'][i],
		# 	fake['contents'][i],
		# 	real['contents'][i]
		# 	)

		if (
			(data["contents"][i] != 0)
			& (fake["contents"][i] != 0)
			& (real["contents"][i] != 0)
		):

			index.append(i)

	data_prc = list(data["contents"][index])
	fake_proc = list(fake["contents"][index])
	real_proc = list(real["contents"][index])
	bin_proc = list(data["bins"][index])
	index_proc = index

	return data_prc, fake_proc, real_proc, bin_proc, index


def make_json(data, fake, real):
	spec = {
		"channels": [
			{
				"name": "region_one",
				"samples": [
					{
						"data": real,
						"modifiers": [
							#{"name": "scale_real", "type": "shapefactor", "data": None},
							{"name": "scale_real", "type": "normfactor", "data": None},
							{"name": "dummy", "type": "normfactor", "data": None},
						],
						"name": "real",
					},
					{
						"data": fake,
						"modifiers": [
							#{"name": "scale_fake", "type": "shapefactor", "data": None}
							{"name": "scale_fake", "type": "normfactor", "data": None}
						],
						"name": "fake",
					},
				],
			},
		],
		"measurements": [
			{
				"config": {
					"parameters": [{"name": "dummy", "fixed": True}],
					"poi": "dummy",
				},
				"name": "shapefactor example",
			}
		],
		"observations": [
			{"data": data, "name": "region_one"},
		],
		"version": "1.0.0",
	}
	return spec


def get_parameter_names(model):
	labels = []
	for parname in model.config.par_order:
		for i_par in range(model.config.param_set(parname).n_parameters):
			labels.append(
				f"{parname}[bin_{i_par}]"
				if model.config.param_set(parname).n_parameters > 1
				else parname
			)
	return labels



def set_limit(infile_name):
	isEB_sieie = 0.01015
	isEE_sieie = 0.0326

	limit =0
	Eta_index = int(infile_name.split('.')[0].split('_')[-1])
	if (Eta_index == 1) or (Eta_index == 2):
		print("is Barrel!")
		return isEB_sieie
	elif (Eta_index == 3) or (Eta_index == 4):
		print("is Endcap!")
		return isEE_sieie
	else:
		raise ValueError


def fake_fraction(data,fake,limit):
	
	data_sum=0.
	fake_sum=0.
	for i in range(len(data['bins'])):
		if data['bins'][i] >= limit: break;
		data_sum += data['contents'][i]
		fake_sum += fake['contents'][i]

	return fake_sum / data_sum

	
	



if __name__ == "__main__":

	## Basic Setup ##

	# Setup data file
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('infile_name', type=str,
			help="python Fit.py PT_1_eta_1.npy")
	args = parser.parse_args()
	





	infile_name = args.infile_name
	data_template, Fake_template, Real_template = read_data(infile_name)

	# Normalize
	# Fake_template["contents"] = normalize(data_template, Fake_template)
	# Real_template["contents"] = normalize(data_template, Real_template)

	# Draw
	# draw(data_template,Fake_template,Real_template,infile_name)

	# Preprocess: Non zero bin
	proc_data, proc_fake, proc_real, proc_bin, idx = preproc(
		data_template, Fake_template, Real_template
	)

	## Fit ##

	# Make Working Spce
	spec = make_json(proc_data, proc_fake, proc_real)
	ws = pyhf.Workspace(spec)
	model = ws.model()

	# Observable
	data = ws.data(model)

	# MLE Fit
	fit_results = pyhf.infer.mle.fit(data, model)

	# Calculate Scale Factor
	SF_real = 0
	SF_fake = 0
	
	for i, label in enumerate(get_parameter_names(model)):
		print(f"{label}: {fit_results[i]}")

		if label.startswith("scale_real"):
			SF_real = fit_results[i]
		elif label.startswith("scale_fake"):
			SF_fake = fit_results[i]

	print("Real template SF: ", SF_real)
	print("Fake template SF: ", SF_fake)

	Fake_template['contents'] = Fake_template['contents'] * SF_fake
	Real_template['contents'] = Real_template['contents'] * SF_real
	Fit_data = Fake_template['contents'] + Real_template['contents']
	## Apply Scale Factor
	#start = idx[0]
	#end = idx[-1] + 1

	#Fit_data = np.zeros(len(Fake_template["contents"]))
	#Fit_data[start:end] = (
	#	Fake_template["contents"][start:end] * SF_fake
	#	+ Real_template["contents"][start:end] * SF_real
	#)
	#
	#Fake_template['contents'][start:end] = Fake_template['contents'][start:end] * SF_fake
	#Real_template['contents'][start:end] = Real_template['contents'][start:end] * SF_real
	limit = set_limit(infile_name)
	fake_frac = fake_fraction(data_template,Fake_template,limit)
	draw(infile_name, data_template, Fake_template, Real_template, Fit_data,fake_frac)
