import pyhf
import numpy as np
import matplotlib.pyplot as plt






def read_data(infile):
	indict = np.load(infile,allow_pickle=True)[()]
	return indict['data_template'], indict['Fake_template'], indict['Real_template']


def normalize(data_template,template):
	Inte_data = np.sum(data_template['contents'])
	Inte_template = np.sum(template['contents'])
	return template['contents'] * Inte_data / Inte_template



def draw(infile_name,data,fake,real,fit=""):

	width=0.0004
	plt.bar(real['bins'],real['contents'],width,fill=False,edgecolor='darkorange',label='Real')
	plt.bar(fake['bins'],fake['contents'],width,fill=False,edgecolor='royalblue',label='Fake')
	plt.scatter(data['bins'],data['contents'],color='black',label='data')

	if not fit=="":
		plt.bar(data['bins'],fit,width,fill=False,edgecolor='crimson',label='fit')

	plt.xlabel(infile_name)
	plt.legend()
	plt.grid()
	plt.show()


def preproc(data,fake,real):
	index=[]
	for i in range(len(data['contents'])):
	#	print(\
	#	i,
	#	data['contents'][i],
	#	fake['contents'][i],
	#	real['contents'][i]
	#	)

		if (data['contents'][i] != 0) &\
		   (fake['contents'][i] != 0) &\
		   (real['contents'][i] != 0):
			
			index.append(i)

	start = index[0]
	end   = index[-1] + 1

	data_prc  = list(data['contents'][start:end])
	fake_proc = list(fake['contents'][start:end])
	real_proc = list(real['contents'][start:end])
	bin_proc  = list(data['bins'][start:end])
	index_proc= index

	return data_prc,fake_proc,real_proc,bin_proc,index



def make_json(data,fake,real):
	spec = {
    "channels": [
        {
            "name": "region_one",
            "samples": [
                {
                    "data": real,
                    "modifiers": [
                        {"name": "scale_real", "type": "shapefactor", "data": None},
                        {"name": "dummy", "type": "normfactor", "data": None},
                    ],
                    "name": "real",
                },
                {
                    "data": fake,
                    "modifiers": [
                        {"name": "scale_fake", "type": "shapefactor", "data": None}
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




if __name__ == '__main__':


		## Basic Setup ##

	# Setup data file
	infile_name = 'PT_1_eta_1.npy'
	data_template, Fake_template, Real_template = read_data(infile_name)
	
	# Normalize
	Fake_template['contents'] = normalize(data_template,Fake_template)
	Real_template['contents'] = normalize(data_template,Real_template)
		 
	
	
	# Draw
	#draw(data_template,Fake_template,Real_template,infile_name)
	
	# Preprocess: Non zero bin
	proc_data, proc_fake, proc_real, proc_bin,idx = preproc(data_template,Fake_template,Real_template)
	

		## Fit ##
	
	# Make Working Spce
	spec = make_json(proc_data, proc_fake, proc_real)
	ws = pyhf.Workspace(spec)
	model = ws.model()

	# Observable
	data = ws.data(model)
	
	# MLE Fit
	fit_results = pyhf.infer.mle.fit(data,model)
	
	# Calculate Scale Factor
	SF_real = []
	SF_fake = []

	for i,label in enumerate(get_parameter_names(model)):
	    print(f"{label}: {fit_results[i]}")
	
	    if label.startswith("scale_real"):
	        SF_real.append(fit_results[i])
	    elif label.startswith("scale_fake"):
	        SF_fake.append(fit_results[i])

	print("Real template SF: ",SF_real)
	print("Fake template SF: ",SF_fake)

	start = idx[0]
	end   = idx[-1] + 1

	Fit_data = np.zeros(len(Fake_template['contents']))
	Fit_data[start:end] = Fake_template['contents'][start:end] * SF_fake +  Real_template['contents'][start:end] * SF_real
	
	draw(infile_name,data_template,Fake_template,Real_template,Fit_data)
	
	
	


	

