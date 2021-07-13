import glob
import subprocess

file_list  = glob.glob("/x4/cms/dylee/Delphes/data/Storage/Second_data/root/signal/condorDelPyOut/*.root")


def calc_Nout(maxfile,nfile):
	nfile = maxfile + nfile - 1
	nout = int(nfile / maxfile)
	return(nout)

maxfile=50 # Max number of input files for each run ( argumnet )
nfile=len(file_list) #  Number of total input files
nout  = calc_Nout(maxfile,nfile) # Number of output files


for i in range(nout):
	start = i*maxfile 
	end = start + maxfile 
	
	infiles = (' '.join(file_list[start:end]))

	print("############################## SET: ",i)
	print(infiles)
	
	# Run specific excutable codes
	#args = 'python' + ' '+ 'excute.py' + ' ' + '-option' + ' ' + fn_out + ' '+  infiles
	#subprocess.call(args,shell=True)
