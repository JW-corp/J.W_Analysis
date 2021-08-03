IsoChg=["from_3_to_8"
,"from_3_to_9"
,"from_3_to_10" 
,"from_3_to_11"
,"from_3_to_12"
,"from_3_to_13"
,"from_4_to_8"
,"from_4_to_9"
,"from_4_to_10"
,"from_4_to_11"
,"from_4_to_12"
,"from_4_to_13"
,"from_5_to_8"
,"from_5_to_9"
,"from_5_to_10"
,"from_5_to_11"
,"from_5_to_12"
,"from_5_to_13"
,"from_6_to_8"
,"from_6_to_9"
,"from_6_to_10"
,"from_6_to_11"
,"from_6_to_12"
,"from_6_to_13"
,"from_7_to_8"
,"from_7_to_9"
,"from_7_to_10"
,"from_7_to_11"
,"from_7_to_12"
,"from_7_to_13"
,"from_8_to_9"
,"from_8_to_10"
,"from_8_to_11"
,"from_8_to_12"
,"from_8_to_13"]



import subprocess
import pathlib

def execute(strs):
	for isochg in IsoChg:
		args = strs + ' ' + isochg

		print(args)
		subprocess.call(args,shell=True)


if __name__ == "__main__":

	execute("python Fit_root.py PT_1_eta_1 100")
	execute("python Fit_root.py PT_1_eta_2 100")
	execute("python Fit_root.py PT_1_eta_3 50")
	execute("python Fit_root.py PT_1_eta_4 25")
	execute("python Fit_root.py PT_2_eta_1 100")
	execute("python Fit_root.py PT_2_eta_2 50")
	execute("python Fit_root.py PT_2_eta_3 25")
	execute("python Fit_root.py PT_2_eta_4 20")
	execute("python Fit_root.py PT_3_eta_1 50")
	execute("python Fit_root.py PT_3_eta_2 50")
	execute("python Fit_root.py PT_3_eta_3 50")
	execute("python Fit_root.py PT_3_eta_4 10")
	execute("python Fit_root.py PT_4_eta_1 25")
	execute("python Fit_root.py PT_4_eta_2 25")
	execute("python Fit_root.py PT_4_eta_3 10")
	execute("python Fit_root.py PT_4_eta_4 10")
