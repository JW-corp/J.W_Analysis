import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('json', type=str,
            help="python N01.submit.py PATH/TO/File.json")
args = parser.parse_args()


metadata=args.json


with open(metadata) as fin:
    datadict = json.load(fin)


os.system('mkdir -p run_condor_log/out run_condor_log/err run_condor_log/log condorOut')

jdl = """universe = vanilla
Executable = N02_run.sh
Transfer_Input_Files = N02_run.sh,N03_run_processor.py,metadata , ../Corrections/corrections.coffea,../Corrections/Pileup/puWeight/pu_weight_RunAB.npy
Output = run_condor_log/out/$ENV(SAMPLE)_$(Cluster)_$(Process).stdout
Error = run_condor_log/err/$ENV(SAMPLE)_$(Cluster)_$(Process).stderr
Log = run_condor_log/log/$ENV(SAMPLE)_$(Cluster)_$(Process).log
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
transfer_output_files=condorOut
Arguments = $ENV(METADATA) $ENV(SAMPLE)
Queue 1"""

jdl_file = open("run.submit", "w") 
jdl_file.write(jdl) 
jdl_file.close() 


for info,dataset in datadict.items():
    #os.system('rm -rf run_condor_log/err/'+info+'*')
    #os.system('rm -rf run_condor_log/log/'+info+'*')
    #os.system('rm -rf run_condor_log/out/'+info+'*')
    os.environ['SAMPLE'] = info
    os.environ['METADATA']   = metadata
    os.system('condor_submit run.submit')
os.system('rm run.submit')
