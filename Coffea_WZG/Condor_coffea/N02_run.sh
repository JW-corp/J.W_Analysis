#!/bin/bash

echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
echo $(hostname)


source /home/jwkim/Anaconda3/setup.sh 
echo "python run.py --metadata ${1} --dataset ${2}"
python N03_run_processor.py --metadata ${1} --dataset ${2}



echo "show me the location"
pwd
echo "show me the space"
du -h
echo "show me files"
ls -alh


mkdir condorOut
out_files=`find . -maxdepth 1 -type f -name "*.futures"`
mv $out_files condorOut/


echo "### Bye! Bye! GoodJob!!!"
ls -alh condorOut


#ls hists/${2}.futures
#cp hists/${2}.futures ${_CONDOR_SCRATCH_DIR}/${2}.futures
