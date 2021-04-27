#!/bin/bash

[[ -d "210427_FakeTemplate" ]] || mkdir 210427_FakeTemplate

# -1- run Fake template
for i in "Egamma_Run2018A_280000" "Egamma_Run2018A_40000" "Egamma_Run2018A_50000" "Egamma_Run2018B_280000" "Egamma_Run2018B_50000" "Egamma_Run2018D_270000" "Egamma_Run2018D_280000" "Egamma_Run2018D_40000" "Egamma_Run2018D_50000"
do
echo $i
python N03_run_processor_fake.py --metadata metadata/skim/Egamma_skimEleIdPt20RunABD.json --dataset $i --year 2018 --isdata True
done

mv *.futures 210427_FakeTemplate




# -2- run Real template
python N03_run_processor_real.py --metadata metadata/2018/WZG.json --dataset WZG  --year 2018

mv *.futures 210427_FakeTemplate




# -3- run Data template
for i in "Egamma_Run2018A_280000" "Egamma_Run2018A_40000" "Egamma_Run2018A_50000" "Egamma_Run2018B_280000" "Egamma_Run2018B_50000" "Egamma_Run2018D_270000" "Egamma_Run2018D_280000" "Egamma_Run2018D_40000" "Egamma_Run2018D_50000"
do
echo $i
python N03_run_processor_fake.py --metadata metadata/skim/Egamma_skimEleIdPt20RunABD.json --dataset $i --year 2018 --isdata True
done

mv *.futures 210427_FakeTemplate



