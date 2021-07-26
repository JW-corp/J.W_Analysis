#!/bin/bash

#year='2018'
#channel='eee'
#filename='210531_eee_2018'
#filename='210608_eee_2018_alpha'
#filename='210609_eee_2018_beta'

#filename='210609_eee_2018_SR'
#filename='210609_eee_2018_Zjet'
#filename='210609_eee_2018_tenri'
#filename='210609_eee_2018_conv'


#channel='eemu'
#filename='210609_eem_2018_conv'
#filename='210609_eem_2018_tenri'
#filename='210609_eem_2018_Zjet'
#filename='210609_eem_2018_SR'

#year='2017'
#channel='eee'
#filename='210531_eee_2017'
#filename='210608_eee_2017_alpha'

#filename='210609_eee_2017_conv'
#filename='210609_eee_2017_tenri'
#filename='210609_eee_2017_Zjet'
#filename='210609_eee_2017_SR'


#channel='eemu'
#filename="210531_eemu_2017"

#filename='210609_eem_2017_conv'
#filename='210609_eem_2017_tenri'
#filename='210609_eem_2017_Zjet'
#filename='210609_eem_2017_SR'



## After MC gen-matchingt
#filename='210611_eee_2018_Zjet'
# -- Expired


## ----  Multi-dim-axis
#filename='210618_eee_2018'


## 2018 eee 
#year='2018'
#channel='eee'
#filename='210622_eee_2018'


## 2017 eee 
#year='2017'
#channel='eee'
#filename='210622_eee_2017'


## 2018 eemu 
#year='2018'
#channel='eemu'
#filename='210622_eemu_2018'

## 2017 eemu 
#year='2017'
#channel='eemu'
#filename='210622_eemu_2017'

# 2018 eee exactly 3 Electrons
#year='2018'
#channel='eee'
#filename='210623_eee_2018'

## 2017 eee exactly 3 Electrons
#year='2017'
#channel='eee'
#filename='210623_eee_2017'


# 2018 eee exactly 3 Electrons
# with fakw lepton
year='2018'
channel='eee'
filename='210723_EgammaUL18eee'
region="Baseline"       
#region="Signal"         
#region="CR_ZJets"       
#region="CR_tEnriched"   
#region="CR_conversion"




echo "year: $year channel: $channel filename: $filename"
#python N04_draw.py met $year $channel $region $filename
#python N04_draw.py ele1pt $year $channel $region $filename
#python N04_draw.py ele2pt $year $channel $region $filename




if [ $channel = 'eee' ]
then
python N04_draw.py ele3pt $year $channel $region $filename
fi


python N04_draw.py ele1eta $year $channel $region $filename
python N04_draw.py ele2eta $year $channel $region $filename
python N04_draw.py ele1phi $year $channel $region $filename
python N04_draw.py ele2phi $year $channel $region $filename


if [ $channel = 'eemu' ]
then
python N04_draw.py mupt $year $channel $region $filename
python N04_draw.py mueta $year $channel $region $filename
python N04_draw.py muphi $year $channel $region $filename
fi


python N04_draw.py pho_EB_pt $year $channel $region $filename
python N04_draw.py pho_EB_eta $year $channel $region $filename
python N04_draw.py pho_EB_phi $year $channel $region $filename
python N04_draw.py pho_EB_sieie $year $channel $region $filename
python N04_draw.py pho_EE_pt $year $channel $region $filename
python N04_draw.py pho_EE_eta $year $channel $region $filename
python N04_draw.py pho_EE_phi $year $channel $region $filename
python N04_draw.py pho_EE_sieie $year $channel $region $filename
python N04_draw.py mass $year $channel $region $filename


#python N04_draw.py MT $year $channel $region $filename
