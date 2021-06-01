#!/bin/bash

#year='2018'
#channel='eee'
#filename='210531_eee_2018'

#channel='eemu'
#filename="210531_eemu_2018"


year='2017'
#channel='eee'
#filename='210531_eee_2017'

channel='eemu'
filename="210531_eemu_2017"



echo "year: $year channel: $channel filename: $filename"

python N04_draw.py cutflow $year $channel $filename
python N04_draw.py met $year $channel $filename
python N04_draw.py ele1pt $year $channel $filename
python N04_draw.py ele2pt $year $channel $filename

if [ $channel = 'eee' ]
then
python N04_draw.py ele3pt $year $channel $filename
fi


python N04_draw.py ele1eta $year $channel $filename
python N04_draw.py ele2eta $year $channel $filename
python N04_draw.py ele1phi $year $channel $filename
python N04_draw.py ele2phi $year $channel $filename



if [ $channel = 'eemu' ]
then
python N04_draw.py mupt $year $channel $filename
python N04_draw.py mueta $year $channel $filename
python N04_draw.py muphi $year $channel $filename
fi


python N04_draw.py pho_EB_pt $year $channel $filename
python N04_draw.py pho_EB_eta $year $channel $filename
python N04_draw.py pho_EB_phi $year $channel $filename
python N04_draw.py pho_EB_sieie $year $channel $filename
python N04_draw.py pho_EE_pt $year $channel $filename
python N04_draw.py pho_EE_eta $year $channel $filename
python N04_draw.py pho_EE_phi $year $channel $filename
python N04_draw.py pho_EE_sieie $year $channel $filename
python N04_draw.py mass $year $channel $filename
python N04_draw.py MT $year $channel $filename
