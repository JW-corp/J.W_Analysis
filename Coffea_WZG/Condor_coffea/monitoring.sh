#!/bin/bash


Starttime=$(date)
i=1
echo $Starttime
# Start first

# Automatic learn
while :
do
		condor_q | grep "jwkim"
		echo $timestamp
        if [ $(condor_q | grep "jwkim" |  wc -l ) -eq 0 ]; then
                echo **********************************done...
        fi
        sleep 30
done
Endtime=$(date + %s)

echo "Elapsed time: $(($Endtime - $Starttime))"
