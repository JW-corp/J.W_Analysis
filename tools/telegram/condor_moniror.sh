while :
do

	condor_q | grep "jwkim"
	if [ $(condor_q | grep jwkim | awk {'print $12'}) -eq 0 ] ; then
	echo *** done
	condor_q | grep jwkim >> condor_log
	python condor_telegram_monitor.py
	break
	fi
	sleep 60
done
