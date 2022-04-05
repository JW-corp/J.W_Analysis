import telegram_send
import sys


telegram_send.send(messages=["################## Your condor job is finished! ##############"])
contents = open('condor_log','r')
for line in contents:
	line = line.rstrip()
	telegram_send.send(messages=[line])



