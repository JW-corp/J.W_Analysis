import telegram_send
import subprocess
import time


def spot_run_file():

	# Need to check token command -> Depend on server setting
	token="condor_q | grep jwkim | awk {'print $6'}"
	output = subprocess.check_output(token,shell=True)
	output = output.decode('utf-8')

	output = output[0].split("\\n")[0]

	# Finish, send telegram-message
	if output == '0':
		print("Finish")
		telegram_send.send(messages=["################## Your condor job is finished! ##############"])
		exit()
	# Still running
	elif output=='_': # Need to check -> Depend on server setting
		print("Running")

	# Wrong command
	else:
		print("wrong... parsing")
		


while True:
	spot_run_file()
	time.sleep(60)


