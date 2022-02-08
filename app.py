import numpy as np
from flask import send_file
from flask import Flask, session,abort,request, jsonify, render_template,redirect,url_for,flash,redirect
import pickle
import pandas as pd
from web3.gas_strategies.time_based import medium_gas_price_strategy
from sklearn.preprocessing import MinMaxScaler as mini
from eth_account.messages import encode_defunct
# import stripe
# import datetime
from web3 import Web3 
from web3 import middleware
import wolframalpha
import wikipedia
# from mnemonic import Mnemonic
# from flask import Flask, flash, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
# mnemo = Mnemonic("english")
# words = mnemo.generate(strength=256)
# seed = mnemo.to_seed(words, passphrase="")
# entropy = mnemo.to_entropy(words)
# import webbrowser
w3 = Web3()
# w3.eth.setGasPriceStrategy(medium_gas_price_strategy)
UPLOAD_FOLDER = 'static/uploads/'
# w3.middleware_onion.add(middleware.time_based_cache_middleware)
# w3.middleware_onion.add(middleware.simple_cache_middleware)
daisy = 'dAIsy'
app = Flask(__name__)
app.secret_key = "vmcxzmvkl;jmgfk;lmvk;lxzkf;vjsfjs1112123jk;djfsdjda;jfkdsa;jfds;fjdskajf;"
pub_key ='pk_live_2pO0yUvt9xKyjAo9rca8Vkc600FWtgJuqZ'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/upload')
def upload():
	return render_template('upload.html')

@app.route('/paid', methods=['POST'])
def paid():
	import os
	from dotenv import load_dotenv
	load_dotenv()

	import json
	from web3.middleware import geth_poa_middleware
	import codecs
	infura_url = "https://rinkeby.infura.io/v3/89f69d97c5c44c35959cc4d15c0f0531"
	web3 = Web3(Web3.HTTPProvider(infura_url))
	web3.middleware_onion.inject(geth_poa_middleware, layer=0)
	abi ='[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"spender","type":"address"},{"indexed":false,"internalType":"uint256","name":"value","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"delegator","type":"address"},{"indexed":true,"internalType":"address","name":"fromDelegate","type":"address"},{"indexed":true,"internalType":"address","name":"toDelegate","type":"address"}],"name":"DelegateChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"delegate","type":"address"},{"indexed":false,"internalType":"uint256","name":"previousBalance","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"newBalance","type":"uint256"}],"name":"DelegateVotesChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"account","type":"address"}],"name":"Paused","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"id","type":"uint256"}],"name":"Snapshot","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":false,"internalType":"uint256","name":"value","type":"uint256"}],"name":"Transfer","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"account","type":"address"}],"name":"Unpaused","type":"event"},{"inputs":[],"name":"DOMAIN_SEPARATOR","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"spender","type":"address"}],"name":"allowance","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"approve","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"snapshotId","type":"uint256"}],"name":"balanceOfAt","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"burn","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"burnFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint32","name":"pos","type":"uint32"}],"name":"checkpoints","outputs":[{"components":[{"internalType":"uint32","name":"fromBlock","type":"uint32"},{"internalType":"uint224","name":"votes","type":"uint224"}],"internalType":"struct ERC20Votes.Checkpoint","name":"","type":"tuple"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"subtractedValue","type":"uint256"}],"name":"decreaseAllowance","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"delegatee","type":"address"}],"name":"delegate","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"delegatee","type":"address"},{"internalType":"uint256","name":"nonce","type":"uint256"},{"internalType":"uint256","name":"expiry","type":"uint256"},{"internalType":"uint8","name":"v","type":"uint8"},{"internalType":"bytes32","name":"r","type":"bytes32"},{"internalType":"bytes32","name":"s","type":"bytes32"}],"name":"delegateBySig","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"delegates","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"flashFee","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC3156FlashBorrower","name":"receiver","type":"address"},{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"flashLoan","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"blockNumber","type":"uint256"}],"name":"getPastTotalSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"},{"internalType":"uint256","name":"blockNumber","type":"uint256"}],"name":"getPastVotes","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"getVotes","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"addedValue","type":"uint256"}],"name":"increaseAllowance","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"}],"name":"maxFlashLoan","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"mint","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"nonces","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"numCheckpoints","outputs":[{"internalType":"uint32","name":"","type":"uint32"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"pause","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"paused","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"value","type":"uint256"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint8","name":"v","type":"uint8"},{"internalType":"bytes32","name":"r","type":"bytes32"},{"internalType":"bytes32","name":"s","type":"bytes32"}],"name":"permit","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"snapshot","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"totalSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"snapshotId","type":"uint256"}],"name":"totalSupplyAt","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"transfer","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"sender","type":"address"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"transferFrom","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"unpause","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
	abi = json.loads(abi)
	address = '0xB600266Ae0EdE4E7826b3f5B8d810a6C1A7469c0'
	tauros = web3.eth.contract(address=address, abi=abi)		
	mint_acct ='0xa71403F82127830fB739E622Cf829D120593FD7F'
	mint_key = web3.toText(text = os.environ['mint_key'])

	
	web3.eth.stake_accnt = stake_accnt
	rec=request.form['Ethereum Address:']
	stake_accnt = rec

	##################################
	##############THIS IS WHERE THE LOGIC OF THE DAO HOLDER WILL GO TO SEE IF THEY CAN ACESS TOKENS########################
		
	nonce =  web3.eth.getTransactionCount(mint_acct)
	mint_tx2 =tauros.functions.transfer(rec,75000000000000000000).buildTransaction({'chainId': 4, 'gas':100000, 'nonce': nonce})
	signed_tx2 = web3.eth.account.signTransaction(mint_tx2, mint_key)
	tx_hash2=web3.eth.sendRawTransaction(signed_tx2.rawTransaction)
	url = "https://rinkeby.etherscan.io/tx/{}".format(web3.toHex(tx_hash2))
	cloud = 'http://127.0.0.1:8000/TaurosDAOCloud'
	return render_template('upload.html',url=url,cloud=cloud) #files=files,url=url filename=filename

import datetime
import pyttsx3
import speech_recognition as sr
import datetime
import shutil
from twilio.rest import Client
from clint.textui import progress
from ecapture import ecapture as ec
from bs4 import BeautifulSoup
import win32com.client as wincl
from urllib.request import urlopen
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from espeakng import ESpeakNG 

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
	engine.say(audio)
	engine.runAndWait()
@app.route('/wishMe')
def wishMe():
	hour = int(datetime.datetime.now().hour)
	if hour>= 0 and hour<12:
		speak("Good Morning !")
  
	elif hour>= 12 and hour<18:
		speak("Good Afternoon!")  
  
	else:
		speak("Good Evening!") 
  
	assname =("TaurosDAO oracle")
	speak("I am your Virtual  Assistant")
	speak(assname)
	return redirect('command.html')
def usrname():
	speak("What should i call you")
	uname = takeCommand()
	speak("Welcome ")
	speak(uname)
	columns = shutil.get_terminal_size().columns
	 
	print("#####################".center(columns))
	print("Welcome ", uname.center(columns))
	print("#####################".center(columns))
	 
	speak("How can i Help you, ")
 
def takeCommand():
	 
	r = sr.Recognizer()
	 
	with sr.Microphone() as source:
		 
		print("Listening...")
		r.pause_threshold = 10
		audio = r.listen(source)
  
	try:
		print("Recognizing...")   
		query = r.recognize_google(audio, language ='en-in')
		print(f"User said: {query}\n")
  
	except Exception as e:
		print(e)   
		print("Unable to Recognize your voice.") 
		return "None"
	 
	return query

import json 

@app.route('/command')
def command():
	return render_template('command.html')
	

@app.route('/answer', methods=['POST'])
def answer():



	command=request.form['Talk to the DAO oracle:']
	# command = jsonify(command)
	#
	while True:
		try:
			app_id = "5PL6G8-KRH7PUAAH5"
			client = wolframalpha.Client(app_id)
			res = client.query(command)
			answers = next(res.results).text 
			answers = str(answers) 
			# answers = jsonify(answers)
			# voice = speak("The answer is "+answers)
		except:
				try:
					command=command.split(' ')
					command = command.join(command[2:]) #input[2:]
					answers = wikipedia.summary(command)  
					answers = jsonify(answers)
					# voice = speak("Searching for Command "+command)
				except:
					answers = 'I dont know the answer' 
					# answers = jsonify(answers)
					# voice = speak(answers)
		break
	# message = {} 
	# data = {}
	# message['message'] = 'command {} , answer {}'.format(command,answer)
	# data['status']= 200
	# data['data'] = message	
	# data = data
	return render_template('command.html',answers=answers)

if __name__ == "__main__":
	app.run(debug=True,host="127.0.0.1",port=6820) #debug=True,host="0.0.0.0",port=50000
	
