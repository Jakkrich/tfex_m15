import zmq
import time
import pandas as pd
from pandas_datareader import data
from trading_module.tfex_trading import *
pd.options.display.float_format = '{:,.2f}'.format

socket = zmq.Context(zmq.REP).socket(zmq.PUB)
socket.bind("tcp://*:1234")

start_date = '2019-08-01'
interval = int(input("please mention time interval: "))
data = pd.read_csv('/home/pi/tfex_m15/S50IF_CON_M15_aug_to_present.csv')
data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%d/%m/%Y %H:%M:%S')
data = data[['Date/Time','Open','High','Low','Close','Volume','ZigZag05','ZigZag10','ZigZag15']]
#data = data[data['Date/Time'] >= start_date]
data_dict = data.to_dict(orient='records')

for line in data_dict:
    print(line)
    socket.send_pyobj(line)
    time.sleep(interval)





