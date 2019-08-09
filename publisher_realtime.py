#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import zmq
import time
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

##########################################################################################
# create the zmq context and socket and bind the socket to port 1234
socket = zmq.Context(zmq.REP).socket(zmq.PUB)
socket.bind("tcp://*:1234")
##########################################################################################

class MyHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = datetime.now()

    def createSSHClient(server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client
    
    def on_modified(self, event):
        if datetime.now() - self.last_modified < timedelta(seconds=5):
            return
        else:
            self.last_modified = datetime.now()
        #print(f'Event type: {event.event_type}  path : {event.src_path}')
        
        import pandas as pd
        from kafka import KafkaProducer
        from json import dumps
        import paramiko
        from scp import SCPClient
        
        def createSSHClient(server, port, user, password):
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(server, port, user, password)
            return client
        
        #ssh = createSSHClient('192.168.1.104', 22, 'pi', 'Nikedymo2')
        #scp = SCPClient(ssh.get_transport())
        #scp.put('D:/BD643/iot/fromAmiBroker_realtime/S50IF_CON_M15_fromAmiBroker_realtime.csv', '/home/pi/Desktop/S50IF_CON_M15_fromAmiBroker_realtime.csv')
        
        time.sleep(1)
        
        path_csv = 'D:/BD643/iot/fromAmiBroker_realtime/S50IF_CON_M15_fromAmiBroker_realtime.csv'
        df_raw = pd.read_csv(path_csv)
        df_raw = df_raw.iloc[-1:,:].copy()
        df_raw.drop(columns=['Ticker'],inplace=True)
        df_raw.reset_index(drop=True, inplace=True)
        df_raw_dict = df_raw.to_dict(orient='records')

        for line in df_raw_dict:
            socket.send_pyobj(line)
            print(line)
            time.sleep(1)
##########################################################################################
            
if __name__ == '__main__':
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='D:/BD643/iot/fromAmiBroker_realtime/', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

