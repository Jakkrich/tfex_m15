from tkinter import *
from tkinter import ttk
from trading_module.tfex_trading import *
from trading_module.line_notify import *
import pandas as pd
import numpy as np
import pymongo
from math import sqrt
import math
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import imgkit
#import emoji
from PIL import Image, ImageDraw, ImageFont
import zmq
from json import loads
import sklearn.utils._cython_blas

def simulation_receiver():
    with open('D:/tfex_m15/receiver_demo.py','r') as output:
        exec(output.read())
        
mainFrame = Tk()
mainFrame.geometry("350x300")
mainFrame.title("MAIN MENU - prediction")

ttk.Style().configure("TButton",font=("Times","30","bold"),background="white",foreground="black",justify='center')
   
topgainer_button = ttk.Button(mainFrame,text="SUB",style="TButton",command=simulation_receiver).place(height=200,width=250,x=50,y=50)

mainFrame.mainloop()




