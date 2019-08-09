from tkinter import *
from tkinter import ttk
import pandas as pd
from trading_module.tfex_trading import *

def simulation_publisher():
    with open('D:/tfex_m15/publisher_demo.py','r') as output:
        exec(output.read())
        
mainFrame = Tk()
mainFrame.geometry("350x300")
mainFrame.title("MAIN MENU - fetching data")

ttk.Style().configure("TButton",font=("Times","30","bold"),background="white",foreground="black",justify='center')
   
topgainer_button = ttk.Button(mainFrame,text="PUB",style="TButton",command=simulation_publisher).place(height=200,width=250,x=50,y=50)

mainFrame.mainloop()




