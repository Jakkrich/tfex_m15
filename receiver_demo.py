#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from trading_module.tfex_trading import *
from trading_module.line_notify import *
import sklearn.utils._cython_blas
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

# create the zmq client and listen on port 1234
socket = zmq.Context(zmq.REP).socket(zmq.SUB)
socket.setsockopt_string(zmq.SUBSCRIBE, '')
socket.connect('tcp://127.0.0.1:1234')

##################################################################################################
# Parameters
host = 'localhost:27017'
db_name = 'TFEX'
start_date = '2019-08-01'
initialEquity = 200000
maintenance = 50000
value = 200 #Value per one index point
commission = 50*2 #Round trip
num_candle = 22 #Number of candlesticks in a day
retrieve_rows = max((5+((datetime.today() - datetime.strptime(start_date,'%Y-%m-%d')).days))*num_candle, 1000)

pkl_location = 'D:/BD643/iot/TFEX_model_svc.pkl'

#LineToken = 'Fq4vsJ7rDeDnfaBgucfrbPhn9YHfra2T1rxg4g6sLz4' #Test
#LineToken = '6GFAgWPTwj4xeEHgop9QFvazhU3AOpiSbg5T5wHmOnF' #Price Alerts
LineToken = 'YH8Ed9dB3kTisbFEU7hnub1CTtJnTl7rRXkTNxwXFTq' #Tfex

##################################################################################################

if __name__ == '__main__':
    dump = pd.read_csv('D:/tfex_m15/S50IF_CON_M15_endJul.csv')
    dump['Date/Time'] = pd.to_datetime(dump['Date/Time'], format='%d/%m/%Y %H:%M:%S')
    dump.drop(columns=['Ticker'],inplace=True)
    #dump_to_MongoDB(host, db_name, 'TFEX_update', dump)
    dump_to_MongoDB(host, db_name, 'TFEX_daily_temp', dump)
    
while True:
    message = socket.recv_pyobj()
    try:
        message['Date/Time'] = datetime.strptime(message['Date/Time'], '%d/%m/%Y %H:%M:%S')
    except:
        pass
    print(message)
    start1 = time.time()
    df = retrieve_from_MongoDB(host, db_name, 'TFEX_daily_temp', retrieve_rows)
    df = df.append(message, ignore_index=True)
    df.drop_duplicates(inplace=True)
    try:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d/%m/%Y %H:%M:%S')
    except:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%Y-%m-%d %H:%M:%S')
    dump_to_MongoDB(host, db_name, 'TFEX_daily_temp', df)

    df_raw = df.copy()
    df_withIndicators = indicators_extraction(df_raw)
    df_feature_engineer = feature_engineering(df_withIndicators)
    df_withLabel = labeling(df_feature_engineer)
    
    import joblib
    model = joblib.load(pkl_location)

    predictors = ['EMA10_above_EMA35?','EMA35_above_EMA50?', 'EMA50_above_EMA89?','EMA89_above_EMA120?',
                  'EMA120_above_EMA200?','Close_above_EMA200?','MACD_above_zero?',
                  'MACD_above_upper?','MACD_below_lower?','MACD_above_Signal?',
                  'MACD_slope_1days','MACD_slope_5days','MACD_slope_10days',
                  'Signal_slope_1days','Signal_slope_5days','Signal_slope_10days',
                  'RSI_MA5_higher90?','RSI_MA5_higher70?','RSI_MA5_lower30?',
                  'RSI_MA5_lower10?','RSI_slope_1days','RSI_slope_5days',
                  'RSI_slope_10days','StochK_MA5_higher90?','StochK_MA5_higher70?',
                  'StochK_MA5_lower30?','StochK_MA5_lower10?','StochK_above_StochD?',
                  'StochK_slope_1days','StochK_slope_5days','StochK_slope_10days',
                  'StochD_slope_1days','StochD_slope_5days','StochD_slope_10days',
                  'PDI_above_MDI?','ADX_slope_1days','ADX_slope_5days',
                  'ADX_slope_10days','ADX_above_upper?','ADX_below_lower?','Hold3days?']

    features = df_withLabel[predictors].copy()

    df_withPrediction = df_withLabel.copy()
    df_withPrediction['Prediction'] = model.predict(features.values)
    
    x = df_withPrediction.iloc[-1,-1]
    y = lambda x: 'BUY' if x == 1 else 'SELL' 
    this_predict = y(x)

    ########################################################################################################
    #PnL Calculation starting here
   
    PnL = df_withPrediction.copy()
    PnL['Date/Time'] = pd.to_datetime(PnL['Date/Time'], format='%d/%m/%Y %H:%M:%S')
    PnL = PnL[PnL['Date/Time'] >= start_date]
    PnL.reset_index(drop=True, inplace=True)

    df_PnL = PnL[['Date/Time','Open','High','Low','Close','Prediction']].copy()
    df_PnL.rename(index=str, columns={'Prediction':'Stance'}, inplace=True)
    df_PnL.reset_index(drop=True, inplace=True)
    
    #################################################################################

    #days = max(1,(len(df_PnL)/num_candle))
    days = max(1,((df_PnL['Date/Time'].iloc[-1] - df_PnL['Date/Time'].iloc[0]).days))
    nrows = df_PnL.shape[0]

    #################################################################################
    # Transform to numpy arrays
    Open = np.array(df_PnL['Open'])
    Close = np.array(df_PnL['Close'])
    Stance = np.array(df_PnL['Stance'])
    Position = np.zeros_like(df_PnL['Close'])
    PosNeg = np.zeros_like(df_PnL['Close'])
    PosNeg_cumulative = np.zeros_like(df_PnL['Close'])
    Win = np.zeros_like(df_PnL['Close'])
    Loss = np.zeros_like(df_PnL['Close'])
    Scratch = np.zeros_like(df_PnL['Close'])
    Commission = np.zeros_like(df_PnL['Close'])
    PnL = np.zeros_like(df_PnL['Close'])
    PnL_cumulative = np.zeros_like(df_PnL['Close'])
    Equity = np.zeros_like(df_PnL['Close'])
    Max_Equity = np.zeros_like(df_PnL['Close'])
    Equity_Growth = np.zeros_like(df_PnL['Close'])
    DD = np.zeros_like(df_PnL['Close'])
    MSDD = np.zeros_like(df_PnL['Close'])
    
    # for row[0]
    Position[0] = max(1, math.floor(initialEquity/maintenance))
    PosNeg[0] = (Close[0] - Open[0]) * Stance[0]
    PosNeg_cumulative[0] = PosNeg[0]
    Win[0] = np.where(PosNeg[0]>0,1,0)
    Loss[0] = np.where(PosNeg[0]<0,1,0)
    Scratch[0] = np.where(PosNeg[0]==0,1,0)
    Commission[0] = Position[0]*commission
    PnL[0] = Position[0]*(PosNeg[0]*value) - Commission[0]
    PnL_cumulative[0] = PnL[0]
    Equity[0] = initialEquity + PnL[0]
    Max_Equity[0] = Equity[0]
    Equity_Growth[0] = Equity[0]/initialEquity
    DD[0] = (Max_Equity[0] - Equity[0]) / Max_Equity[0]
    MSDD[0] = 0.0

    for i in range(1,len(Close),1):
        Position[i] = min(50, math.floor(Equity[i-1]/maintenance))
        PosNeg[i] = (Close[i] - Close[i-1]) * Stance[i-1]
        PosNeg_cumulative[i] = PosNeg[i]
        Win[i] = np.where(PosNeg[i]>0,1,0)
        Loss[i] = np.where(PosNeg[i]<0,1,0)
        Scratch[i] = np.where(PosNeg[i]==0,1,0)
        
        if (Stance[i] == Stance[i-1]):
            Commission[i] = 0
        else:
            Commission[i] = Position[i]*commission
            
        PnL[i] = Position[i]*(PosNeg[i]*value) - Commission[i]
        PnL_cumulative[i] = PnL_cumulative[i-1] + PnL[i]
        Equity[i] = Equity[i-1] + PnL[i]
        Max_Equity[i] = max(Equity[i], Max_Equity[i-1])
        Equity_Growth[i] = Equity[i]/initialEquity
        DD[i] = (Max_Equity[i] - Equity[i]) / Max_Equity[i]
        MSDD[i] = max(DD[i],MSDD[i-1])

    df_PnL['Position'] = Position
    df_PnL['PosNeg'] = PosNeg
    df_PnL['PosNeg_cumulative'] = PosNeg_cumulative
    df_PnL['Win'] = Win
    df_PnL['Loss'] = Loss
    df_PnL['Scratch'] = Scratch
    df_PnL['Commission'] = Commission
    df_PnL['PnL'] = PnL
    df_PnL['PnL_cumulative'] = PnL_cumulative
    df_PnL['Equity'] = Equity
    df_PnL['Max_Equity'] = Max_Equity
    df_PnL['Equity_Growth'] = Equity_Growth
    df_PnL['DD'] = DD
    df_PnL['MSDD'] = MSDD

    #################################################################################
    # Performance
    
    print('='*38)
    print('Signal for ' + str(df_withPrediction.loc[df_withPrediction.index[-1],'Date/Time'] )+ ' => ' + this_predict)
    print('='*38)
    
    print('='*31)
    print('TRADE PERFORMANCE - PROFIT/LOSS')
    
    print('='*31)

    print('Trading Days =', str("{0:,.0f}".format(days))+' Days')
    print('Investment =', str("{0:,.0f}".format(initialEquity))+' Baht')
    print('Margin =', str("{0:,.0f}".format(maintenance))+' Baht')
    print('Net Profit so far =', str("{0:,.0f}".format(df_PnL['PnL_cumulative'].iloc[-1]))+' Baht')
    print('%Net Profit so far =', str("{0:,.2f}".format(df_PnL['PnL_cumulative'].iloc[-1]/initialEquity*100))+'%')

    cagr = ((df_PnL['Equity'].iloc[-1] / initialEquity)**(365.0/days))-1
    print('CAGR =', str("{0:,.2f}".format(cagr*100))+'%')

    print('MSDD =', str("{0:,.2f}".format(df_PnL['MSDD'].iloc[-1]*100))+'%')
    print('CAGR/MSDD =', str("{0:,.2f}".format(cagr/df_PnL['MSDD'].iloc[-1])))

    win_ratio = df_PnL['Win'].sum()/(df_PnL['Win'].sum()+df_PnL['Loss'].sum()+df_PnL['Scratch'].sum())
    print('%WIN (based on each record) =', str("{0:,.2f}".format(win_ratio*100))+'%')
    
    stop1 = time.time()
    print('time_consume#1 =', stop1-start1, 'sec')
    
    #################################################################################
    # Line Notification
    
    start2 = time.time()
    main_message = 'Signal for ' + str(df_withPrediction.loc[df_withPrediction.index[-1],'Date/Time'] ) + ' => ' + this_predict
   
    Message1 = '='*20
    Message2 = 'TRADE PERFORMANCE'
    Message3 = '='*20
    Message4 = 'Trading Days = '+str("{0:,.0f}".format(days))+' Days'
    Message5 = 'Investment = '+str("{0:,.0f}".format(initialEquity))+' Baht'
    Message6 = 'Margin = '+str("{0:,.0f}".format(maintenance))+' Baht'
    Message7 = 'Net Profit so far = '+str("{0:,.0f}".format(df_PnL['PnL_cumulative'].iloc[-1]))+' Baht'
    Message8 = '%Net Profit so far = '+str("{0:,.2f}".format(df_PnL['PnL_cumulative'].iloc[-1]/initialEquity*100))+'%'
    Message9 = 'CAGR = '+str("{0:,.2f}".format(cagr*100))+'%'
    Message10 = 'MSDD = '+str("{0:,.2f}".format(df_PnL['MSDD'].iloc[-1]*100))+'%'
    Message11 = 'CAGR/MSDD = '+str("{0:,.2f}".format(cagr/df_PnL['MSDD'].iloc[-1]))
    Message12 = '%WIN = '+str("{0:,.2f}".format(win_ratio*100))+'%'

    img = Image.new('RGB', (230, 210), color = (73, 109, 137))
    font = ImageFont.truetype('C:/Windows/Fonts/Arial.ttf', 13) 
    Message = Message1+'\n'+Message2+'\n'+Message3+'\n'+Message4+'\n'+Message5+'\n'+Message6+'\n'+Message7+'\n'+Message8+'\n'+Message9+'\n'+Message10+'\n'+Message11+'\n'+Message12
    d = ImageDraw.Draw(img)
    d.text((10,10), Message, font=font, fill=(255,255,0))

    img.save('D:/BD643/iot/performance.png')
    
    try:
        #ResponseLine = LineNotify(main_message, LineToken)  
        ResponseLine = LineNotifyImage(main_message, 'D:/BD643/iot/performance.png', LineToken)
    except:
        pass
    
    #################################################################################
    
    stop2 = time.time()
    print('time-consume#2 =', stop2-start2, 'sec\n')


# In[ ]:




