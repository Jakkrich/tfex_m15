import pandas as pd
import numpy as np
import math
import time
import joblib
from datetime import datetime
from trading_module.tfex_trading import *
import matplotlib as mpl
import matplotlib.pyplot as plt

start = time.time()
#################################################################################
#Parameters

start_date = str(input('start date (ex. 2019-01-01) = ')) #'2019-07-01' 
model_name = str(input('model name (svc, xgb, rf, lr) = ')) # svc, xgb, rf, or lr
initialEquity = 200000
maintenance = 50000
value = 200 #Value per one index point
commission = 50*2 #Round trip
num_candle = 22 #Number of candlesticks in a day
retrieve_rows = max((5+((datetime.today() - datetime.strptime(start_date,'%Y-%m-%d')).days))*num_candle, 300)

#################################################################################

df = pd.read_csv('D:/BD643/iot/fromAmiBroker_static/S50IF_CON_M15_fromAmiBroker_static.csv')
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d/%m/%Y %H:%M:%S')
df.reset_index(drop=True,inplace=True)

df_withIndicators = indicators_extraction(df)
df_feature_engineer = feature_engineering(df_withIndicators)
df_withLabel = labeling(df_feature_engineer)

if (model_name=='svc' or  model_name=='s'):
    model = joblib.load('D:/BD643/iot/TFEX_model_svc.pkl')
elif (model_name=='xgb' or model_name =='x'):
    model = joblib.load('D:/BD643/iot/TFEX_model_xgboost.pkl')
elif (model_name=='rf' or model_name=='r'):
    model = joblib.load('D:/BD643/iot/TFEX_model_randomforest.pkl')
elif (model_name =='lr' or model_name=='l'):
    model = joblib.load('D:/BD643/iot/TFEX_model_lr.pkl')

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

features = df_withLabel[predictors]
label = df_withLabel['beLong']

df_withPrediction = df_withLabel.copy()
df_withPrediction['Prediction'] = model.predict(features.values)

df_PnL = df_withPrediction[['Date/Time','Open','High','Low','Close','Prediction']].copy()
df_PnL = df_PnL[df_PnL['Date/Time'] >= start_date]
df_PnL.rename(index=str, columns={'Prediction':'Stance'}, inplace=True)
df_PnL.reset_index(drop=True,inplace=True)

#################################################################################

days = max(1,(df_PnL['Date/Time'].iloc[-1] - df_PnL['Date/Time'].iloc[0]).days)
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

    PnL[i] = Position[i]*(PosNeg[i]*value)-Commission[i]
    PnL_cumulative[i] = PnL_cumulative[i-1]+PnL[i]
    Equity[i] = Equity[i-1]+PnL[i]
    Max_Equity[i] = max(Equity[i], Max_Equity[i-1])
    Equity_Growth[i] = Equity[i]/initialEquity
    DD[i] = (Max_Equity[i]-Equity[i])/Max_Equity[i]
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
print(df_PnL.head())
print(df_PnL.tail())

#################################################################################
# Performance
print('='*47)
print('TRADE PERFORMANCE from ' + start_date + ' till present')
print('='*47)

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

fig = plt.figure(figsize=[12,8])
ax = fig.add_subplot(111)
df_PnL['Equity'].plot(c='r',grid=True,title='Equity Curve');
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

stop = time.time()
print('\nTotal time used',stop-start,'sec')



