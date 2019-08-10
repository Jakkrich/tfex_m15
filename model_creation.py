import numpy as np
import pandas as pd
from datetime import datetime
from trading_module.tfex_trading import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import sqrt
import math
import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option("display.max_rows", 1000)

ticker = 'S50IF_CON'
df_raw = pd.read_csv('/home/pi/tfex_m15/S50IF_CON_M15_train_tillend2018.csv')
df_raw['Date/Time'] = pd.to_datetime(df_raw['Date/Time'],format='%d/%m/%Y %H:%M:%S')
df_raw.drop(columns=['Ticker'], inplace=True)

if __name__ == '__main__':
    dump_to_MongoDB('localhost:27017', 'TFEX', 'TFEX_train', df_raw)
    df_withIndicators = indicators_extraction(df_raw)
    df_feature_engineer = feature_engineering(df_withIndicators)
    df_withLabel = labeling(df_feature_engineer)
    
train_data_range = round(len(df_withLabel.index)*0.9,0)
train_df_withLabel = df_withLabel.iloc[:int(train_data_range),:]
validate_df_withLabel = df_withLabel.iloc[int(train_data_range):,:]   

model = LogisticRegression(solver='liblinear', random_state=2019)
#model = DecisionTreeClassifier(random_state=2019)
#model = RandomForestClassifier(n_estimators=200, random_state=2019)
#model = GradientBoostingClassifier(n_estimators=200, random_state=2019)
#model = SVC(kernel='rbf', random_state=2019)
#model = xgboost.XGBClassifier()

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

features = train_df_withLabel[predictors]
label = train_df_withLabel['beLong']

if __name__ == '__main__':
    fitting_evaluating(ticker=ticker, features_df=features, label_series=label, model=model, 
                       partition_split=3, test_size=0.25, seed=2019)
    
joblib.dump(model, '/home/pi/tfex_m15/TFEX_model_lr.pkl') 

validate_features = validate_df_withLabel[predictors]
validate_label = validate_df_withLabel['beLong']

validate_predicted = model.predict(validate_features.values)

print('-'*60)
print('Confusion Matrix Report - Validated data set')
print('-'*60)
print(classification_report(validate_label.values, validate_predicted))

#################################################################################

features = df_withLabel[predictors]
df_withPrediction = df_withLabel.copy()
df_withPrediction['Prediction'] = model.predict(features.values)

df_pred = df_withPrediction[['Date/Time','Open','High','Low','Close','Prediction']].copy()
df_pred['Date/Time'] = pd.to_datetime(df_pred['Date/Time'], format='%d/%m/%Y %H:%M:%S')
df_pred.rename(index=str, columns={'Prediction':'Stance'}, inplace=True)
df_pred.reset_index(drop=True, inplace=True)

#################################################################################

initialEquity = 200000
maintenance = 50000
value = 200 #Value per one index point
commission = 50*2 #Round trip

#################################################################################

days = ((df_pred['Date/Time'].iloc[-1] - df_pred['Date/Time'].iloc[0]).days)
nrows = df_pred.shape[0]

#################################################################################

# Transform to numpy arrays
Open = np.array(df_pred['Open'])
Close = np.array(df_pred['Close'])
Stance = np.array(df_pred['Stance'])

Position = np.zeros_like(df_pred['Close'])
PosNeg = np.zeros_like(df_pred['Close'])
PosNeg_cumulative = np.zeros_like(df_pred['Close'])
Win = np.zeros_like(df_pred['Close'])
Loss = np.zeros_like(df_pred['Close'])
Scratch = np.zeros_like(df_pred['Close'])
Commission = np.zeros_like(df_pred['Close'])
PnL = np.zeros_like(df_pred['Close'])
PnL_cumulative = np.zeros_like(df_pred['Close'])
Equity = np.zeros_like(df_pred['Close'])
Max_Equity = np.zeros_like(df_pred['Close'])
Equity_Growth = np.zeros_like(df_pred['Close'])
DD = np.zeros_like(df_pred['Close'])
MSDD = np.zeros_like(df_pred['Close'])

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
    
df_pred['Position'] = Position
df_pred['PosNeg'] = PosNeg
df_pred['PosNeg_cumulative'] = PosNeg_cumulative
df_pred['Win'] = Win
df_pred['Loss'] = Loss
df_pred['Scratch'] = Scratch
df_pred['Commission'] = Commission
df_pred['PnL'] = PnL
df_pred['PnL_cumulative'] = PnL_cumulative
df_pred['Equity'] = Equity
df_pred['Max_Equity'] = Max_Equity
df_pred['Equity_Growth'] = Equity_Growth
df_pred['DD'] = DD
df_pred['MSDD'] = MSDD

#################################################################################
# Performance
print('='*31)
print('TRADE PERFORMANCE')
print('='*31)

print('Trading Days =', days, 'Days')
print('Investment =', initialEquity, 'Baht')
print('Net Profit so far =', str(round(df_pred['PnL_cumulative'].iloc[-1],4))+' Baht')
print('%Net Profit so far =', str(round(df_pred['PnL_cumulative'].iloc[-1]/initialEquity*100,4))+'%')

cagr = ((df_pred['Equity'].iloc[-1] / initialEquity)**(365.0/days)) - 1
print ('CAGR =', str(round(cagr*100,4))+'%')

print('MSDD =', str(round(df_pred['MSDD'].iloc[-1]*100,4))+'%')
print('CAGR/MSDD =', str(round(cagr/df_pred['MSDD'].iloc[-1],4)))

win_ratio = df_pred['Win'].sum()/(df_pred['Win'].sum()+df_pred['Loss'].sum()+df_pred['Scratch'].sum())
print('%WIN (based on each record) =', str(win_ratio*100)+'%')

fig = plt.figure(figsize=[12,8])
df_pred['Equity'].plot(grid=True,color='r');
ax = fig.add_subplot(111)
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

