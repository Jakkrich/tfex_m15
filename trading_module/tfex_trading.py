def indicators_extraction(df):
    import talib
    ema10 = talib.EMA(df['Close'], timeperiod=10)
    ema35 = talib.EMA(df['Close'], timeperiod=35)
    ema50 = talib.EMA(df['Close'], timeperiod=50)
    ema89 = talib.EMA(df['Close'], timeperiod=89)
    ema120 = talib.EMA(df['Close'], timeperiod=120)
    ema200 = talib.EMA(df['Close'], timeperiod=200)
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    rsi = talib.RSI(df['Close'], timeperiod=10)
    adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=10)
    mdi = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=10)
    pdi = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=10)
    roc = talib.ROC(df['Close'], timeperiod=10)
    stochK, stochD = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=9, slowk_period=3, slowd_period=3)
    cci = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=10)
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=10)
    
    df['ZigZag05_1'] = df['ZigZag05'].shift(1)
    df['ZigZag10_1'] = df['ZigZag10'].shift(1)
    df['ZigZag15_1'] = df['ZigZag15'].shift(1)
    
    df['EMA10'] = ema10
    df['EMA10_1'] = df['EMA10'].shift(1)
    df['EMA10_2'] = df['EMA10'].shift(2)
    df['EMA10_3'] = df['EMA10'].shift(3)
    df['EMA10_4'] = df['EMA10'].shift(4)
    df['EMA10_5'] = df['EMA10'].shift(5)
    df['EMA10_6'] = df['EMA10'].shift(6)
    df['EMA10_7'] = df['EMA10'].shift(7)
    df['EMA10_8'] = df['EMA10'].shift(8)
    df['EMA10_9'] = df['EMA10'].shift(9)
    df['EMA10_10'] = df['EMA10'].shift(10)

    df['EMA35'] = ema35
    df['EMA35_1'] = df['EMA35'].shift(1)
    df['EMA35_2'] = df['EMA35'].shift(2)
    df['EMA35_3'] = df['EMA35'].shift(3)
    df['EMA35_4'] = df['EMA35'].shift(4)
    df['EMA35_5'] = df['EMA35'].shift(5)
    df['EMA35_6'] = df['EMA35'].shift(6)
    df['EMA35_7'] = df['EMA35'].shift(7)
    df['EMA35_8'] = df['EMA35'].shift(8)
    df['EMA35_9'] = df['EMA35'].shift(9)
    df['EMA35_10'] = df['EMA35'].shift(10)

    df['EMA50'] = ema50
    df['EMA50_1'] = df['EMA50'].shift(1)
    df['EMA50_2'] = df['EMA50'].shift(2)
    df['EMA50_3'] = df['EMA50'].shift(3)
    df['EMA50_4'] = df['EMA50'].shift(4)
    df['EMA50_5'] = df['EMA50'].shift(5)
    df['EMA50_6'] = df['EMA50'].shift(6)
    df['EMA50_7'] = df['EMA50'].shift(7)
    df['EMA50_8'] = df['EMA50'].shift(8)
    df['EMA50_9'] = df['EMA50'].shift(9)
    df['EMA50_10'] = df['EMA50'].shift(10)

    df['EMA89'] = ema89
    df['EMA89_1'] = df['EMA89'].shift(1)
    df['EMA89_2'] = df['EMA89'].shift(2)
    df['EMA89_3'] = df['EMA89'].shift(3)
    df['EMA89_4'] = df['EMA89'].shift(4)
    df['EMA89_5'] = df['EMA89'].shift(5)
    df['EMA89_6'] = df['EMA89'].shift(6)
    df['EMA89_7'] = df['EMA89'].shift(7)
    df['EMA89_8'] = df['EMA89'].shift(8)
    df['EMA89_9'] = df['EMA89'].shift(9)
    df['EMA89_10'] = df['EMA89'].shift(10)

    df['EMA120'] = ema120
    df['EMA120_1'] = df['EMA120'].shift(1)
    df['EMA120_2'] = df['EMA120'].shift(2)
    df['EMA120_3'] = df['EMA120'].shift(3)
    df['EMA120_4'] = df['EMA120'].shift(4)
    df['EMA120_5'] = df['EMA120'].shift(5)
    df['EMA120_6'] = df['EMA120'].shift(6)
    df['EMA120_7'] = df['EMA120'].shift(7)
    df['EMA120_8'] = df['EMA120'].shift(8)
    df['EMA120_9'] = df['EMA120'].shift(9)
    df['EMA120_10'] = df['EMA120'].shift(10)

    df['EMA200'] = ema200
    df['EMA200_1'] = df['EMA200'].shift(1)
    df['EMA200_2'] = df['EMA200'].shift(2)
    df['EMA200_3'] = df['EMA200'].shift(3)
    df['EMA200_4'] = df['EMA200'].shift(4)
    df['EMA200_5'] = df['EMA200'].shift(5)
    df['EMA200_6'] = df['EMA200'].shift(6)
    df['EMA200_7'] = df['EMA200'].shift(7)
    df['EMA200_8'] = df['EMA200'].shift(8)
    df['EMA200_9'] = df['EMA200'].shift(9)
    df['EMA200_10'] = df['EMA200'].shift(10)

    df['MACD'] = macd
    df['MACD_1'] = df['MACD'].shift(1)
    df['MACD_2'] = df['MACD'].shift(2)
    df['MACD_3'] = df['MACD'].shift(3)
    df['MACD_4'] = df['MACD'].shift(4)
    df['MACD_5'] = df['MACD'].shift(5)
    df['MACD_6'] = df['MACD'].shift(6)
    df['MACD_7'] = df['MACD'].shift(7)
    df['MACD_8'] = df['MACD'].shift(8)
    df['MACD_9'] = df['MACD'].shift(9)
    df['MACD_10'] = df['MACD'].shift(10)

    df['MACD_MA5'] = talib.MA(df['MACD'], timeperiod=5)

    df['Signal'] = signal
    df['Signal_1'] = signal.shift(1)
    df['Signal_2'] = signal.shift(2)
    df['Signal_3'] = signal.shift(3)
    df['Signal_4'] = signal.shift(4)
    df['Signal_5'] = signal.shift(5)
    df['Signal_6'] = signal.shift(6)
    df['Signal_7'] = signal.shift(7)
    df['Signal_8'] = signal.shift(8)
    df['Signal_9'] = signal.shift(9)
    df['Signal_10'] = signal.shift(10)

    df['Signal_MA5'] = talib.MA(df['Signal'], timeperiod=5)

    df['RSI'] = rsi
    df['RSI_1'] = df['RSI'].shift(1)
    df['RSI_2'] = df['RSI'].shift(2)
    df['RSI_3'] = df['RSI'].shift(3)
    df['RSI_4'] = df['RSI'].shift(4)
    df['RSI_5'] = df['RSI'].shift(5)
    df['RSI_6'] = df['RSI'].shift(6)
    df['RSI_7'] = df['RSI'].shift(7)
    df['RSI_8'] = df['RSI'].shift(8)
    df['RSI_9'] = df['RSI'].shift(9)
    df['RSI_10'] = df['RSI'].shift(10)

    df['RSI_MA5'] = talib.MA(df['RSI'], timeperiod=5)

    df['ADX'] = adx
    df['ADX_1'] = df['ADX'].shift(1)
    df['ADX_2'] = df['ADX'].shift(2)
    df['ADX_3'] = df['ADX'].shift(3)
    df['ADX_4'] = df['ADX'].shift(4)
    df['ADX_5'] = df['ADX'].shift(5)
    df['ADX_6'] = df['ADX'].shift(6)
    df['ADX_7'] = df['ADX'].shift(7)
    df['ADX_8'] = df['ADX'].shift(8)
    df['ADX_9'] = df['ADX'].shift(9)
    df['ADX_10'] = df['ADX'].shift(10)

    df['ADX_MA5'] = talib.MA(df['ADX'], timeperiod=5)

    df['MDI'] = mdi
    df['MDI_1'] = df['MDI'].shift(1)
    df['MDI_2'] = df['MDI'].shift(2)
    df['MDI_3'] = df['MDI'].shift(3)
    df['MDI_4'] = df['MDI'].shift(4)
    df['MDI_5'] = df['MDI'].shift(5)
    df['MDI_6'] = df['MDI'].shift(6)
    df['MDI_7'] = df['MDI'].shift(7)
    df['MDI_8'] = df['MDI'].shift(8)
    df['MDI_9'] = df['MDI'].shift(9)
    df['MDI_10'] = df['MDI'].shift(10)

    df['MDI_MA5'] = talib.MA(df['MDI'], timeperiod=5)

    df['PDI'] = pdi
    df['PDI_1'] = df['PDI'].shift(1)
    df['PDI_2'] = df['PDI'].shift(2)
    df['PDI_3'] = df['PDI'].shift(3)
    df['PDI_4'] = df['PDI'].shift(4)
    df['PDI_5'] = df['PDI'].shift(5)
    df['PDI_6'] = df['PDI'].shift(6)
    df['PDI_7'] = df['PDI'].shift(7)
    df['PDI_8'] = df['PDI'].shift(8)
    df['PDI_9'] = df['PDI'].shift(9)
    df['PDI_10'] = df['PDI'].shift(10)

    df['PDI_MA5'] = talib.MA(df['PDI'], timeperiod=5)

    df['ROC'] = roc
    df['ROC_1'] = df['ROC'].shift(1)
    df['ROC_2'] = df['ROC'].shift(2)
    df['ROC_3'] = df['ROC'].shift(3)
    df['ROC_4'] = df['ROC'].shift(4)
    df['ROC_5'] = df['ROC'].shift(5)
    df['ROC_6'] = df['ROC'].shift(6)
    df['ROC_7'] = df['ROC'].shift(7)
    df['ROC_8'] = df['ROC'].shift(8)
    df['ROC_9'] = df['ROC'].shift(9)
    df['ROC_10'] = df['ROC'].shift(10)

    df['ROC_MA5'] = talib.MA(df['ROC'], timeperiod=5)

    df['StochK'] = stochK
    df['StochK_1'] = df['StochK'].shift(1)
    df['StochK_2'] = df['StochK'].shift(2)
    df['StochK_3'] = df['StochK'].shift(3)
    df['StochK_4'] = df['StochK'].shift(4)
    df['StochK_5'] = df['StochK'].shift(5)
    df['StochK_6'] = df['StochK'].shift(6)
    df['StochK_7'] = df['StochK'].shift(7)
    df['StochK_8'] = df['StochK'].shift(8)
    df['StochK_9'] = df['StochK'].shift(9)
    df['StochK_10'] = df['StochK'].shift(10)

    df['StochK_MA5'] = talib.MA(df['StochK'], timeperiod=5)

    df['StochD'] = stochD
    df['StochD_1'] = df['StochD'].shift(1)
    df['StochD_2'] = df['StochD'].shift(2)
    df['StochD_3'] = df['StochD'].shift(3)
    df['StochD_4'] = df['StochD'].shift(4)
    df['StochD_5'] = df['StochD'].shift(5)
    df['StochD_6'] = df['StochD'].shift(6)
    df['StochD_7'] = df['StochD'].shift(7)
    df['StochD_8'] = df['StochD'].shift(8)
    df['StochD_9'] = df['StochD'].shift(9)
    df['StochD_10'] = df['StochD'].shift(10)

    df['StochD_MA5'] = talib.MA(df['StochD'], timeperiod=5)

    df['CCI'] = cci
    df['CCI_1'] = df['CCI'].shift(1)
    df['CCI_2'] = df['CCI'].shift(2)
    df['CCI_3'] = df['CCI'].shift(3)
    df['CCI_4'] = df['CCI'].shift(4)
    df['CCI_5'] = df['CCI'].shift(5)
    df['CCI_6'] = df['CCI'].shift(6)
    df['CCI_7'] = df['CCI'].shift(7)
    df['CCI_8'] = df['CCI'].shift(8)
    df['CCI_9'] = df['CCI'].shift(9)
    df['CCI_10'] = df['CCI'].shift(10)

    df['CCI_MA5'] = talib.MA(df['CCI'], timeperiod=5)

    df['ATR'] = atr
    df['ATR_1'] = df['ATR'].shift(1)
    df['ATR_2'] = df['ATR'].shift(2)
    df['ATR_3'] = df['ATR'].shift(3)
    df['ATR_4'] = df['ATR'].shift(4)
    df['ATR_5'] = df['ATR'].shift(5)
    df['ATR_6'] = df['ATR'].shift(6)
    df['ATR_7'] = df['ATR'].shift(7)
    df['ATR_8'] = df['ATR'].shift(8)
    df['ATR_9'] = df['ATR'].shift(9)
    df['ATR_10'] = df['ATR'].shift(10)

    df['ATR_MA5'] = talib.MA(df['ATR'], timeperiod=5)

    df.dropna(inplace=True)
    
    return df

def feature_engineering(df):
    df['EMA10_above_EMA35?'] = [1.0 if (df.loc[df.index[i],'EMA10'] > df.loc[df.index[i],'EMA35']) 
                                else 0.0 for i in range(len(df.index))]
    df['EMA35_above_EMA50?'] = [1.0 if (df.loc[df.index[i],'EMA35'] > df.loc[df.index[i],'EMA50']) 
                                else 0.0 for i in range(len(df.index))]
    df['EMA50_above_EMA89?'] = [1.0 if (df.loc[df.index[i],'EMA50'] > df.loc[df.index[i],'EMA89']) 
                                else 0.0 for i in range(len(df.index))]
    df['EMA89_above_EMA120?'] = [1.0 if (df.loc[df.index[i],'EMA89'] > df.loc[df.index[i],'EMA120']) 
                                else 0.0 for i in range(len(df.index))]
    df['EMA120_above_EMA200?'] = [1.0 if (df.loc[df.index[i],'EMA120'] > df.loc[df.index[i],'EMA200']) 
                                else 0.0 for i in range(len(df.index))]
    df['Close_above_EMA200?'] = [1.0 if (df.loc[df.index[i],'Close'] > df.loc[df.index[i],'EMA200']) 
                                else 0.0 for i in range(len(df.index))]
    df['MACD_above_zero?'] = [1.0 if (df.loc[df.index[i],'MACD'] > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['MACD_above_upper?'] = [1.0 if (df.loc[df.index[i],'MACD'] > 10) 
                                else 0.0 for i in range(len(df.index))]
    df['MACD_below_lower?'] = [1.0 if (df.loc[df.index[i],'MACD'] < -15) 
                                else 0.0 for i in range(len(df.index))]
    df['MACD_above_Signal?'] = [1.0 if (df.loc[df.index[i],'MACD'] > df.loc[df.index[i],'Signal']) 
                                else 0.0 for i in range(len(df.index))]
    df['MACD_slope_1days'] = [1.0 if ((df.loc[df.index[i],'MACD'] - df.loc[df.index[i],'MACD_1'])/1 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['MACD_slope_5days'] = [1.0 if ((df.loc[df.index[i],'MACD'] - df.loc[df.index[i],'MACD_5'])/5 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['MACD_slope_10days'] = [1.0 if ((df.loc[df.index[i],'MACD'] - df.loc[df.index[i],'MACD_10'])/10 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['Signal_slope_1days'] = [1.0 if ((df.loc[df.index[i],'Signal'] - df.loc[df.index[i],'Signal_1'])/1 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['Signal_slope_5days'] = [1.0 if ((df.loc[df.index[i],'Signal'] - df.loc[df.index[i],'Signal_5'])/5 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['Signal_slope_10days'] = [1.0 if ((df.loc[df.index[i],'Signal'] - df.loc[df.index[i],'Signal_10'])/10 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['RSI_MA5_higher90?'] = [1.0 if (df.loc[df.index[i],'RSI_MA5'] > 90) 
                               else 0.0 for i in range(len(df.index))]
    df['RSI_MA5_higher70?'] = [1.0 if (df.loc[df.index[i],'RSI_MA5'] > 70) 
                               else 0.0 for i in range(len(df.index))]
    df['RSI_MA5_lower30?'] = [1.0 if (df.loc[df.index[i],'RSI_MA5'] < 30) 
                               else 0.0 for i in range(len(df.index))]
    df['RSI_MA5_lower10?'] = [1.0 if (df.loc[df.index[i],'RSI_MA5'] < 10) 
                               else 0.0 for i in range(len(df.index))]
    df['RSI_slope_1days'] = [1.0 if ((df.loc[df.index[i],'RSI'] - df.loc[df.index[i],'RSI_1'])/1 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['RSI_slope_5days'] = [1.0 if ((df.loc[df.index[i],'RSI'] - df.loc[df.index[i],'RSI_5'])/5 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['RSI_slope_10days'] = [1.0 if ((df.loc[df.index[i],'RSI'] - df.loc[df.index[i],'RSI_10'])/10 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['StochK_MA5_higher90?'] = [1.0 if (df.loc[df.index[i],'StochK_MA5'] > 90) 
                               else 0.0 for i in range(len(df.index))]
    df['StochK_MA5_higher70?'] = [1.0 if (df.loc[df.index[i],'StochK_MA5'] > 70) 
                               else 0.0 for i in range(len(df.index))]
    df['StochK_MA5_lower30?'] = [1.0 if (df.loc[df.index[i],'StochK_MA5'] < 30) 
                               else 0.0 for i in range(len(df.index))]
    df['StochK_MA5_lower10?'] = [1.0 if (df.loc[df.index[i],'StochK_MA5'] < 10) 
                               else 0.0 for i in range(len(df.index))]
    df['StochK_above_StochD?'] = [1.0 if (df.loc[df.index[i],'StochK'] > df.loc[df.index[i],'StochD']) 
                                 else 0.0 for i in range(len(df.index))]
    df['StochK_slope_1days'] = [1.0 if ((df.loc[df.index[i],'StochK'] - df.loc[df.index[i],'StochK_1'])/1 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['StochK_slope_5days'] = [1.0 if ((df.loc[df.index[i],'StochK'] - df.loc[df.index[i],'StochK_5'])/5 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['StochK_slope_10days'] = [1.0 if ((df.loc[df.index[i],'StochK'] - df.loc[df.index[i],'StochK_10'])/10 > 0) 
                                else 0.0 for i in range(len(df.index))]
    df['StochD_slope_1days'] = [1.0 if ((df.loc[df.index[i],'StochD'] - df.loc[df.index[i],'StochD_1'])/1 > 0) 
                                 else 0.0 for i in range(len(df.index))]
    df['StochD_slope_5days'] = [1.0 if ((df.loc[df.index[i],'StochD'] - df.loc[df.index[i],'StochD_5'])/5 > 0) 
                                 else 0.0 for i in range(len(df.index))]
    df['StochD_slope_10days'] = [1.0 if ((df.loc[df.index[i],'StochD'] - df.loc[df.index[i],'StochD_10'])/10 > 0) 
                                 else 0.0 for i in range(len(df.index))]
    df['PDI_above_MDI?'] = [1.0 if (df.loc[df.index[i],'PDI'] > df.loc[df.index[i],'MDI']) 
                                 else 0.0 for i in range(len(df.index))]
    df['ADX_slope_1days'] = [1.0 if ((df.loc[df.index[i],'ADX'] - df.loc[df.index[i],'ADX_1'])/1 > 0) 
                                 else 0.0 for i in range(len(df.index))]
    df['ADX_slope_5days'] = [1.0 if ((df.loc[df.index[i],'ADX'] - df.loc[df.index[i],'ADX_5'])/5 > 0) 
                                 else 0.0 for i in range(len(df.index))]
    df['ADX_slope_10days'] = [1.0 if ((df.loc[df.index[i],'ADX'] - df.loc[df.index[i],'ADX_10'])/10 > 0) 
                                 else 0.0 for i in range(len(df.index))]
    df['ADX_above_upper?'] = [1.0 if (df.loc[df.index[i],'ADX'] > 85) 
                                else 0.0 for i in range(len(df.index))]
    df['ADX_below_lower?'] = [1.0 if (df.loc[df.index[i],'ADX'] < 12) 
                                else 0.0 for i in range(len(df.index))]
    
    df.dropna(inplace=True)
    
    return df

def labeling(df):
    df['beLong'] = [1.0 if ((df.loc[df.index[i],'ZigZag05'] - df.loc[df.index[i],'ZigZag05_1']) > 0)
                   else -1.0 for i in range(len(df.index))]
    df['beLong_1'] = df['beLong'].shift(1)
    df['beLong_2'] = df['beLong'].shift(2)
    df['beLong_3'] = df['beLong'].shift(3)
    df.dropna(inplace=True)
    df['Hold3days?'] = [1.0 if ((df.loc[df.index[i],'beLong_1'] + df.loc[df.index[i],'beLong_2'] +
                                 df.loc[df.index[i],'beLong_3']) == 3) else 0.0 for i in range(len(df.index))]
    
    return df

def dump_to_MongoDB(host, db_name, collection_name, df):
    import pymongo

    client = pymongo.MongoClient(host)
    db = client[db_name]
    collection = db[collection_name]
    collection.drop()

    df_dict = df.to_dict(orient='records')

    collection.insert_many(df_dict)
    
def insert_to_MongoDB(host, db_name, collection_name, df):
    import pymongo

    client = pymongo.MongoClient(host)
    db = client[db_name]
    collection = db[collection_name]

    df_dict = df.to_dict()

    collection.insert_one(df_dict)
    
def retrieve_from_MongoDB(host, db_name, collection_name, no_of_rows):
    import pymongo
    import pandas as pd

    client = pymongo.MongoClient(host)
    db = client[db_name]
    collection = db[collection_name]

    df = pd.DataFrame(list(collection.find().sort('_id', -1).limit(no_of_rows)))
    df.drop(columns=['_id'], inplace=True)
    df.sort_index(ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
    
def fitting_evaluating(ticker, features_df, label_series, model, partition_split, test_size, seed):  
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix
    
    features = features_df.values
    label = label_series.values
    
    sss = StratifiedShuffleSplit(n_splits=partition_split, test_size=test_size, random_state=seed)
    
    cm_sum_is = np.zeros((2,2))
    cm_sum_oos = np.zeros((2,2))

    for train_index,test_index in sss.split(features,label):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = label[train_index], label[test_index] 

        #test the in-sample fit    
        y_pred_is = model.fit(X_train, y_train).predict(X_train)
        cm_is = confusion_matrix(y_train, y_pred_is)
        cm_sum_is = cm_sum_is + cm_is

        #test the out-of-sample data
        y_pred_oos = model.fit(X_train, y_train).predict(X_test)
        cm_oos = confusion_matrix(y_test, y_pred_oos)
        cm_sum_oos = cm_sum_oos + cm_oos
        
    tpIS = cm_sum_is[1,1]
    fnIS = cm_sum_is[1,0]
    fpIS = cm_sum_is[0,1]
    tnIS = cm_sum_is[0,0]
    precisionIS_pos = tpIS /(tpIS+fpIS)
    precisionIS_neg = tnIS /(tnIS+fnIS)
    recallIS_pos = tpIS /(tpIS+fnIS)
    recallIS_neg = tnIS /(fpIS+tnIS)
    accuracyIS = (tpIS+tnIS)/(tpIS+fnIS+fpIS+tnIS)
    f1IS_pos = (2.0 * precisionIS_pos * recallIS_pos) / (precisionIS_pos+recallIS_pos) 
    f1IS_neg = (2.0 * precisionIS_neg * recallIS_neg) / (precisionIS_neg+recallIS_neg) 

    tpOOS = cm_sum_oos[1,1]
    fnOOS = cm_sum_oos[1,0]
    fpOOS = cm_sum_oos[0,1]
    tnOOS = cm_sum_oos[0,0]
    precisionOOS_pos = tpOOS /(tpOOS+fpOOS)
    precisionOOS_neg = tnOOS /(tnOOS+fnOOS)
    recallOOS_pos = tpOOS /(tpOOS+fnOOS)
    recallOOS_neg = tnOOS /(fpOOS+tnOOS)
    accuracyOOS = (tpOOS+tnOOS)/(tpOOS+fnOOS+fpOOS+tnOOS)
    f1OOS_pos = (2.0 * precisionOOS_pos * recallOOS_pos) / (precisionOOS_pos+recallOOS_pos) 
    f1OOS_neg = (2.0 * precisionOOS_neg * recallOOS_neg) / (precisionOOS_neg+recallOOS_neg)

    print('-'*30)
    print('Symbol is', ticker)
    print('-'*30)

    print('In sample\n')
    print('     predicted')
    print('        pos neg')
    print('pos:  %i  %i  %.2f' % (tpIS, fnIS, recallIS_pos))
    print('neg:  %i  %i  %.2f' % (fpIS, tnIS, recallIS_neg))
    print('      %.2f   %.2f' % (precisionIS_pos, precisionIS_neg))
    print('\naccuracy:   %.4f' % (accuracyIS*100))
    print('f-measure positive:   %.4f' % (f1IS_pos*100))
    print('f-measure negative:   %.4f' % (f1IS_neg*100))

    print('-'*30)

    print('Out of sample\n')
    print('       predicted')
    print('        pos neg')
    print('pos:  %i  %i  %.2f' % (tpOOS, fnOOS, recallOOS_pos))
    print('neg:  %i  %i  %.2f' % (fpOOS, tnOOS, recallOOS_neg))
    print('      %.2f   %.2f' % (precisionOOS_pos, precisionOOS_neg))
    print('\naccuracy:   %.4f' % (accuracyOOS*100))
    print('f-measure positive:   %.4f' % (f1OOS_pos*100))  
    print('f-measure negative:   %.4f' % (f1OOS_neg*100))




