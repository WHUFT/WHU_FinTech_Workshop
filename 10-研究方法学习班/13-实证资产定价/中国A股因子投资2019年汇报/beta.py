# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:12:09 2020

@author: Mei
"""
import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import statsmodels.tsa.api as smt
##########首先读入所有日度收益数据表格
a10 = pd.read_csv('10.csv')
a11 = pd.read_csv('11.csv')
a12 = pd.read_csv('12.csv')
a13 = pd.read_csv('13.csv')

##表格合并为大表，然后计算所属的
daily_data = pd.concat([a10,a11,a12,a13])##concat之后就可以删掉前面单独的数据
daily_data['date'] =  pd.to_datetime(daily_data['date'])
daily_data['rt'] = daily_data['rt']*100
daily_data['year'] = daily_data['date'].dt.year
daily_data['month'] = daily_data['date'].dt.month
daily_data = daily_data[daily_data['year']==2019]
daily_data['month_num'] = (daily_data['year']-1997)*12 + daily_data['month']

mktcap = pd.read_csv('fivefactor_daily.csv')
mktcap['date'] =  pd.to_datetime(mktcap['date'])
mktcap = mktcap[['date','mkt','rf']]
mktcap[['mkt','rf']] = mktcap[['mkt','rf']] *100

daily_data = pd.merge(daily_data,mktcap,on='date')
daily_data['rt'] = daily_data['rt']-daily_data['rf']

Acode = pd.read_csv('Acode.csv')
daily_data = pd.merge(daily_data,Acode,on='code')
daily_data = daily_data[(daily_data['exchcd']!=2)&(daily_data['exchcd']!=8)]

full_data = pd.pivot(daily_data,index='date',columns='code',values='rt')
full_data['month_num'] = (full_data.index.year-1997)*12+full_data.index.month
mktcap2 = mktcap['mkt']
mktcap2.index = mktcap['date']

def beta_calculator(data,factor,span,low_limit):
    '''
    用来计算beta的表格函数，输出是某一种计算方式的beta的表格。
    
    输入参数
    ----------
    data是以month_num为columns，code为index，rt为value
    span是每次回归跨度月份数，一年为12
    low_limit是计算beta的最低样本数（天数），一个月为10，三个月为50等
    输出
    -------
    index为股票代码，columns为月份编号，value为对应规则算出beta 的df
    '''
    X = pd.DataFrame()
    for i in range(max(data['month_num'])-span+1):
        same_time_data = data[(data['month_num']>i)&(data['month_num']<=i+span)]
        same_time = []
        code_list = list(same_time_data.columns[:-1])
        for code in code_list:
            temp_data = same_time_data[code]
            temp_data.name = 'rt'
            reg_data = pd.concat([temp_data,factor],axis=1,join='inner')
            if reg_data['rt'].notna().sum() >= low_limit:
                model = smf.ols('rt~mkt',reg_data,missing='drop').fit()
                beta = model.params[1]
            else:
                beta = np.nan
            same_time.append(beta)
        same_time = pd.Series(same_time,index = code_list,name = i+span)
        X = pd.concat([X,same_time],axis=1)
    return X

beta_3m = beta_calculator(full_data,mktcap2,3,50)
beta = beta_3m[[264,267,270,273]]
beta.columns = [0,1,2,3]