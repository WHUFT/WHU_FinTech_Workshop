    #复制Goyal and Welch
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import random
from itertools import product

os.chdir('C:/Users/longzhen/Desktop/组会/GW2008_replication')

#%%#导入数据 构造变量  (这里仅展示年度数据构造，月度频率构造类似)
raw_data = pd.read_excel("originaldata_annual.xlsx", index_col=0)

def cons_var():
    temp = pd.DataFrame({
        "ret": np.log((raw_data.Index + raw_data.D12) / raw_data.Index.shift(1)) - np.log(raw_data.Rfree + 1),
        "dfy": raw_data.BAA - raw_data.AAA,
        "infl": raw_data.infl,
        "svar": raw_data.svar,
        "de": np.log(raw_data.D12) - np.log(raw_data.E12),
        'lty': raw_data.lty,
        "tms": raw_data.lty - raw_data.tbl,
        "tbl": raw_data.tbl,
        "dfr": raw_data.corpr - raw_data.ltr,   #raw_data.lty - raw_data.ltr,
        "dp": np.log(raw_data.D12) - np.log(raw_data.Index),
        "dy": np.log(raw_data.D12) - np.log(raw_data.Index.shift(1)),
        "ltr": raw_data.ltr,
        "ep": np.log(raw_data.E12) - np.log(raw_data.Index),
        "bm": raw_data["b/m"],
        'ik': raw_data.ik,
        "ntis": raw_data.ntis,
        'eqis': raw_data.eqis,
#        'csp': raw_data.csp,  年度数据中没有csp
        'cay': raw_data.cay
    })
    return temp
a_data = cons_var()

#a_data.to_excel("annually_data.xlsx")
#%%

#a_data = pd.read_excel('annually_data.xlsx',index_col = 0)
#m_data = pd.read_excel('monthly_data.xlsx',index_col = 0)
def samp(var, samp_period):
    '''
    按照不同的样本区间取出x,y (年度)
    '''
    x = a_data[var]
    y = a_data.ret
    start = x.dropna().index[0]
    if samp_period == 'full':
        x = x.loc[start:][:-1]
        y = y.loc[(start+1):]
    elif samp_period == '20y-':
        x = x.loc[(start+19):][:-1]
        y = y.loc[(start+20):]
    elif samp_period == '1965-':
        x = x.loc[1965:][:-1]
        y = y.loc[(1965+1):]
    else:  # 1927-
        if start >= 1927:
            return pd.Series([np.nan]), pd.Series([np.nan])
        else:
            x = x.loc[1927:][:-1]
            y = y.loc[(1927+1):]
    return y,x


def IS(var, samp_period):
    '''
    table 1: 样本内回归(年度，including dividends)

    input:
    var: str, 自变量名称
    samp_period: str, 回归的样本区间
    'full': the full sample
    '20y-': 20 years after sample
    '1965-': after 1965
    '1927-': after 1927

    output:
    adjusted R2
    bootstrap F-statistics
    '''
    y, x = samp(var, samp_period)# 按照不同的样本区间取出x,y
    if np.isnan(x.iloc[0]):
        isR2 = 'same'
    else:
        result = sm.OLS(y.tolist(), sm.add_constant(x.tolist())).fit()  #拟合
        isR2 = result.rsquared_adj
    return isR2


variables = a_data.columns[1:]  # 去掉ret因变量

table1 = pd.DataFrame(index = variables, columns = ['ISR2','ISR2_20','OOSR2_20',
                                                    'rmse_20','power_20','ISR2_65',
                                                    'OOSR2_65','rmse_65','power_65',
                                                    'ISR2_27'])
for v in variables:
    table1.loc[v].ISR2 = IS(v,'full') * 100
    table1.loc[v].ISR2_20 = IS(v,'20y-')* 100
    table1.loc[v].ISR2_65 = IS(v,'1965-')* 100
    if IS(v,'1927-') != 'same':
        table1.loc[v].ISR2_27 = IS(v,'1927-')* 100
    else:
        table1.loc[v].ISR2_27 = 'same'
#%%
# kitchen sink
data_ks = a_data.drop(['ik','cay'],axis = 1).dropna()  # 1927-2005
table1 = table1.T
table1['all'] = np.nan
table1 = table1.T

result = sm.OLS(data_ks.ret[1:21].tolist(), sm.add_constant(data_ks.iloc[:20,1:])).fit()
table1.loc['all'].ISR2 = result.rsquared_adj
result = sm.OLS(data_ks.ret[21:].tolist(), sm.add_constant(data_ks.iloc[20:-1,1:])).fit()
table1.loc['all'].ISR2_20 = result.rsquared_adj
result = sm.OLS(data_ks.ret[39:].tolist(), sm.add_constant(data_ks.iloc[38:-1,1:])).fit()
table1.loc['all'].ISR2_65 = result.rsquared_adj
table1.loc['all'].ISR2_27 = 'same'

#%%
def histmean(data):  # prevailing mean
    premean = pd.Series(index = data.index)  # prevailing mean estimation
    for i in data.index:
        premean.loc[i] = data.loc[:i-1].mean()
    return premean
premean = histmean(a_data.ret)

def OOS(var):
    '''
    样本外回归预测序列
    '''
    x = a_data[var]
    y = a_data.ret
    start = x.dropna().index[0]
    pred = pd.Series(index = a_data.index)
    for i in range(start+20, 2005):# 样本外预测区间
        x_reg = x.loc[start:i-1]  # expanding forecast window
        y_reg = y.loc[start+1:i]
        x_test = a_data[var].loc[i]  # 用0~i-1期预测第i期
        result = sm.OLS(y_reg.tolist(), sm.add_constant(x_reg.tolist())).fit()  #拟合
        pred.loc[i+1] = result.predict([1, x_test])[0]  #预测
#        print(i,result.rsquared,result.params)
    return pred

OOSpred = pd.DataFrame(index = a_data.index, columns = variables)
for v in variables:
    OOSpred[v] = OOS(v)
    
# 加入样本外 all
OOSpred['all'] = np.nan
for i in range(1947,2005):
    result = sm.OLS(data_ks.iloc[1:i-1926,0].tolist(), sm.add_constant(data_ks.iloc[:i-1927,1:])).fit()
    OOSpred['all'].loc[i+1] = result.predict([1] + data_ks.iloc[i-1927,1:].tolist())[0]

#%%
def mse(y,pred):
    return ((y-pred)**2).mean()

def OOS_stat(var,samp_period):
    '''
    同时算两种预测区间的OOSR2和rmse
    '''
    start = OOSpred[var].dropna().index[0]
    if samp_period == '20y-':
        OOSstart = start+21
    else:
        OOSstart = 1965
    r = a_data.ret.loc[OOSstart:]
    MSEa = mse(r, OOSpred[var].loc[OOSstart:])
    MSEn = mse(r, premean.loc[OOSstart:])
    OOS_R2 = 1 - MSEa/MSEn
    RMSE = np.sqrt(MSEn) - np.sqrt(MSEa)
    return OOS_R2, RMSE

for v in OOSpred.columns:
    table1.loc[v].OOSR2_20 = OOS_stat(v,'20y-')[0] *100
    table1.loc[v].rmse_20 = OOS_stat(v,'20y-')[1] *100
    table1.loc[v].OOSR2_65 = OOS_stat(v,'1965-')[0] *100
    table1.loc[v].rmse_65 = OOS_stat(v,'1965-')[1] *100


#%%
#  ms 模型
OOSpred['ms_20'] = np.nan
OOSpred['ms_65'] = np.nan

def varselect(data, samp_period):
    '''
    对每一期的模型坐一次变量筛选，minimum cumulative squared errors为标准
    返回筛选过后的自变量
    '''
    a = [np.nan,1]
    loop_val = [a]*15  # 12个自变量
    pro = []
    for i in product(*loop_val):
        pro.append(i)
    pro = pro[1:]   # 全0的一组不要，函数中至少要有一个自变量
    optx = pd.DataFrame()
    optcse = 999
    for j in range(len(pro)):
        X = np.array(pro[j])*data.iloc[:-1,1:] # X要比y滞后一期
        X = X.dropna(axis = 1)
        y = data.iloc[1:,0]
        end = y.index[-1]
        if samp_period == '20y-':
            start = 1947
        if samp_period == '1965-':
            start = 1965
        r = y.loc[start:end]
        pred = []
        for t in range(start,end+1):
            x_reg = X.loc[:t-1]  # expanding forecast window
            y_reg = y.loc[1:t]
            result = sm.OLS(y_reg.tolist(), sm.add_constant(x_reg)).fit()  #拟合
            pred.append(result.predict([1]+ (np.array(pro[j])*data.loc[t][1:]).dropna().tolist())[0])
        cse = ((r-pred)**2).sum()
        if cse < optcse:
            optx = X
            optcse = cse
    return optx

for i in range(1927,2005):
    msX = varselect(data_ks.loc[:i],'20y-')# x 1927-1946, y 1928-1947
    result = sm.OLS(data_ks.iloc[1:i-1926,0].tolist(), sm.add_constant(msX)).fit()
    OOSpred['ms_20'].loc[i+1] = result.predict([1] + data_ks[msX.columns].iloc[i-1926,:].tolist())[0]

for i in range(1965,2005):
    msX = varselect(data_ks.loc[:i],'1965-')# x 1927-1946, y 1928-1947
    result = sm.OLS(data_ks.iloc[1:i-1926,0].tolist(), sm.add_constant(msX)).fit()
    OOSpred['ms_65'].loc[i+1] = result.predict([1] + data_ks[msX.columns].iloc[i-1926,:].tolist())[0]

#%%
random.seed(0)
def OOS_2mse(var,yboot,xboot,samp_period):
    premean = histmean(yboot).dropna()
    if samp_period == '20y-':
        start = yboot.index[0] + 20
    else:
        start = 1965
    pred = pd.Series(index = range(start,2005))
    for i in pred.index :# 样本外预测区间
        x_reg = xboot.loc[:i-1]  # expanding forecast window
        y_reg = yboot.loc[:i]
        x_test = xboot.loc[i]  # 用0~i-1期预测第i期
        result = sm.OLS(y_reg.tolist(), sm.add_constant(x_reg.tolist())).fit()  #拟合
        pred.loc[i+1] = result.predict([1, x_test])[0]  #预测
    r = yboot.loc[pred.index]
    MSEa = mse(r, pred)
    MSEn = mse(r, premean)
    rmse = np.sqrt(MSEn) - np.sqrt(MSEa)
    msef = len(yboot) * (MSEn - MSEa) / MSEa
    return rmse, msef

def distr_null(var,regtype,samp_period,iter_num = 10000):
    '''
    对IS： 输出F值分布
    对OOS：输出rmse、mse-f分布

    '''
    y,x = samp(var,'full')
    yres = sm.OLS(y, [1]*len(y)).fit()
    u1 = yres.resid
    alp = yres.params
    xres = sm.OLS(x.iloc[1:].tolist(), sm.add_constant(x.iloc[:-1].tolist())).fit()  #拟合
    u2 = xres.resid
    beta0, beta1 = xres.params
    if regtype == 'IS':
        dst = pd.DataFrame(index = range(iter_num),columns = ['f-stat'])
        for i in range(iter_num):
#            initial = x.dropna().index[0]
            initial = random.randint(x.index[0],x.index[-1])  # 任选一天作为起始点 （这样理解对吗？？orz）
            u1boot = pd.Series(np.random.choice(u1,x.index[-1]-initial+1),index = range(initial+1,2006))
            u2boot = pd.Series(np.random.choice(u2,x.index[-1]-initial+1),index = range(initial,2005))
            yboot = alp.values + u1boot
            xboot = (beta0 + beta1 * a_data[var] + u2boot).loc[initial:2004]
            result = sm.OLS(yboot.tolist(), sm.add_constant(xboot.tolist())).fit()
            dst.iloc[i,0] = result.fvalue
    else:
        dst = pd.DataFrame(index = range(iter_num), columns = ['rmse','mse_f'])
        for i in range(iter_num):
            u1boot = pd.Series(np.random.choice(u1,len(y)),index = y.index)
            u2boot = pd.Series(np.random.choice(u2,len(y)),index = x.index)
            yboot = alp.values + u1boot
            xboot = (beta0 + beta1 * x + u2boot)
            dst.iloc[i,:] = OOS_2mse(var,yboot,xboot,samp_period)
    return np.percentile(dst,[90,95,99],axis = 0)

# IS critical value
IS_cri = pd.DataFrame(index = variables, columns = ['90','95','99','f-stat'])
for v in variables:
    cri = distr_null(v,'IS','full')
    IS_cri.loc[v][0] = cri[0][0]
    IS_cri.loc[v][1] = cri[1][0]
    IS_cri.loc[v][2] = cri[2][0]
    y,x = samp(v, 'full')
    result = sm.OLS(y.tolist(), sm.add_constant(x.tolist())).fit()
    IS_cri.loc[v][3] = result.fvalue
    print(v)
#%%
OOS_cri_eqis = distr_null('eqis','OOS','20y-')

def distr_power(var,samp_period,iter_num = 10000):
    y,x = samp(var,'full')
    yres = sm.OLS(y.tolist(), sm.add_constant(x.tolist())).fit()
    u1 = yres.resid
    alp0,alp1 = yres.params
    xres = sm.OLS(x[1:].tolist(), sm.add_constant(x[:-1].tolist())).fit()  #拟合
    u2 = xres.resid
    beta0, beta1 = xres.params
    rmse = []
    for i in range(iter_num):
        u1boot = pd.Series(np.random.choice(u1,len(u1)),index = y.index)
        u2boot = pd.Series(np.random.choice(u2,len(u2)),index = x[1:].index)
        yboot = (alp0 + alp1 * x.values + u1boot)
        xboot = (beta0 + beta1 * x[:-1].values + u2boot)
        rmse.append(OOS_2mse(var,yboot[1:],xboot,samp_period)[0]) # yboot比xboot多一期
    power = (np.array(rmse)>distr_null(var,'OOS',samp_period)[1][0]).mean() # 95% rmse
    return power


for v in variables:
    table1.loc[v].power_20 = distr_power(v,'20y-')
    table1.loc[v].power_65 = distr_power(v,'1965-')



