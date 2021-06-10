# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import os
import datetime
import statsmodels.formula.api as smf
from dateutil.parser import parse
import statsmodels.api as sm
os.chdir(r'E:\python3\Empirial_Asset_Pricing')
path = os.getcwd()

# 为什么要读取月度因子数据并计算年度的因子值？
# 因为回归是用年度数据

# 读取月度因子数据
factor_monthly = pd.read_csv('data/F-F_Year_Factors.csv')
factor_monthly['dateff'] = factor_monthly['dateff'].apply(int).apply(str).apply(parse)
factor_monthly = factor_monthly[(factor_monthly['dateff'] >= datetime.datetime(1987, 12, 31, 0, 0))&(factor_monthly['dateff'] <= datetime.datetime(2013, 1, 1, 0, 0))]
factor_monthly.rename(columns = {'dateff':'month','mktrf':'mkt'},inplace = True)
factor_monthly = factor_monthly[['month','mkt','smb','hml','umd']]

# 由于常用因子值也为根据变量值分组投资组合的收益率，因此将月度数据按年度连乘（这里为什么不是取均值？）计算年度因子值
factor_yearly = factor_monthly.copy()
factor_yearly['year'] = factor_yearly['month'].apply(lambda x:x.year)
factor_yearly[['mkt','smb','hml','umd']] = factor_yearly[['mkt','smb','hml','umd']].applymap(lambda x:x+1)
factor_yearly = factor_yearly.groupby('year').prod()
factor_yearly = factor_yearly.applymap(lambda x:(x-1)*100)
factor_yearly = factor_yearly.reset_index()

factor_yearly_yearlag1 = factor_yearly.copy()
factor_yearly_yearlag1['year'] = factor_yearly_yearlag1['year']-1


all_data = pd.read_csv(os.path.join(path,'data','alldata_mktcap.csv'),index_col=0)
all_data = all_data.drop_duplicates(subset=['permno','year'])
all_data = all_data[['permno', 'year', 'beta', 'rt+1', 'bm','mktcap']]

# 分别提取不同变量所需分组数据至多个dataframe
proxy_name_list = ['beta','mktcap','bm']
def select_proxy(proxy_name_list):
    for proxy_name in proxy_name_list:
        globals()[proxy_name] = all_data[['permno', 'year','rt+1',proxy_name]]
select_proxy(proxy_name_list)

## 做等权重的7分组
# 根据bm,beta,mktcap分别进行单变量7分组
def mutate_group(proxy_name_list):
    for proxy_name in proxy_name_list:
        globals()[proxy_name] = globals()[proxy_name].dropna()

        quantiles_proxy = globals()[proxy_name].groupby('year')[proxy_name].describe(
        percentiles=[0.1,0.2,0.4,0.6,0.8,0.9]).reset_index()[['year','10%','20%','40%','60%','80%','90%']]

        df= pd.merge(globals()[proxy_name], quantiles_proxy, how = 'left', on = 'year')

        globals()[proxy_name]['group'] = np.select([df[proxy_name] <= df['10%'],
                           (df[proxy_name] > df['10%']) & (df[proxy_name] <= df['20%']),
                           (df[proxy_name] > df['20%']) & (df[proxy_name] <= df['40%']),
                            (df[proxy_name] > df['40%']) & (df[proxy_name] <= df['60%']),
                            (df[proxy_name] > df['60%']) & (df[proxy_name] <= df['80%']),
                            (df[proxy_name] > df['80%']) & (df[proxy_name] <= df['90%']),
                           (df[proxy_name] > df['90%'])],
                           ['1','2','3','4','5','6','7'])
mutate_group(proxy_name_list)

# 计算等权投资组合收益率
def portfolio_ret(proxy_name_list):
    for proxy_name in proxy_name_list:
        globals()[proxy_name] = globals()[proxy_name].dropna()

        globals()[proxy_name] = globals()[proxy_name].groupby(['group', 'year']).apply(
            lambda x: np.average(x['rt+1'], weights=None)).reset_index()
        globals()[proxy_name].rename(columns={0: 'ret_excess'}, inplace=True)
        globals()[proxy_name] = pd.pivot(globals()[proxy_name], index='year', columns='group')[
            'ret_excess'].reset_index()

        # 没有股票的组用0填充（试验）
        globals()[proxy_name].replace(np.nan,0,inplace = True)

        globals()[proxy_name]['7-1'] = globals()[proxy_name]['7'] - globals()[proxy_name]['1']
portfolio_ret(proxy_name_list)

def nw_adjust(df, group, lags=6):
    df.dropna(subset = [group], inplace = True)
    adj_a = np.array(df[group])
    # 对常数回归
    model = sm.OLS(adj_a, [1] * len(adj_a)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return adj_a.mean().round(2), format(float(model.tvalues), ".2f"),float(model.bse)

# generate table8
table8_index = ['Beta','','MKtCap','','BM','']
table8_columns = ['1','2','3','4','5','6','7','7-1']
table8 = pd.DataFrame(index = table8_index, columns = table8_columns)
table8.columns.name = 'Sort Variable'
def generate_table8(proxy_name_list):
    for i in range(len(proxy_name_list)):
        for j in range(len(table8_columns)):
            table8.iloc[2*i,j] = nw_adjust(globals()[proxy_name_list[i]], table8_columns[j])[0]
            table8.iloc[2*i+1,j] = nw_adjust(globals()[proxy_name_list[i]],table8_columns[j])[1]
generate_table8(proxy_name_list)
print(table8)

# generate table9
table9_index = ['Excess return','',
                'CAPM','','','',
                'FF','','','','','','','',
                'FFC','','','','','','','','','']
table9_columns = ['Coefficient','1','2','3','4','5','6','7','7-1']
table9 = pd.DataFrame(index = table9_index, columns = table9_columns)
table9.columns.name = 'Model'
table9['Coefficient'] = ['Excess return','',
                         'alpha','','MKT','',
                         'alpha','','MKT','','SMB','','HML','',
                         'alpha','','MKT','','SMB','','HML','','MOM','']

def mutate_factor(proxy_name_list):
    for proxy_name in proxy_name_list:
        globals()[proxy_name] = pd.merge(globals()[proxy_name], factor_yearly_yearlag1, how='left', on=['year'])
mutate_factor(proxy_name_list)

def capm_adjust(df, group, lags = 6):
    df.dropna(subset = [group,'mkt'], inplace = True)
    x = sm.add_constant(df['mkt'])
    y = df[group]
    model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return model

def ff3_adjust(df, group, lags = 6):
    df.dropna(subset=[group, 'mkt', 'smb', 'hml'], inplace = True)
    x = sm.add_constant(df[['mkt', 'smb', 'hml']])
    y = df[group]
    model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return model

def ffc_adjust(df, group, lags = 6):
    df.dropna(subset=[group, 'mkt', 'smb', 'hml', 'umd'], inplace = True)
    x = sm.add_constant(df[['mkt', 'smb', 'hml','umd']])
    y = df[group]
    model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return model

# 因子模型解释后alpha反变大，说明年度因子算的很可能有问题
def generate_table9():
    for j in range(len(table9.columns) - 1):
        table9.iloc[0,j+1] = nw_adjust(globals()['beta'], table9_columns[j+1])[0]
        table9.iloc[1,j+1] = nw_adjust(globals()['beta'], table9_columns[j+1])[1]

        for i in range(2):
            table9.iloc[2+2*i, j+1] = capm_adjust(globals()['beta'], table9_columns[j+1]).params[i].round(2)
            table9.iloc[3+2*i, j+1] = capm_adjust(globals()['beta'], table9_columns[j+1]).tvalues[i].round(2)

        for i in range(4):
            table9.iloc[6+2*i, j+1] = ff3_adjust(globals()['beta'], table9_columns[j+1]).params[i].round(2)
            table9.iloc[7+2*i, j+1] = ff3_adjust(globals()['beta'], table9_columns[j+1]).tvalues[i].round(2)

        for i in range(5):
            table9.iloc[14+2*i, j+1] = ffc_adjust(globals()['beta'], table9_columns[j+1]).params[i].round(2)
            table9.iloc[15+2*i, j+1] = ffc_adjust(globals()['beta'], table9_columns[j+1]).tvalues[i].round(2)
generate_table9()
print(table9)

# generate table10
# bivariate independent-sort portfolio，
# The first sort variable is beta and the second sort variable is MktCap；
def generate_table10():
    quantiles_proxy_beta = all_data.groupby('year')['beta'].describe(
            percentiles=[0.3, 0.7]).reset_index()[['year', '30%', '70%']]

    quantiles_proxy_mktcap = all_data.groupby('year')['mktcap'].describe(
            percentiles=[0.25, 0.5, 0.75]).reset_index()[['year', '25%', '50%', '75%']]

    beta_mktcap_group = pd.merge(quantiles_proxy_beta, quantiles_proxy_mktcap, on=['year'])
    beta_mktcap_group.rename(columns = {'30%':'B1_1t','70%':'B1_2t',
                                        '25%':'B2_1t','50%':'B2_2t','75%':'B2_3t'},inplace=True)
    return beta_mktcap_group
table10 = generate_table10()
print(table10)

# generate table11
all_data_and_breakpoints = pd.merge(all_data,table10,on='year')
def portfolio_ind_3x4():
    df = all_data_and_breakpoints.copy()
    df = df.dropna(subset=['beta','mktcap'])
    df['X1_group'] = np.select([
        (df['beta'] < df['B1_1t']),
        (df['beta'] >= df['B1_1t']) & (df['beta'] < df['B1_2t']),
        (df['beta'] >= df['B1_2t'])],
        ['1','2','3'])
    df['X2_group'] = np.select([
        (df['mktcap'] < df['B2_1t']),
        (df['mktcap'] >= df['B2_1t']) & (df['mktcap'] < df['B2_2t']),
        (df['mktcap'] >= df['B2_2t']) & (df['mktcap'] < df['B2_3t']),
        (df['mktcap'] >= df['B2_3t'])],
        ['1','2','3','4'])
    return df[['permno','year','beta','rt+1','mktcap','X1_group','X2_group']]
all_data_and_groups = portfolio_ind_3x4()

def generate_table11(group_table):
    n = group_table.groupby(['year', 'X1_group', 'X2_group'])['permno'].count().reset_index().rename(
            columns={'permno': 'n_firms'})
    df = pd.pivot_table(n, index=['year', 'X2_group'], columns='X1_group')['n_firms'].reset_index()
    df.rename(columns={'1':'beta1','2':'beta2','3':'beta3'},inplace=True)
    return df
table11 = generate_table11(all_data_and_groups)
print(table11)

# generate table13
def avg_r(group_table):
    ewret = group_table.groupby(['year','X2_group','X1_group'])['rt+1'].mean().reset_index()
    df = pd.pivot_table(ewret,index=['year','X2_group'],columns='X1_group')['rt+1'].reset_index()
    df['diff'] = df['3']-df['1']
    df['avg'] = (df['1']+df['2']+df['3'])/3
    df.loc[:,'1':'avg']= df.loc[:,'1':'avg'].applymap(lambda x:round(x,2))
    return df
table13 = avg_r(all_data_and_groups)

def get_diff(table):
    X = pd.DataFrame()
    for i in range(1988,2012):
        x = table[table['year']==i]
        x1 = pd.DataFrame(columns=x.columns,index=list(range(2)))
        x1['year'] = [i,i]
        x1.iloc[0,1] = ['mktcap_diff']
        x1.iloc[1,1] = ['mktcap_avg']
        x1.iloc[0, 2:] = (x.iloc[3, 2:] - x.iloc[0, 2:]).apply(lambda x:round(x,2))
        x1.iloc[1, 2:] = x.iloc[0:3, 2:].mean().apply(lambda x:round(x,2))
        X = pd.concat([X, x])
        X = pd.concat([X, x1])
    return X
table13 = get_diff(table13)
table13 = table13.reset_index(drop= True)
print(table13)

# generate table14
##计算综合excess return和FFC调整alpha
table14_sub_index = ['Excess return','','FFC alpha','']
table14_sub_column = ['beta1','beta2','beta3','beta Diff','beta Avg']
table14_sub = pd.DataFrame(index=table14_sub_index,columns=table14_sub_column)
subs = table13['X2_group'].unique()

def generate_table14_subs():
    for sub in subs:
        globals()[sub + '_table'] = table14_sub.copy()
        globals()[sub + '_data'] = table13[table13['X2_group']==sub].copy()
        globals()[sub + '_data'].loc[:,'1':'avg'] = globals()[sub + '_data'].loc[:,'1':'avg'].applymap(lambda x:np.float64(x))
        globals()[sub + '_data'] = pd.merge(globals()[sub + '_data'],factor_yearly_yearlag1,on='year')
        for i in range(len(globals()[sub + '_table'].columns)):
            globals()[sub + '_table'].iloc[0,i] = nw_adjust(globals()[sub + '_data'],globals()[sub + '_data'].columns[2+i])[0]
            globals()[sub + '_table'].iloc[1,i] = nw_adjust(globals()[sub + '_data'],globals()[sub + '_data'].columns[2+i])[1]
            globals()[sub + '_table'].iloc[2,i] = ffc_adjust(globals()[sub + '_data'],globals()[sub + '_data'].columns[2+i]).params[0].round(2)
            globals()[sub + '_table'].iloc[3,i] = ffc_adjust(globals()[sub + '_data'],globals()[sub + '_data'].columns[2+i]).tvalues[0].round(2)
generate_table14_subs()

table14 = pd.DataFrame()
for sub in subs:
    table14 = pd.concat([table14, globals()[sub + '_table']])
table14 = table14.reset_index().reset_index()
table14['level_0'] = ['MktCap 1','','','',
                'MktCap 2', '', '', '',
                'MktCap 3', '', '', '',
                'MktCap 4', '', '', '',
                'MktCap Diff', '', '', '',
                'MktCap Avg', '', '', '',]
table14.rename(columns={'index':'Coefficient',
                        'level_0':''},inplace=True)
print(table14)


