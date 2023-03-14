# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:30:49 2020

@author: shuoshuo
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from NWttest import nwttest_1samp

#ff处理
ff=pd.read_csv('FFC_Factors.csv')
ff['year'] = ff['dateff']//10000
ff=ff[(ff['year']>1987)&(ff['year']<2012)]
del ff['dateff']
del ff['rf']
ff = ff.applymap(lambda x:x+1)
ff['year'] = ff['year']-1
ff = ff.groupby('year').prod()
ff = ff.applymap(lambda x:(x-1)*100)
#其他数据导入
df0=pd.read_csv("alldata_mktcap.csv",index_col=0)
df0=df0.drop_duplicates(subset=['permno','year'])
df0 = df0.rename(columns={'me':'mktcap'})
beta=df0[['permno','year','beta']]
mktcap=df0[['permno','year','mktcap']]
ret=df0[['permno','year','rt+1']]
bm=df0[['permno','year','bm']]
################################# 5.1 #######################################

#计算beta断点
beta_breakpoint = beta.groupby(['year'])['beta'].describe(percentiles=[0.1,0.2,0.4,0.6,0.8,0.9]).reset_index()
beta_breakpoint = beta_breakpoint[['year','10%','20%','40%','60%','80%','90%']]
beta_breakpoint.columns = ['year','B1','B2','B3','B4','B5','B6']

#生成table1
def showtable1():
    table = beta_breakpoint.copy()
    for i in range(1,7):
        name = beta_breakpoint.columns[i]
        table[name] = table[name].apply(lambda x:round(x, 2))
    print(table)
showtable1()

##对样本进行分组
beta_group = pd.merge(beta,ret,on=['permno','year'])


'''
TODO by Bin Li, 20200315:
1. 这个代码写得不容易懂，参考Fama-French Factors (Python).pdf的分组代码重新写一下

'''

def portfolio_single(df,breakpoints,X_name):
    data = df.copy()
    X = pd.DataFrame()
    for year in range(1988,2013):
        temp_data = data[data['year'] == year]
        tempb = breakpoints[breakpoints['year'] == year].iloc[0,:]
        B1 = [-np.inf,tempb[1],tempb[2],tempb[3],tempb[4],tempb[5],tempb[6],np.inf]
        temp = pd.DataFrame()
        for i in range(7):
            x = temp_data[(temp_data[X_name]>=B1[i])&(temp_data[X_name]<=B1[i+1])]
            x['group'] = X_name+str(i+1)
            temp = pd.concat([temp,x])
        X = pd.concat([X,temp])
    return X

beta_group = portfolio_single(beta_group,beta_breakpoint,'beta')

##生成table2
def showtable2():
    table = beta_group.groupby(['year','group'])['permno'].count().reset_index().rename(columns={'permno':'n_firms'})
    table = pd.pivot_table(table,index = ['year'],columns = 'group')['n_firms'].reset_index()
    table.columns=['year','n1_t','n2_t','n3_t','n4_t','n5_t','n6_t','n7_t']
    print(table)

showtable2()

##求每年等权平均
def annual_equal_mean(group_data):
    df = group_data.groupby(['year','group'])['rt+1'].mean().reset_index()
    df = pd.pivot_table(df,index = 'year',columns = 'group')['rt+1'].reset_index()
    name = df.columns
    df['diff'] = df[name[-1]]-df[name[1]]
    return df

beta_group = beta_group.dropna()
beta_equal_annual = annual_equal_mean(beta_group)

###生成table3
def showtable3():
    table = beta_equal_annual.copy()
    table['year'] = table['year'] +1
    table = table.rename(columns= {'year':'t+1'})
    table.index = range(1988,2012)
    table.index.name = 't'
    for i in range(1,9):
        name = beta_equal_annual.columns[i]
        table[name] = table[name].apply(lambda x:round(x, 2))
    print(table)
showtable3()


'''
TODO by Bin Li, 20200315:
1. 同上一条评论，参考重写一下。
'''
##求每年加权平均
def annual_weight_mean(group_data,weight):
    df = pd.merge(group_data,weight,on=['permno','year'])
    df = df.dropna()
    weight_name = df.columns[-1]
    df['multipul'] = df['rt+1']*df[weight_name]
    df1 = df.groupby(['year','group'])['multipul'].sum().reset_index()
    df2 = df.groupby(['year','group'])[weight_name].sum().reset_index()
    df = pd.merge(df1,df2,on=['year','group'])
    df['mean'] = df['multipul']/df[weight_name]
    df = pd.pivot_table(df,index = 'year',columns = 'group')['mean'].reset_index()
    name = df.columns
    df['diff'] = df[name[-1]]-df[name[1]]
    return df

beta_weight_annual = annual_weight_mean(beta_group,mktcap)

###生成table4
def showtable4():
    table = beta_weight_annual.copy()
    table['year'] = table['year'] +1
    table = table.rename(columns= {'year':'t+1'})
    table.index = range(1988,2012)
    table.index.name = 't'
    for i in range(1,9):
        name = beta_equal_annual.columns[i]
        table[name] = table[name].apply(lambda x:round(x, 2))
    print(table)
showtable4()

###生成table5
def showtable5():
    df1 = beta_equal_annual.mean()
    df2 = pd.Series(nwttest_1samp(beta_equal_annual,0)[0],index = df1.index)
    df3 = pd.Series(nwttest_1samp(beta_equal_annual,0)[1],index = df1.index)
    table = pd.concat([df1,df2,df3],axis=1).T
    del table['year']
    table.index = ['Average','t-statistic','p-value']
    table = table.applymap(lambda x:round(x, 2))
    print(table)
showtable5()

###生成table6
def showtable6():
    df1 = beta_equal_annual.mean()
    df2 = pd.Series(nwttest_1samp(beta_equal_annual,0)[0],index = df1.index)
    table = pd.concat([df1,df2],axis=1).T
    del table['year']
    table.index = ['Average','t-statistic']
    table = table.applymap(lambda x:round(x, 2))
    print(table)
showtable6()

###生成table7
all_beta_group = pd.merge(beta_group,mktcap,on=['permno','year'])
all_beta_group = pd.merge(all_beta_group,bm,on=['permno','year'])

def showtable7():
    data = all_beta_group.copy()
    col = [1,2,3,4,5,6,7]
    df1 = data.groupby('group')['beta'].mean()
    df2 = data.groupby('group')['mktcap'].mean()
    df3 = data.groupby('group')['bm'].mean()
    df1.index = col
    df2.index = col
    df3.index = col
    X = []
    for i in ['beta','mktcap','bm']:
        temp = data.groupby(['year','group'])[i].mean().reset_index()
        temp = pd.pivot_table(temp,index = 'year',columns = 'group')[i].reset_index()
        name = temp.columns
        temp['diff'] = temp[name[-1]]-temp[name[1]]
        x = nwttest_1samp(temp['diff'],0)[0]
        X.append(x)
    X = pd.Series(X,name = '7-1 t-statistic',index = ['beta','MktCap','BM'])
    table = pd.concat([df1,df2,df3],axis=1).T
    table.index = ['beta','MktCap','BM']
    table['7-1']= table[7]-table[1]
    table['7-1 t-statistic'] = X
    print(table)
showtable7()

'''
TODO by Bin Li, 20200315:
1. 下面两段完全是上面重复的代码，可以整合成一个函数后直接调用
2. 这里的分组设置可以作为参数
'''
###用对beta同样的函数用mktcap和bm进行分组计算
mktcap_breakpoint = mktcap.groupby(['year'])['mktcap'].describe(percentiles=[0.1,0.2,0.4,0.6,0.8,0.9]).reset_index()
mktcap_breakpoint = mktcap_breakpoint[['year','10%','20%','40%','60%','80%','90%']]
mktcap_breakpoint.columns = ['year','B1','B2','B3','B4','B5','B6']
mktcap_group = portfolio_single(mktcap,mktcap_breakpoint,'mktcap')
mktcap_group = pd.merge(mktcap_group,ret,on=['permno','year'])
mktcap_group = mktcap_group.dropna()
mktcap_equal_annual = annual_equal_mean(mktcap_group)

bm_breakpoint = bm.groupby(['year'])['bm'].describe(percentiles=[0.1,0.2,0.4,0.6,0.8,0.9]).reset_index()
bm_breakpoint = bm_breakpoint[['year','10%','20%','40%','60%','80%','90%']]
bm_breakpoint.columns = ['year','B1','B2','B3','B4','B5','B6']
bm_group = portfolio_single(bm,bm_breakpoint,'bm')
bm_group = pd.merge(bm_group,ret,on=['permno','year'])
bm_group = bm_group.dropna()
bm_equal_annual = annual_equal_mean(bm_group)

'''
TODO by Bin Li, 20200315:
1. 这里的表格也是大量重复，重复的代码尽量少，写成函数。
'''
###生成表格8
def showtable8():
    table = pd.DataFrame()
    datalist = [beta_equal_annual,mktcap_equal_annual,bm_equal_annual]
    namelist = ['beta','MktCap','BM']
    for i in range(3):
        df1 = datalist[i].mean()
        df2 = pd.Series(nwttest_1samp(datalist[i],0)[0],index = df1.index)
        x = pd.concat([df1,df2],axis=1).T
        del x['year']
        x.columns = [1,2,3,4,5,6,7,'7-1']
        x.index = [namelist[i],' ']
        table = pd.concat([table,x])
    table = table.applymap(lambda x:round(x, 2))
    print(table)
showtable8()


###生成表格9
beta_equal_annual = annual_equal_mean(beta_group)
beta_equal_annual.index = beta_equal_annual['year']
del beta_equal_annual['year']
#excessreturn部分结果
df1 = beta_equal_annual.mean()
df2 = pd.Series(nwttest_1samp(beta_equal_annual,0)[0],index = df1.index)
excess_return = pd.concat([df1,df2],axis=1).T
excess_return.index = ['excess return',' ']
excess_return.index.name = None
excess_return.columns = [1,2,3,4,5,6,7,8]
#制作capm部分结果
CAPM = pd.DataFrame(index = ['alpha',' ','beta_mkt',' '])
for i in range(8):
    name = beta_equal_annual.columns
    data = pd.concat([beta_equal_annual[name[i]],ff['mktrf']],axis=1)
    capm=smf.ols(name[i]+'~mktrf',data).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    X = [capm.params[0],capm.tvalues[0],capm.params[1],capm.tvalues[1]]
    X = pd.Series(X,name = i+1,index = ['alpha',' ','beta_mkt',' '])
    CAPM = pd.concat([CAPM,X],axis=1)
#制作FF部分结果
inx = ['alpha',' ','beta_mkt',' ','beta_smb',' ','beta_hml',' ']
FF3 = pd.DataFrame(index = inx)
for i in range(8):
    name = beta_equal_annual.columns
    data = pd.concat([beta_equal_annual[name[i]],ff[['mktrf','smb','hml']]],axis=1)
    ff3=smf.ols(name[i]+'~mktrf+smb+hml',data).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    X = []
    for j in range(4):
        X.append(ff3.params[j])
        X.append(ff3.tvalues[j])
    X = pd.Series(X,name = i+1,index = inx)
    FF3 = pd.concat([FF3,X],axis=1)
#制作FFC部分结果
inx = ['alpha',' ','beta_mkt',' ','beta_smb',' ','beta_hml',' ','beta_umd',' ']
FFC = pd.DataFrame(index = inx)
for i in range(8):
    name = beta_equal_annual.columns
    data = pd.concat([beta_equal_annual[name[i]],ff],axis=1)
    ffc=smf.ols(name[i]+'~mktrf+smb+hml+umd',data).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    X = []
    for j in range(5):
        X.append(ffc.params[j])
        X.append(ffc.tvalues[j])
    X = pd.Series(X,name = i+1,index = inx)
    FFC = pd.concat([FFC,X],axis=1)
#综合
table = pd.concat([excess_return,CAPM,FF3,FFC],keys=['ExcessReturn','CAPM','FF','FFC'])
table.rename(columns= {8:'7-1'})
print(table.applymap(lambda x:round(x, 2)))

############################ 5.2 #####################################
def get_decile3_4_indep(x1,x2):

    data = pd.merge(x1,x2,on = ['permno','year'])
    data = data.dropna()
    data = data.drop_duplicates(subset=['permno','year'])
    x1_name = data.columns[2]
    x2_name = data.columns[3]

    x1 = data.groupby(['year'])[x1_name].describe(percentiles=[0.3, 0.7]).reset_index()
    x1 = x1[['year','30%','70%']]
    x2 = data.groupby(['year'])[x2_name].describe(percentiles=[0.25, 0.5, 0.75]).reset_index()
    x2 = x2[['year','25%','50%','75%']]

    df = pd.merge(x1, x2, how='inner', on=['year'])
    df.columns=['year','B1_1t','B1_2t','B2_1t','B2_2t','B2_3t']
    return df

table10 = get_decile3_4_indep(beta,mktcap)

table10 = table10.apply(lambda x:round(x, 2))
# 独立分组

def portfolio_ind_3x4(breakpoints,X1_name,X2_name):
    data = alldata_mktcap.copy()
    X = pd.DataFrame()
    for year in range(1988,2013):
        temp_data = data[data['year'] == year]
        temp_breakpoint = breakpoints[breakpoints['year'] == year].iloc[0,:]
        B1 = [-np.inf,temp_breakpoint[1],temp_breakpoint[2],np.inf]
        B2 = [-np.inf,temp_breakpoint[3],temp_breakpoint[4],temp_breakpoint[5],np.inf]
        temp = pd.DataFrame()
        for i in range(3):
            x = temp_data[(temp_data[X1_name]>=B1[i])&(temp_data[X1_name]<=B1[i+1])]
            x['X1_group'] = X1_name+str(i+1)
            temp = pd.concat([temp,x])
        temp2 = pd.DataFrame()
        for j in range(4):
            x = temp[(temp[X2_name]>=B2[j])&(temp[X2_name]<=B2[j+1])]
            x['X2_group'] = X2_name+str(j+1)
            temp2 = pd.concat([temp2,x])
        X = pd.concat([X,temp2])
    return X

class_1 = portfolio_ind_3x4(table10,'beta','mktcap')

#计数
def counting(decile_table):
    n = decile_table.groupby(['year','X1_group','X2_group'])['permno'].count().reset_index().rename(columns={'permno':'n_firms'})
    df = pd.pivot_table(n,index = ['year','X2_group'],columns = 'X1_group')['n_firms'].reset_index()
    return df

table11 = counting(class_1)
#print(table11)

#获取分组后的超额收益
def avg_r(decile_table):
    ewret=decile_table.groupby(['year','X2_group','X1_group'])['rt+1'].mean().reset_index()
    df = pd.pivot_table(ewret,index = ['year','X2_group'],columns = 'X1_group')['rt+1'].reset_index()
    df['diff'] = df.iloc[:,4]-df.iloc[:,2]
    df['avg'] = (df.iloc[:,2]+df.iloc[:,3]+df.iloc[:,4])/3
    for i in range(2,6):
        col = df.columns[i]
        df[col] = df[col].apply(lambda x:round(x, 2))
    return df

table12a = avg_r(class_1)

#print(table12)

def get_diff(table):
    X = pd.DataFrame()
    for i in range(1988,2012):
        x = table[table['year'] == i]
        x1 = pd.DataFrame(columns = x.columns,index = list(range(2)))
        x1['year'] = [i,i]
        x1.iloc[0,1] = ['mktcap_diff']
        x1.iloc[1,1] = ['mktcap_avg']
        x1.iloc[0,2:] = x.iloc[3,2:] - x.iloc[0,2:]
        x1.iloc[1,2:] = x.iloc[0:3,2:].mean()
        X = pd.concat([X,x])
        X = pd.concat([X,x1])
    return X

a = get_diff(table12a)
b = a.reset_index(drop = 'True')


'''
TODO by Bin Li, 20200315:
1. 这里重复了代码，比上面好些，统一成一个函数。
'''
##计算综合excess return和FFC调整alpha
def beta_mktcap_excess(data):
    #    df = pd.DataFrame(columns = data.columns[2:])
    df = pd.DataFrame()
    for i in data['X2_group'][:6]:
        X = pd.DataFrame(columns = data.columns[2:])
        x = pd.DataFrame(columns = data.columns)
        for year in range(1988,2012):
            temp = data[(data['year'] == year)&(data['X2_group'] == i)]
            x = x.append(temp,ignore_index=True)
        x.index = x['year']
        x = x[x.columns[2:]]
        x = x.astype(np.float64)
        X = X.append(x.mean(),ignore_index=True)
        tv1 = pd.Series(nwttest_1samp(x,0)[0],index = x.columns)
        X = X.append(tv1,ignore_index=True)
        #接下来回归FFC_alpha,用到前面的四因子表格ff
        temp_alpha = []
        temp_tv = []
        for j in x.columns:
            temp_data = pd.concat([x[j],ffc],axis=1)
            model = smf.ols(str(j)+'~mktrf+smb+hml+umd',temp_data).fit(cov_type='HAC',cov_kwds={'maxlags':6})
            temp_alpha.append(model.params[0])
            temp_tv.append(model.tvalues[0])
        temp_alpha = pd.Series(temp_alpha,index=x.columns)
        temp_tv = pd.Series(temp_tv,index=x.columns)
        X = X.append(temp_alpha,ignore_index=True)
        X = X.append(temp_tv,ignore_index=True)
        X.index = [[i,i,i,i],['excess_return','t_value_er','FFC_alpha','t_value_alpha']]
        X.index.name = ['mktport','Coefficient']
        df = pd.concat([df,X])
    return df

beta_mktcap_total_mean = beta_mktcap_excess(b)
table14 = beta_mktcap_total_mean.apply(lambda x:round(x, 2))

def get_ffc(data):
    df = pd.DataFrame()
    for i in data['X2_group'][:6]:
        X = pd.DataFrame(columns = data.columns[2:])
        x = pd.DataFrame(columns = data.columns)
        for year in range(1988,2012):
            temp = data[(data['year'] == year)&(data['X2_group'] == i)]
            x = x.append(temp,ignore_index=True)
        x.index = x['year']
        x = x[x.columns[2:]]
        x = x.astype(np.float64)
        temp_alpha = []
        temp_tv = []
        for j in x.columns:
            temp_data = pd.concat([x[j],ffc],axis=1)
            model = smf.ols(str(j)+'~mktrf+smb+hml+umd',temp_data).fit(cov_type='HAC',cov_kwds={'maxlags':6})
            temp_alpha.append(model.params[0])
            temp_tv.append(model.tvalues[0])
        temp_alpha = pd.Series(temp_alpha,index=x.columns)
        temp_tv = pd.Series(temp_tv,index=x.columns)
        X = X.append(temp_alpha,ignore_index=True)
        X = X.append(temp_tv,ignore_index=True)
        X.index = [[i,i],['FFC_alpha','t_value_alpha']]
        X.index.name = ['mktport','Coefficient']
        df = pd.concat([df,X])
        df = df.apply(lambda x:round(x, 2))
    df.iloc[1:9:2,:3] = ''

    return df

ff_alpha = get_ffc(b)

#bm取代beta
table_a = get_decile3_4_indep(bm,mktcap)
class_2 = portfolio_ind_3x4(table_a,'bm','mktcap')
table11_bm = counting(class_2)
table12_bm = avg_r(class_2)
#获得完整收益率表
table_input = get_diff(table12_bm)
table_input = table_input.reset_index(drop = 'True')
#获得回归值
bm_mktcap_total_mean = beta_mktcap_excess(table_input)

def get_table16(table1,table2):
    col = ['1','2', '3', 'Avg', '3-1']
    inx = ['Excess return', '', 'FFc Alpha','']
    x = table1.iloc[16:20,:]
    x.columns = col
    x.index = inx
    y = table2.iloc[16:20,:]
    y.columns = col
    y.index = inx
    table = pd.concat([x,y],keys = ['beta','bm'])
    table = table.applymap(lambda x:round(x, 2))
    return table
table16 = get_table16(beta_mktcap_total_mean,bm_mktcap_total_mean)
print(table16)

def get_avg(table1,table2):
    col = ['MktCap1', 'MktCap2', 'MktCap3', 'MktCap4', 'MktCap4-1','MktCapAvg']
    inx = ['Excess return', '', 'FFc Alpha','']
    x1 = table1['avg']
    temp_beta = pd.DataFrame(columns = col)
    for i in range(4):
        temp = x1.iloc[i::4]
        temp.index = col
        temp.name = inx[i]
        temp_beta = temp_beta.append(temp)
    x2 = table2['avg']
    temp_bm = pd.DataFrame(columns = col)
    for j in range(4):
        temp = x2.iloc[j::4]
        temp.index = col
        temp.name = inx[j]
        temp_bm = temp_bm.append(temp)
    table = pd.concat([temp_beta,temp_bm],keys = ['beta','bm'])
    table = table.applymap(lambda x:round(x,2))
    return table

table17 = get_avg(beta_mktcap_total_mean,bm_mktcap_total_mean)
print(table17)

#bm取代mktcap
table_b = get_decile3_4_indep(beta,bm)
class_3 = portfolio_ind_3x4(table_b,'beta','bm')
table11_bm2 = counting(class_3)
table12_bm2 = avg_r(class_3)
#获得完整收益率表
table_input2 = get_diff(table12_bm2)
table_input2 = table_input2.reset_index(drop = 'True')
#获得回归值
beta_bm_total_mean = beta_mktcap_excess(table_input2)

table18 = get_table16(beta_mktcap_total_mean,beta_bm_total_mean)
table19 = get_avg(beta_mktcap_total_mean,beta_bm_total_mean)
print(table18)
print(table19)

####################################5.3##############################################
##确定Breakpoint
def get_decile3_4_dep(X1,X2,year):
    # X1=beta,X2=mrtcap
    # date=1988
    df = pd.DataFrame(columns=['t','k','B2_1','B1_1','B2_2','B1_2','B2_3'])
    indicator = pd.merge(X1,X2,on=['permno','year'])
    x = indicator[indicator['year']==year]
    x = x.dropna()#x是有两个变量所有信息的df
    x = x.drop_duplicates(subset=['permno','year'])
    x1_name = x.columns[2]
    x2_name = x.columns[3]
    x1 = x[x1_name]
    B1_1 = np.percentile(x1,30)
    B1_2 =  np.percentile(x1,70)
    B1_range = [-np.inf,B1_1,B1_2,np.inf]
    for i in range(1,4):
        down = B1_range[i-1]
        up = B1_range[i]
        temp = x[(x[x1_name]<=up)&(x[x1_name]>=down)]
        x2 = temp[x2_name]
        B2=[np.percentile(x2,25),np.percentile(x2,50),np.percentile(x2,75)]
        df['B2_'+str(i)]=B2
    df['t'] = year
    df['k'] = [1,2,3]
    df['B1_1'] = B1_1
    df['B1_2'] = B1_2
    return df

def get_dep_3x4_break(X1,X2):
    df = pd.DataFrame(columns=['t','k','B2_1','B1_1','B2_2','B1_2','B2_3'])
    for i in range(1988,2013):
        df1 = get_decile3_4_dep(X1,X2,i)
        df = pd.concat([df,df1])
    return df

breakpoint_dep_beta_mktcap = get_dep_3x4_break(beta,mktcap)
breakpoint_dep_beta_mktcap.to_csv('第五章table20.csv')

##生成table20
def showtable20():
    table = breakpoint_dep_beta_mktcap.copy()
    for i in range(2,7):
        name = breakpoint_dep_beta_mktcap.columns[i]
        table[name] = table[name].apply(lambda x:round(x, 2))
    print(table)
showtable20()

##按照breakpoint分组
def portfolio_dep_3x4(breakpoints,X1,X2):
    data = pd.merge(X1,X2,on=['permno','year'])
    data = data.dropna()
    data = data.drop_duplicates(subset=['permno','year'])
    copy = pd.DataFrame()
    x1_name = data.columns[2]
    x2_name = data.columns[3]
    for k in range(1988,2013):
        temp_breakpoints = breakpoints[breakpoints['t'] == k]
        temp_value = data[data['year'] == k]
        B1 = [-np.inf,temp_breakpoints['B1_1'][0],temp_breakpoints['B1_2'][0]]
        for i in range(3):
            temp_value.loc[temp_value[x1_name]>=B1[i],'X1_group'] = x1_name+str(i+1)
        for j in range(1,4):
            B2 = temp_breakpoints['B2_'+str(j)].tolist()
            B2.insert(0,-np.inf)
            for l in range(4):
                temp_value.loc[(temp_value['X1_group'] == x1_name+str(j))&(temp_value[x2_name]>=B2[l]),'X2_group'] = x2_name + str(l+1)
        temp = temp_value[['permno','year','X1_group','X2_group']]
        copy = pd.concat([copy,temp])
    df = pd.merge(data,copy,on=['permno','year'])
    return df

beta_mktcap_group = portfolio_dep_3x4(breakpoint_dep_beta_mktcap,beta,mktcap)

##生成table21
def showtable21():
    x = beta_mktcap_group.groupby(['year','X1_group','X2_group'])['permno'].count().reset_index()
    x = pd.pivot_table(x,index = ['year','X2_group'],columns = 'X1_group')['permno'].reset_index()
    print(x)

showtable21()

##计算port平均
beta_mktcap_group = pd.merge(beta_mktcap_group,ret,on=['permno','year'])

def beta_mktcap_mean():
    X = pd.DataFrame()
    df = beta_mktcap_group.groupby(['year','X1_group','X2_group'])['rt+1'].mean().reset_index()
    df = pd.pivot_table(df,index = ['year','X2_group'],columns = 'X1_group')['rt+1'].reset_index()
    for i in range(1988,2012):
        x = df[df['year'] == i]
        x1 = pd.DataFrame(columns = x.columns)
        x1['year'] = [i]
        x1['X2_group'] = ['mktcap_diff']
        x1.iloc[0,2:] = x.iloc[3,2:]-x.iloc[0,2:]
        x = pd.concat([x,x1])
        X = pd.concat([X,x])
    X['beta_avg'] = (X['beta1']+X['beta2']+X['beta3'])/3
    return X

beta_mktcap_annual_mean = beta_mktcap_mean().reset_index(drop=True)

#生成table22
def showtable22():
    table = beta_mktcap_annual_mean.copy()
    for i in range(2,6):
        name = beta_mktcap_annual_mean.columns[i]
        table[name] = table[name].apply(lambda x:round(x, 2))
    print(table)
showtable22()

'''
TODO by Bin Li, 20200315:
1. 重复，将该功能写成一个函数，替换以上所有的绩效分析

'''
##计算综合excess return和FFC调整alpha
def beta_mktcap_excess(data):
    #df = pd.DataFrame(columns = data.columns[2:])
    df = pd.DataFrame()
    for i in data['X2_group'][:5]:
        X = pd.DataFrame(columns = data.columns[2:])
        x = pd.DataFrame(columns = data.columns)
        for year in range(1988,2012):
            temp = data[(data['year'] == year)&(data['X2_group'] == i)]
            x = x.append(temp,ignore_index=True)
        x.index = x['year']
        x = x[x.columns[2:]]
        x = x.astype(np.float64)
        X = X.append(x.mean(),ignore_index=True)
        tv1 = pd.Series(nwttest_1samp(x,0)[0],index = x.columns)
        X = X.append(tv1,ignore_index=True)
        #接下来回归FFC_alpha,用到前面的四因子表格ff
        temp_alpha = []
        temp_tv = []
        for j in x.columns:
            temp_data = pd.concat([x[j],ff],axis=1)
            model = smf.ols(str(j)+'~mktrf+smb+hml+umd',temp_data).fit(cov_type='HAC',cov_kwds={'maxlags':6})
            temp_alpha.append(model.params[0])
            temp_tv.append(model.tvalues[0])
        temp_alpha = pd.Series(temp_alpha,index=x.columns)
        temp_tv = pd.Series(temp_tv,index=x.columns)
        X = X.append(temp_alpha,ignore_index=True)
        X = X.append(temp_tv,ignore_index=True)
        X.index = [[i,i,i,i],['Excess_return','t_value_er','FFC_alpha','t_value_alpha']]
        X.index.name = ['X2_group','Coefficient']
        df = pd.concat([df,X])
    return df

beta_mktcap_total_mean = beta_mktcap_excess(beta_mktcap_annual_mean)

#生成表格23
def showtable23():
    table = beta_mktcap_total_mean.copy()
    for i in beta_mktcap_total_mean.columns:
        table[i] = table[i].apply(lambda x:round(x, 2))
    print(table)
showtable23()

#生成表格24
def showtable24():
    table = beta_mktcap_total_mean.copy()
    for i in beta_mktcap_total_mean.columns:
        table[i] = table[i].apply(lambda x:round(x, 2))
    df = table.iloc[:17:4,:]
    x = table.iloc[17:,:]
    df = pd.concat([df,x])
    print(df)
showtable24()

#为了后面的表格对比要加一个beta的差分
def beta_mktcap_mean2():
    X = pd.DataFrame()
    df = beta_mktcap_group.groupby(['year','X1_group','X2_group'])['rt+1'].mean().reset_index()
    df = pd.pivot_table(df,index = ['year','X2_group'],columns = 'X1_group')['rt+1'].reset_index()
    for i in range(1988,2012):
        x = df[df['year'] == i]
        x1 = pd.DataFrame(columns = x.columns)
        x1['year'] = [i]
        x1['X2_group'] = ['mktcap_diff']
        x1.iloc[0,2:] = x.iloc[3,2:]-x.iloc[0,2:]
        x = pd.concat([x,x1])
        X = pd.concat([X,x])
    X['beta_avg'] = (X['beta1']+X['beta2']+X['beta3'])/3
    X['beta_diff'] = X['beta3']-X['beta1']
    return X

beta_mktcap_annual_mean = beta_mktcap_mean2().reset_index(drop=True)
beta_mktcap_total_mean = beta_mktcap_excess(beta_mktcap_annual_mean)

#用BM替换beta如法炮制
breakpoint_dep_bm_mktcap = get_dep_3x4_break(bm,mktcap)
bm_mktcap_group = portfolio_dep_3x4(breakpoint_dep_bm_mktcap,bm,mktcap)
bm_mktcap_group = pd.merge(bm_mktcap_group,ret,on=['permno','year'])
def bm_mktcap_mean():
    X = pd.DataFrame()
    df = bm_mktcap_group.groupby(['year','X1_group','X2_group'])['rt+1'].mean().reset_index()
    df = pd.pivot_table(df,index = ['year','X2_group'],columns = 'X1_group')['rt+1'].reset_index()
    for i in range(1988,2012):
        x = df[df['year'] == i]
        x1 = pd.DataFrame(columns = x.columns)
        x1['year'] = [i]
        x1['X2_group'] = ['mktcap_diff']
        x1.iloc[0,2:] = x.iloc[3,2:]-x.iloc[0,2:]
        x = pd.concat([x,x1])
        X = pd.concat([X,x])
    X['bm_avg'] = (X['bm1']+X['bm2']+X['bm3'])/3
    X['bm_diff'] = X['bm3'] - X['bm1']
    return X
bm_mktcap_annual_mean = bm_mktcap_mean().reset_index(drop=True)
bm_mktcap_total_mean = beta_mktcap_excess(bm_mktcap_annual_mean)


#生成表格25
def showtable25():
    x1 = beta_mktcap_total_mean.iloc[-4:,:]
    x2 = bm_mktcap_total_mean.iloc[-4:,:]
    col = ['1','2','3','avg','3-1']
    inx = ['Excess return',' ','FFC alpha',' ']
    x1.columns = col
    x2.columns = col
    x1.index = inx
    x2.index = inx
    table = pd.concat([x1,x2],keys=['beta','bm'])
    table = table.applymap(lambda x:round(x, 2))
    print(table)
showtable25()

#生成表格26
def showtable26():
    col = ['MktCap1','MktCap2','MktCap3','MktCap4','MktCap4-1']
    inx = ['Excess return',' ','FFC alpha',' ']
    x1 = beta_mktcap_total_mean['beta_avg']
    temp_beta = pd.DataFrame(columns=col)
    for i in range(4):
        temp = x1.iloc[i::4]
        temp.index = col
        temp.name = inx[i]
        temp_beta = temp_beta.append(temp)
    x2 = bm_mktcap_total_mean['bm_avg']
    temp_bm = pd.DataFrame(columns=col)
    for j in range(4):
        temp = x2.iloc[j::4]
        temp.index = col
        temp.name = inx[j]
        temp_bm = temp_bm.append(temp)
    table = pd.concat([temp_beta,temp_bm],keys=['beta','bm'])
    table = table.applymap(lambda x:round(x, 2))
    print(table)
showtable26()

#用bm替代mktcap如法炮制
breakpoint_dep_beta_bm = get_dep_3x4_break(beta,bm)
beta_bm_group = portfolio_dep_3x4(breakpoint_dep_beta_bm,beta,bm)
beta_bm_group = pd.merge(beta_bm_group,ret,on=['permno','year'])
def beta_bm_mean():
    X = pd.DataFrame()
    df = beta_bm_group.groupby(['year','X1_group','X2_group'])['rt+1'].mean().reset_index()
    df = pd.pivot_table(df,index = ['year','X2_group'],columns = 'X1_group')['rt+1'].reset_index()
    for i in range(1988,2012):
        x = df[df['year'] == i]
        x1 = pd.DataFrame(columns = x.columns)
        x1['year'] = [i]
        x1['X2_group'] = ['bm_diff']
        x1.iloc[0,2:] = x.iloc[3,2:]-x.iloc[0,2:]
        x = pd.concat([x,x1])
        X = pd.concat([X,x])
    X['beta_avg'] = (X['beta1']+X['beta2']+X['beta3'])/3
    X['beta_diff'] = X['beta3']-X['beta1']
    return X
beta_bm_annual_mean = beta_bm_mean().reset_index(drop=True)
beta_bm_total_mean = beta_mktcap_excess(beta_bm_annual_mean)

#生成表格27
def showtable27():
    x1 = beta_mktcap_total_mean.iloc[-4:,:]
    x2 = beta_bm_total_mean.iloc[-4:,:]
    col = ['beta1','beta2','beta3','beta avg','beta3-1']
    inx = ['Excess return',' ','FFC alpha',' ']
    x1.columns = col
    x2.columns = col
    x1.index = inx
    x2.index = inx
    table = pd.concat([x1,x2],keys=['MktCap','BM'])
    table = table.applymap(lambda x:round(x, 2))
    print(table)
showtable27()

#生成表格28
def showtable28():
    col = ['1','2','3','4','4-1']
    inx = ['Excess return',' ','FFC alpha',' ']
    x1 = beta_mktcap_total_mean['beta_avg']
    temp_mktcap = pd.DataFrame(columns=col)
    for i in range(4):
        temp = x1.iloc[i::4]
        temp.index = col
        temp.name = inx[i]
        temp_mktcap = temp_mktcap.append(temp)
    x2 = beta_bm_total_mean['beta_avg']
    temp_bm = pd.DataFrame(columns=col)
    for j in range(4):
        temp = x2.iloc[j::4]
        temp.index = col
        temp.name = inx[j]
        temp_bm = temp_bm.append(temp)
    table = pd.concat([temp_mktcap,temp_bm],keys=['MktCap','BM'])
    table = table.applymap(lambda x:round(x, 2))
    print(table)
showtable28()

#生成表格29
def showtable29():
    data = portfolio_ind_3x4(table10,'beta','mktcap')
    table29 = pd.pivot_table(data,index = 'X2_group',columns = 'X1_group')['mktcap']
    table29 = table29.applymap(lambda x : int(x))
    print(table29)
showtable29()

#生成表格30
def showtable30():
    table = pd.pivot_table(beta_mktcap_group,index = 'X2_group',columns = 'X1_group')['mktcap']
    table = table.applymap(lambda x : int(x))
    print(table)
showtable30()