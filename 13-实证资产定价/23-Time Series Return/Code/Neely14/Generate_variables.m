%%%%%%%%%%%%%%%%%%%%%%
% Generate_variables.m
%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating equity premium, 1927:01-2011:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Generating equity premium');
input_file='Returns_econ_tech_data';
input_sheet='Monthly';
market_return=xlsread(input_file,input_sheet,'p674:p1693');
risk_free_lag=xlsread(input_file,input_sheet,'k673:k1692');
log_equity_premium=log(1+market_return)-log(1+risk_free_lag);
equity_premium=market_return-risk_free_lag;
output_file='Returns_econ_tech_results';
output_sheet='Equity premium';
xlswrite(output_file,[log_equity_premium equity_premium],...
    output_sheet,'b2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating macroeconomic variables, 1927:01-2011:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Generating macroeconomic variables');

% Dividend-price ratio (log)

D12=xlsread(input_file,input_sheet,'c674:c1693');
SP500=xlsread(input_file,input_sheet,'b674:b1693');
DP=log(D12)-log(SP500);

% Dividend yield (log)

SP500_lag=xlsread(input_file,input_sheet,'b673:b1692');
DY=log(D12)-log(SP500_lag);

% Earnings-price ratio (log)

E12=xlsread(input_file,input_sheet,'d674:d1693');
EP=log(E12)-log(SP500);

% Payout ratio (log)

DE=log(D12)-log(E12);

% Book-to-market ratio

BM=xlsread(input_file,input_sheet,'e674:e1693');

% Net equity expansion

NTIS=xlsread(input_file,input_sheet,'j674:j1693');

% Treasury bill rate (annual %)

TBL=xlsread(input_file,input_sheet,'f674:f1693');
TBL=100*TBL;

% Long-term yield (annual %)

LTY=xlsread(input_file,input_sheet,'i674:i1693');
LTY=100*LTY;

% Long-term return (%)

LTR=xlsread(input_file,input_sheet,'m674:m1693');
LTR=100*LTR;

% Term spread (annual %)

TMS=LTY-TBL;

% Default yield spread

AAA=xlsread(input_file,input_sheet,'g674:g1693');
BAA=xlsread(input_file,input_sheet,'h674:h1693');
DFY=100*(BAA-AAA);

% Default return spread

CORPR=xlsread(input_file,input_sheet,'n674:n1693');
DFR=100*CORPR-LTR;

% Inflation (%, lagged)

INFL=xlsread(input_file,input_sheet,'l673:l1692');
INFL=100*INFL;

% Equity risk premium volatility (Mele 2007, JFE)

T=size(equity_premium,1);
RVOL=nan(T,1);
for t=12:T;
    RVOL(t)=sqrt(pi/2)*sqrt(12)*(1/12)*sum(abs(equity_premium(t-11:t)));
end;

% Collecting economic variables

ECON=[DP DY EP DE RVOL BM NTIS TBL LTY LTR TMS DFY DFR INFL];
output_sheet='Macroeconomic variables';
xlswrite(output_file,ECON,output_sheet,'b2');
risk_free=xlsread(input_file,input_sheet,'k674:k1693');
E12=xlsread(input_file,input_sheet,'d674:d1693');
xlswrite(output_file,[risk_free E12],output_sheet,'q2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating technical indicators, various starting dates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Generating technical indicators');
SP500=xlsread(input_file,input_sheet,'b674:b1693'); % 1927:01-2011:12

% MA indicators, 1927:12-2011:12

n_0_MA=1*11; % initial in-sample period, 1927:01-1927:11
MA_short=[1 2 3];
MA_long=[9 12];
MA_parameters=nan(length(MA_short)*length(MA_long),2);
for iter_1=1:length(MA_short);
    for iter_2=1:length(MA_long);
        MA_parameters((iter_1-1)*length(MA_long)+iter_2,:)=...
            [MA_short(iter_1) MA_long(iter_2)];
    end;
end;
S_MA=zeros(T-n_0_MA,length(MA_parameters)); % 1927:12-2011:12
for iter_t=1:length(S_MA);
    for iter_i=1:length(MA_parameters);
        short=mean(SP500(n_0_MA+iter_t-(MA_parameters(iter_i,1)-1):...
            n_0_MA+iter_t));
        long=mean(SP500(n_0_MA+iter_t-(MA_parameters(iter_i,2)-1):...
            n_0_MA+iter_t));
        if short>long;
            S_MA(iter_t,iter_i)=1;
        end;
    end;
    disp([iter_t S_MA(iter_t,:)]);
end;
output_sheet='Technical indicators';
xlswrite(output_file,S_MA,output_sheet,'b13');

% Momentum indicators, 1928:01-2011:12

n_0_MOM=1*12; % initial in-sample period, 1927:01-1927:12
MOM_parameters=[9 12];
S_MOM=zeros(T-n_0_MOM,length(MOM_parameters)); % 1928:01-2011:12
for iter_t=1:length(S_MOM);
    for iter_i=1:length(MOM_parameters);
        P_difference=SP500(n_0_MOM+iter_t)-...
            SP500(n_0_MOM+iter_t-MOM_parameters(iter_i));
        if P_difference>=0;
            S_MOM(iter_t,iter_i)=1;
        end;
    end;
    disp([iter_t S_MOM(iter_t,:)]);
end;
xlswrite(output_file,S_MOM,output_sheet,'h14');

% Volume-based indicators, 1950:12-2011:12

volume=xlsread(input_file,input_sheet,'r950:s1693'); % 1950:01-2011:12
T_VOL=size(volume,1);
n_0_VOL=1*12-1; % initial in-sample period, 1950:01-1950:11
VOL_short=[1 2 3];
VOL_long=[9 12];
VOL_parameters=nan(length(VOL_short)*length(VOL_long),2);
for iter_1=1:length(VOL_short);
    for iter_2=1:length(VOL_long);
        VOL_parameters((iter_1-1)*length(VOL_long)+iter_2,:)=...
            [VOL_short(iter_1) VOL_long(iter_2)];
    end;
end;
OBV=zeros(T_VOL,1);
for t_VOL=2:T_VOL;
    P_change=SP500(T-T_VOL+t_VOL)-SP500(T-T_VOL+t_VOL-1);
    if P_change>=0;
        OBV(t_VOL)=volume(t_VOL);
    else
        OBV(t_VOL)=-volume(t_VOL);
    end;
end;
OBV=cumsum(OBV); % 1950:01-2011:12
S_VOL=zeros(T_VOL-n_0_VOL,size(VOL_parameters,1)); % 1950:12-2011:12
for iter_t=1:size(S_VOL,1);
    for iter_i=1:size(VOL_parameters,1);
        short=mean(OBV(n_0_VOL+iter_t-...
            (VOL_parameters(iter_i,1)-1):n_0_VOL+iter_t));
        long=mean(OBV(n_0_VOL+iter_t-...
            (VOL_parameters(iter_i,2)-1):n_0_VOL+iter_t));
        if short>long;
            S_VOL(iter_t,iter_i)=1;
        end;
    end;
    disp([iter_t S_VOL(iter_t,:)]);
end;
xlswrite(output_file,S_VOL,output_sheet,'j289');
