%%%%%%%%%%%%%%%%%%%%%%%
% Forecasts_quarterly.m
%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 07-02-2012

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data and defining variables, 1926:4-2010:4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Equity premium

market_return=xlsread('Returns_handbook_data','Quarterly',...
    'p225:p561'); % S&P 500 VW return
r_f_lag=xlsread('Returns_handbook_data','Quarterly',...
    'k224:k560'); % risk-free rate, lagged (1926:3-2010:3)
equity_premium=market_return-r_f_lag; % excess return

% Predictors

D4=xlsread('Returns_handbook_data','Quarterly','c225:c561'); % dividends
SP500=xlsread('Returns_handbook_data','Quarterly',...
    'b225:b561'); % S&P 500 index
DP=log(D4)-log(SP500); % log dividend-price ratio
SP500_lag=xlsread('Returns_handbook_data','Quarterly',...
    'b224:b560'); % S&P 500 index, lagged (1926:3-2010:3)
DY=log(D4)-log(SP500_lag); % log dividend yield
E4=xlsread('Returns_handbook_data','Quarterly','d225:d561'); % earnings
EP=log(E4)-log(SP500); % log earnings-price ratio
DE=log(D4)-log(E4); % log dividend-payout ratio
SVAR=xlsread('Returns_handbook_data','Quarterly','o225:o561'); % volatility
BM=xlsread('Returns_handbook_data','Quarterly',...
    'e225:e561'); % book-to-market ratio
NTIS=xlsread('Returns_handbook_data','Quarterly',...
    'j225:j561'); % net equity issuing activity
TBL=xlsread('Returns_handbook_data','Quarterly',...
    'f225:f561'); % T-bill rate
LTY=xlsread('Returns_handbook_data','Quarterly',...
    'i225:i561'); % long-term government bond yield
LTR=xlsread('Returns_handbook_data','Quarterly',...
    'm225:m561'); % long-term government bond return
TMS=LTY-TBL; % term spread
AAA=xlsread('Returns_handbook_data','Quarterly',...
    'g225:g561'); % AAA-rated corporate bond yield
BAA=xlsread('Returns_handbook_data','Quarterly',...
    'h225:h561'); % BAA-rated corporate bond yield
DFY=BAA-AAA; % default yield spread
CORPR=xlsread('Returns_handbook_data','Quarterly',...
    'n225:n561'); % long-term corporate bond return
DFR=CORPR-LTR; % default return spread
INFL_lag=xlsread('Returns_handbook_data','Quarterly',...
    'l224:l560'); % inflation, lagged (1926:3-2010:3)
ECON=[DP DY EP DE SVAR BM NTIS TBL LTY LTR TMS DFY DFR INFL_lag];
ECON_sink=[DP DY EP SVAR BM NTIS TBL LTY LTR DFY DFR INFL_lag];

% Sum-of-the-parts variables

E4_lag=xlsread('Returns_handbook_data','Quarterly',...
    'd224:d560'); % earnings, lagged (1926:3-2010:3)
E_growth=(E4-E4_lag)./E4_lag; % earnings growth
DP_SOP=(1/4)*D4./SP500; % D/P
r_f=xlsread('Returns_handbook_data','Quarterly',...
    'k225:k561'); % risk-free rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating full-sample parameters for Campbell-Thompson restrictions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta_full=zeros(size(ECON,2),1);
T=size(equity_premium,1);
LHS=equity_premium(2:T);
for i=1:size(ECON,2);
    RHS_i=[ECON(1:T-1,i) ones(T-1,1)];
    results_i=ols(LHS,RHS_i);
    beta_full(i)=results_i.beta(1);
    disp(i); 
end;
beta_full(5)=1; % restricting SVAR slope coefficient to be positive

% NB: following Campbell and Thompson (2008) by using full-sample
% estimates as theoretical 'priors' (although we make sure that
% SVAR slope coefficient is positive)

%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

Y=equity_premium;
T=size(Y,1);
N=size(ECON,2);
R=(1946-1926)*4+1; % in-sample period, 1926:4-1946:4
P_0=(1956-1946)*4; % holdout out-of-sample period, 1947:1-1956:4
P=T-(R+P_0); % forecast evaluation period, 1957:1-2010:4
theta=0.75; % discound factor for DMSFE pooled forecast
r=1; % number of principal components
MA_SOP=20*4; % window size for SOP forecast

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing forecasts, 1947:1-2010:4 (includes holdout OOS period)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FC_HA=zeros(P_0+P,1);
FC_ECON=zeros(P_0+P,N);
beta_ECON=zeros(P_0+P,N,2);
FC_ECON_CT=zeros(P_0+P,N);
FC_OTHER=zeros(P_0+P,5+size(theta,1));
FC_OTHER_CT=zeros(P_0+P,5+size(theta,1));
for t=1:P_0+P;
    FC_HA(t)=mean(Y(1:R+(t-1)));

    % Individual predictive regression forecasts

    X_t=ECON(1:R+(t-1)-1,:);
    Y_t=Y(2:R+(t-1));
    for i=1:N;
        results_t_i=ols(Y_t,[X_t(:,i) ones(R+(t-1)-1,1)]);
        FC_ECON(t,i)=[ECON(R+(t-1),i) 1]*results_t_i.beta;
        beta_ECON(t,i,1)=results_t_i.beta(1);
        beta_ECON(t,i,2)=results_t_i.bstd(1);
        if beta_full(i)>0;
            if results_t_i.beta(1)>0;
                FC_ECON_CT(t,i)=FC_ECON(t,i);
            elseif results_t_i.beta(1)<0;
                FC_ECON_CT(t,i)=FC_HA(t);
            end;
        elseif beta_full(i)<0;
            if results_t_i.beta(1)<0;
                FC_ECON_CT(t,i)=FC_ECON(t,i);
            elseif results_t_i.beta(1)>0;
                FC_ECON_CT(t,i)=FC_HA(t);
            end;
        end;
        if FC_ECON_CT(t,i)<0;
            FC_ECON_CT(t,i)=0;
        end;
    end;
    if t>P_0;

        % Kitchen sink forecast

        X_t_sink=ECON_sink(1:R+(t-1)-1,:);
        results_t_sink=ols(Y_t,[X_t_sink ones(R+(t-1)-1,1)]);
        FC_OTHER(t,1)=[ECON_sink(R+(t-1),:) 1]*results_t_sink.beta;
        if FC_OTHER(t,1)<0;
            FC_OTHER_CT(t,1)=0;
        else
            FC_OTHER_CT(t,1)=FC_OTHER(t,1);
        end;

        % SIC forecast

        j_max=3; % consider models with up to 3 predictors to save time
        SIC_t=[];
        for j=1:j_max;
            select_j=nchoosek(1:1:size(X_t_sink,2),j);
            for k=1:size(select_j,1);
                X_t_j_k=[X_t_sink(1:R+(t-1)-1,select_j(k,:)) ...
                    ones(R+(t-1)-1,1)];
                results_t_j_k=ols(Y_t,X_t_j_k);
                SIC_t_j_k=log(results_t_j_k.resid'*results_t_j_k.resid/...
                    size(Y_t,1))+log(size(Y_t,1))*size(X_t_j_k,2)/...
                    size(Y_t,1);
                FC_t_j_k=[ECON_sink(R+(t-1),select_j(k,:)) 1]*...
                    results_t_j_k.beta;
                SIC_t=[SIC_t ; SIC_t_j_k FC_t_j_k];
            end;
        end;
        [SIC_t_min,SIC_t_min_index]=min(SIC_t(:,1));
        FC_OTHER(t,2)=SIC_t(SIC_t_min_index,2);
        if FC_OTHER(t,2)<0;
            FC_OTHER_CT(t,2)=0;
        else
            FC_OTHER_CT(t,2)=FC_OTHER(t,2);
        end;

        % Pooled forecast: simple average

        FC_OTHER(t,3)=mean(FC_ECON(t,:),2);
        if FC_OTHER(t,3)<0;
            FC_OTHER_CT(t,3)=0;
        else
            FC_OTHER_CT(t,3)=FC_OTHER(t,3);
        end;

        % Pooled forecast: DMSFE

        powers_t=(t-2:-1:0)';
        m=sum((kron(ones(1,N),(theta*ones(t-1,1)).^powers_t)).*...
            ((kron(ones(1,N),Y(R+1:R+(t-1)))-FC_ECON(1:(t-1),:)).^2))';
        omega=(m.^(-1))/(sum(m.^(-1)));
        FC_OTHER(t,4)=FC_ECON(t,:)*omega;
        if FC_OTHER(t,4)<0;
            FC_OTHER_CT(t,4)=0;
        else
            FC_OTHER_CT(t,4)=FC_OTHER(t,4);
        end;

        % Diffusion index forecast

        X_t_DI_standardize=zscore(ECON(1:R+(t-1),:));
        [Lambda_t F_t]=princomp(X_t_DI_standardize);
        results_t_DI=ols(Y_t,[F_t(1:end-1,1:r) ones(R+(t-1)-1,1)]);
        FC_OTHER(t,5)=[F_t(end,1:r) 1]*results_t_DI.beta;
        if FC_OTHER(t,5)<0;
            FC_OTHER_CT(t,5)=0;
        else
            FC_OTHER_CT(t,5)=FC_OTHER(t,5);
        end;

        % Sum-of-the-parts forecast

        FC_OTHER(t,6)=mean(E_growth(R+(t-1)-MA_SOP+1:R+(t-1)))+...
            DP_SOP(R+(t-1))-r_f(R+(t-1));
        if FC_OTHER(t,6)<0;
            FC_OTHER_CT(t,6)=0;
        else
            FC_OTHER_CT(t,6)=FC_OTHER(t,6);
        end;
    end;
    disp([t FC_HA(t) FC_OTHER(t,:)]); 
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forecast evaluation based on asselt allocation exercise, 1957:1-2010:4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

actual=Y(R+P_0+1:end);
FC_HA=FC_HA(P_0+1:end);
FC_ECON=FC_ECON(P_0+1:end,:);
FC_ECON_CT=FC_ECON_CT(P_0+1:end,:);
FC_OTHER=FC_OTHER(P_0+1:end,:);
FC_OTHER_CT=FC_OTHER_CT(P_0+1:end,:);
REC=xlsread('Returns_handbook_data','Quarterly',...
    's346:s561'); % recession dummies, 1957:1-2010:4
EXP=-1*(REC-ones(size(REC,1),1));
index_EXP=find(EXP);
index_REC=find(REC);
gamma_MV=5; % coefficient of relative risk aversion
window_VOL=4*5; % window size for estimating volatility
FC_VOL=zeros(P,1);
for t=1:P;
    FC_VOL(t)=mean(Y(R+P_0+(t-1)-window_VOL+1:R+P_0+(t-1)).^2)-...
        (mean(Y(R+P_0+(t-1)-window_VOL+1:R+P_0+(t-1))))^2;
end;
r_f_lag_P=r_f_lag(R+P_0+1:R+P_0+P);
U_HA=zeros(3,1);
[U_HA(1),w_HA]=Perform_asset_allocation(actual,r_f_lag_P,FC_HA,FC_VOL,...
    gamma_MV);
[U_HA(2),xxx]=Perform_asset_allocation(actual(index_EXP),...
    r_f_lag_P(index_EXP),FC_HA(index_EXP),FC_VOL(index_EXP),gamma_MV);
[U_HA(3),xxx]=Perform_asset_allocation(actual(index_REC),...
    r_f_lag_P(index_REC),FC_HA(index_REC),FC_VOL(index_REC),gamma_MV);
U_ECON=zeros(size(FC_ECON,2),3);
U_ECON_CT=zeros(size(FC_ECON_CT,2),3);
w_ECON=zeros(P,size(FC_ECON,2));
w_ECON_CT=zeros(P,size(FC_ECON_CT,2));
for i=1:size(U_ECON,1);

    % Overall

    [U_ECON(i,1),w_ECON(:,i)]=Perform_asset_allocation(actual,...
        r_f_lag_P,FC_ECON(:,i),FC_VOL,gamma_MV);
    [U_ECON_CT(i,1),w_ECON_CT(:,i)]=Perform_asset_allocation(actual,...
        r_f_lag_P,FC_ECON_CT(:,i),FC_VOL,gamma_MV);

    % Expansion

    [U_ECON(i,2),xxx]=Perform_asset_allocation(actual(index_EXP),...
        r_f_lag_P(index_EXP),FC_ECON(index_EXP,i),FC_VOL(index_EXP),...
        gamma_MV);
    [U_ECON_CT(i,2),xxx]=Perform_asset_allocation(actual(index_EXP),...
        r_f_lag_P(index_EXP),FC_ECON_CT(index_EXP,i),FC_VOL(index_EXP),...
        gamma_MV);

    % Recession

    [U_ECON(i,3),xxx]=Perform_asset_allocation(actual(index_REC),...
        r_f_lag_P(index_REC),FC_ECON(index_REC,i),FC_VOL(index_REC),...
        gamma_MV);
    [U_ECON_CT(i,3),xxx]=Perform_asset_allocation(actual(index_REC),...
        r_f_lag_P(index_REC),FC_ECON_CT(index_REC,i),FC_VOL(index_REC),...
        gamma_MV);
end;
delta_ECON=400*(U_ECON-kron(U_HA',ones(size(FC_ECON,2),1)));
delta_ECON_CT=400*(U_ECON_CT-kron(U_HA',ones(size(FC_ECON_CT,2),1)));
U_OTHER=zeros(size(FC_OTHER,2),3);
U_OTHER_CT=zeros(size(FC_OTHER_CT,2),3);
w_OTHER=zeros(P,size(FC_OTHER,2));
w_OTHER_CT=zeros(P,size(FC_OTHER_CT,2));
for i=1:size(U_OTHER,1);

    % Overall

    [U_OTHER(i,1),w_OTHER(:,i)]=Perform_asset_allocation(actual,...
        r_f_lag_P,FC_OTHER(:,i),FC_VOL,gamma_MV);
    [U_OTHER_CT(i,1),w_OTHER_CT(:,i)]=Perform_asset_allocation(actual,...
        r_f_lag_P,FC_OTHER_CT(:,i),FC_VOL,gamma_MV);

    % Expansion

    [U_OTHER(i,2),xxx]=Perform_asset_allocation(actual(index_EXP),...
        r_f_lag_P(index_EXP),FC_OTHER(index_EXP,i),FC_VOL(index_EXP),...
        gamma_MV);
    [U_OTHER_CT(i,2),xxx]=Perform_asset_allocation(actual(index_EXP),...
        r_f_lag_P(index_EXP),FC_OTHER_CT(index_EXP,i),FC_VOL(index_EXP),...
        gamma_MV);

    % Recession

    [U_OTHER(i,3),xxx]=Perform_asset_allocation(actual(index_REC),...
        r_f_lag_P(index_REC),FC_OTHER(index_REC,i),FC_VOL(index_REC),...
        gamma_MV);
    [U_OTHER_CT(i,3),xxx]=Perform_asset_allocation(actual(index_REC),...
        r_f_lag_P(index_REC),FC_OTHER_CT(index_REC,i),FC_VOL(index_REC),...
        gamma_MV);
end;
delta_OTHER=400*(U_OTHER-kron(U_HA',ones(size(FC_OTHER,2),1)));
delta_OTHER_CT=400*(U_OTHER_CT-kron(U_HA',ones(size(FC_OTHER_CT,2),1)));
%save('Forecasts_quarterly_store','actual','FC_HA','FC_ECON','beta_ECON',...
%    'FC_ECON_CT','FC_OTHER','FC_OTHER_CT','FC_VOL','w_HA','w_ECON',...
%    'w_ECON_CT','w_OTHER','w_OTHER_CT','delta_ECON','delta_ECON_CT',...
%    'delta_OTHER_CT');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_handbook_results',delta_ECON(:,1),...
%    'Quarterly','d5');
%xlswrite('Returns_handbook_results',delta_OTHER(:,1),...
%    'Quarterly','d20');
%xlswrite('Returns_handbook_results',delta_ECON_CT(:,1),...
%    'Quarterly','p5');
%xlswrite('Returns_handbook_results',delta_OTHER_CT(:,1),...
%    'Quarterly','p20');
%xlswrite('Returns_handbook_results',delta_ECON(:,2),...
%    'Quarterly','h5');
%xlswrite('Returns_handbook_results',delta_OTHER(:,2),...
%    'Quarterly','h20');
%xlswrite('Returns_handbook_results',delta_ECON_CT(:,2),...
%    'Quarterly','t5');
%xlswrite('Returns_handbook_results',delta_OTHER_CT(:,2),...
%    'Quarterly','t20');
%xlswrite('Returns_handbook_results',delta_ECON(:,3),...
%    'Quarterly','l5');
%xlswrite('Returns_handbook_results',delta_OTHER(:,3),...
%    'Quarterly','l20');
%xlswrite('Returns_handbook_results',delta_ECON_CT(:,3),...
%    'Quarterly','x5');
%xlswrite('Returns_handbook_results',delta_OTHER_CT(:,3),...
%    'Quarterly','x20');
