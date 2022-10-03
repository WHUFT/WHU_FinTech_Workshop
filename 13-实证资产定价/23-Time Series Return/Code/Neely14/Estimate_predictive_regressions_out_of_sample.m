%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate_predictive_regressions_out_of_sample.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data/defining variables, 1950:12-2011:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='Equity premium';
y=xlsread(input_file,input_sheet,'b289:b1021');
y=100*y; % percent equity premium
input_sheet='Macroeconomic variables';
X_ECON=xlsread(input_file,input_sheet,'b289:o1021');
r_f=xlsread(input_file,input_sheet,'q289:q1021');
E12=xlsread(input_file,input_sheet,'r289:r1021');
E12_lag=xlsread(input_file,input_sheet,'r288:r1020');
E_growth=log((1/12)*E12)-log((1/12)*E12_lag);
DP=exp(X_ECON(:,1));
DP_SOP=log(1+(1/12)*DP);
input_sheet='Technical indicators';
X_TECH=xlsread(input_file,input_sheet,'b289:o1021');

% Adjusting economic variables where necessary for positive expected slope

X_ECON(:,7)=-X_ECON(:,7);
X_ECON(:,8)=-X_ECON(:,8);
X_ECON(:,9)=-X_ECON(:,9);
X_ECON(:,14)=-X_ECON(:,14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constructing out-of-sample forecasts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preliminaries

disp('Generating out-of-sample forecasts');
T=size(y,1);
N_ECON=size(X_ECON,2);
N_TECH=size(X_TECH,2);
N_ALL=N_ECON+N_TECH;
R=(1965-1950)*12+1; % 1950:12-1965:12 in-sample period
P=T-R; % forecast evaluation period starts in 1966:01
actual=nan(P,1);
FC_HA=nan(P,1);
FC_ECON=nan(P,N_ECON+2);
FC_TECH=nan(P,N_TECH+2);
FC_ALL=nan(P,2);
k_max_ECON=3;
k_max_TECH=1;
k_max_ALL=4;
IC=3;
FC_SOP=nan(P,1);
MA_SOP=20*12;

% Computing recursive forecasts

for t=1:P;
    actual(t)=y(R+t);
    y_t=y(1:R+(t-1));
    FC_HA(t)=mean(y_t);
    X_ECON_t=X_ECON(1:R+(t-1),:);
    X_TECH_t=X_TECH(1:R+(t-1),:);

    % Bivariate predictive regression forecasts: macroeconomic variables

    for i=1:N_ECON;
        results_ECON_i_t=ols(y_t(2:end),...
            [ones(R+(t-2),1) X_ECON_t(1:end-1,i)]);
        FC_ECON(t,i)=[1 X_ECON_t(end,i)]*results_ECON_i_t.beta;
    end;

    % Bivariate predictive regression forecasts: technical indicators

    for i=1:N_TECH;
        results_TECH_i_t=ols(y_t(2:end),...
            [ones(R+(t-2),1) X_TECH_t(1:end-1,i)]);
        FC_TECH(t,i)=[1 X_TECH_t(end,i)]*results_TECH_i_t.beta;
    end;

    % Additional forecasts based on all macroeconomic variables

    FC_ECON(t,N_ECON+1)=mean(FC_ECON(t,1:N_ECON),2);
    [Lambda_ECON_t,F_hat_ECON_t]=princomp(zscore(X_ECON_t));
    [k_ECON_t]=Perform_selection_IC(y_t(2:end),...
        F_hat_ECON_t(1:end-1,1:k_max_ECON),IC);
    F_hat_ECON_t=F_hat_ECON_t(:,1:k_ECON_t);
    results_ECON_PC_t=ols(y_t(2:end),...
        [ones(R+(t-2),1) F_hat_ECON_t(1:end-1,:)]);
    FC_ECON(t,N_ECON+2)=[1 F_hat_ECON_t(end,:)]*results_ECON_PC_t.beta;

    % Additional forecasts based on all technical indicators

    FC_TECH(t,N_TECH+1)=mean(FC_TECH(t,1:N_TECH),2);
    [Lambda_TECH_t,F_hat_TECH_t]=princomp(zscore(X_TECH_t));
    [k_TECH_t]=Perform_selection_IC(y_t(2:end),...
        F_hat_TECH_t(1:end-1,1:k_max_TECH),IC);
    F_hat_TECH_t=F_hat_TECH_t(:,1:k_TECH_t);
    results_TECH_PC_t=ols(y_t(2:end),...
        [ones(R+(t-2),1) F_hat_TECH_t(1:end-1,:)]);
    FC_TECH(t,N_TECH+2)=[1 F_hat_TECH_t(end,:)]*results_TECH_PC_t.beta;

    % Additional forecasts based on all predictors taken together

    FC_ALL(t,1)=mean([FC_ECON(t,1:N_ECON) FC_TECH(t,1:N_TECH)],2);
    [Lambda_ALL_t,F_hat_ALL_t]=princomp(zscore([X_ECON_t X_TECH_t]));
    [k_ALL_t]=Perform_selection_IC(y_t(2:end),...
        F_hat_ALL_t(1:end-1,1:k_max_ALL),IC);
    F_hat_ALL_t=F_hat_ALL_t(:,1:k_ALL_t);
    results_ALL_PC_t=ols(y_t(2:end),...
        [ones(R+(t-2),1) F_hat_ALL_t(1:end-1,:)]);
    FC_ALL(t,2)=[1 F_hat_ALL_t(end,:)]*results_ALL_PC_t.beta;
 
    % Sum-of-the-parts forecast

    if R+(t-1)>=MA_SOP;
        FC_SOP(t)=100*(mean(E_growth(R+(t-1)-MA_SOP+1:R+(t-1)))+...
            DP_SOP(R+(t-1))-r_f(R+(t-1)));
    else
        FC_SOP(t)=100*(mean(E_growth(1:R+(t-1)))+...
            DP_SOP(R+(t-1))-r_f(R+(t-1)));
    end;
    disp([t actual(t) FC_HA(t)]);
    disp([FC_ECON(t,end) k_ECON_t]);
    disp([FC_TECH(t,end) k_TECH_t]);
    disp([FC_ALL(t,end) k_ALL_t]);
end;
output_file='Estimate_predictive_regressions_out_of_sample';
save(output_file,'actual','FC_HA','FC_ECON','FC_TECH','FC_ALL');

%%%%%%%%%%%%%%%%%%%%%%
% Evaluating forecasts
%%%%%%%%%%%%%%%%%%%%%%

% Preliminaries

disp('Evaluating forecasts');
results_ECON=nan(size(FC_ECON,2),6);
results_ECON_exp=nan(size(FC_ECON,2),3);
results_ECON_rec=nan(size(FC_ECON,2),3);
results_TECH=nan(size(FC_TECH,2),6);
results_TECH_exp=nan(size(FC_TECH,2),3);
results_TECH_rec=nan(size(FC_TECH,2),3);
results_ALL=zeros(size(FC_ALL,2),6);
results_ALL_exp=nan(size(FC_ALL,2),3);
results_ALL_rec=nan(size(FC_ALL,2),3);
input_sheet='Equity premium';
recession=xlsread(input_file,input_sheet,'d470:d1021');
expansion=-1*(recession-ones(size(recession,1),1));
index_exp=find(expansion);
index_rec=find(recession);
actual_exp=actual(index_exp);
actual_rec=actual(index_rec);
FC_HA_exp=FC_HA(index_exp);
FC_HA_rec=FC_HA(index_rec);
FC_ECON_exp=FC_ECON(index_exp,:);
FC_ECON_rec=FC_ECON(index_rec,:);
FC_TECH_exp=FC_TECH(index_exp,:);
FC_TECH_rec=FC_TECH(index_rec,:);
FC_ALL_exp=FC_ALL(index_exp,:);
FC_ALL_rec=FC_ALL(index_rec,:);
FC_SOP_exp=FC_SOP(index_exp,:);
FC_SOP_rec=FC_SOP(index_rec,:);

% MSFE criterion, historical average

MSFE_HA=mean((actual-FC_HA).^2);
MSFE_HA_exp=mean((actual_exp-FC_HA_exp).^2);
MSFE_HA_rec=mean((actual_rec-FC_HA_rec).^2);
bias_HA=mean(actual-FC_HA);
decomp_HA=[bias_HA^2 MSFE_HA-bias_HA^2];
FC_6=6/12;
MSFE_6=mean((actual-FC_6).^2);

% MSFE criterion, macroeconomic variables

for i=1:size(FC_ECON,2);
    MSFE_ECON_i=mean((actual-FC_ECON(:,i)).^2);
    R2OS_ECON_i=100*(1-(MSFE_ECON_i/MSFE_HA));
    [MSFE_adjusted_ECON_i,p_value_ECON_i]=Perform_CW_test(actual,...
        FC_HA,FC_ECON(:,i));
    bias_ECON_i=mean(actual-FC_ECON(:,i));
    decomp_ECON_i=[bias_ECON_i^2 MSFE_ECON_i-bias_ECON_i^2];
    results_ECON(i,:)=[MSFE_ECON_i R2OS_ECON_i MSFE_adjusted_ECON_i ...
        p_value_ECON_i decomp_ECON_i];
    MSFE_ECON_i_exp=mean((actual_exp-FC_ECON_exp(:,i)).^2);
    R2OS_ECON_i_exp=100*(1-(MSFE_ECON_i_exp/MSFE_HA_exp));
    [MSFE_adjusted_ECON_i_exp,p_value_ECON_i_exp]=...
        Perform_CW_test(actual_exp,FC_HA_exp,FC_ECON_exp(:,i));
    results_ECON_exp(i,:)=[R2OS_ECON_i_exp MSFE_adjusted_ECON_i_exp ...
        p_value_ECON_i_exp];
    MSFE_ECON_i_rec=mean((actual_rec-FC_ECON_rec(:,i)).^2);
    R2OS_ECON_i_rec=100*(1-(MSFE_ECON_i_rec/MSFE_HA_rec));
    [MSFE_adjusted_ECON_i_rec,p_value_ECON_i_rec]=...
        Perform_CW_test(actual_rec,FC_HA_rec,FC_ECON_rec(:,i));
    results_ECON_rec(i,:)=[R2OS_ECON_i_rec MSFE_adjusted_ECON_i_rec ...
        p_value_ECON_i_rec];
end;

% MSFE criterion, technical indicators

for i=1:size(FC_TECH,2);
    MSFE_TECH_i=mean((actual-FC_TECH(:,i)).^2);
    R2OS_TECH_i=100*(1-(MSFE_TECH_i/MSFE_HA));
    [MSFE_adjusted_TECH_i,p_value_TECH_i]=Perform_CW_test(actual,...
        FC_HA,FC_TECH(:,i));
    bias_TECH_i=mean(actual-FC_TECH(:,i));
    decomp_TECH_i=[bias_TECH_i^2 MSFE_TECH_i-bias_TECH_i^2];
    results_TECH(i,:)=[MSFE_TECH_i R2OS_TECH_i MSFE_adjusted_TECH_i ...
        p_value_TECH_i decomp_TECH_i];
    MSFE_TECH_i_exp=mean((actual_exp-FC_TECH_exp(:,i)).^2);
    R2OS_TECH_i_exp=100*(1-(MSFE_TECH_i_exp/MSFE_HA_exp));
    [MSFE_adjusted_TECH_i_exp,p_value_TECH_i_exp]=...
        Perform_CW_test(actual_exp,FC_HA_exp,FC_TECH_exp(:,i));
    results_TECH_exp(i,:)=[R2OS_TECH_i_exp MSFE_adjusted_TECH_i_exp ...
        p_value_TECH_i_exp];
    MSFE_TECH_i_rec=mean((actual_rec-FC_TECH_rec(:,i)).^2);
    R2OS_TECH_i_rec=100*(1-(MSFE_TECH_i_rec/MSFE_HA_rec));
    [MSFE_adjusted_TECH_i_rec,p_value_TECH_i_rec]=...
        Perform_CW_test(actual_rec,FC_HA_rec,FC_TECH_rec(:,i));
    results_TECH_rec(i,:)=[R2OS_TECH_i_rec MSFE_adjusted_TECH_i_rec ...
        p_value_TECH_i_rec];
end;

% MSFE criterion, all predictors taken together

for i=1:size(FC_ALL,2);
    MSFE_ALL_i=mean((actual-FC_ALL(:,i)).^2);
    R2OS_ALL_i=100*(1-(MSFE_ALL_i/MSFE_HA));
    [MSFE_adjusted_ALL_i,p_value_ALL_i]=...
        Perform_CW_test(actual,FC_HA,FC_ALL(:,i));
    bias_ALL_i=mean(actual-FC_ALL(:,i));
    decomp_ALL_i=[bias_ALL_i^2 MSFE_ALL_i-bias_ALL_i^2];
    results_ALL(i,:)=[MSFE_ALL_i R2OS_ALL_i MSFE_adjusted_ALL_i ...
        p_value_ALL_i decomp_ALL_i];
    MSFE_ALL_i_exp=mean((actual_exp-FC_ALL_exp(:,i)).^2);
    R2OS_ALL_i_exp=100*(1-(MSFE_ALL_i_exp/MSFE_HA_exp));
    [MSFE_adjusted_ALL_i_exp,p_value_ALL_i_exp]=...
        Perform_CW_test(actual_exp,FC_HA_exp,FC_ALL_exp(:,i));
    results_ALL_exp(i,:)=[R2OS_ALL_i_exp MSFE_adjusted_ALL_i_exp ...
        p_value_ALL_i_exp];
    MSFE_ALL_i_rec=mean((actual_rec-FC_ALL_rec(:,i)).^2);
    R2OS_ALL_i_rec =100*(1-(MSFE_ALL_i_rec/MSFE_HA_rec));
    [MSFE_adjusted_ALL_i_rec,p_value_ALL_i_rec]=...
        Perform_CW_test(actual_rec,FC_HA_rec,FC_ALL_rec(:,i));
    results_ALL_rec(i,:)=[R2OS_ALL_i_rec MSFE_adjusted_ALL_i_rec ...
        p_value_ALL_i_rec];
end;

% MSFE criterion, sum-of-the-parts

MSFE_SOP=mean((actual-FC_SOP).^2);
R2OS_SOP=100*(1-(MSFE_SOP/MSFE_HA));
[MSFE_adjusted_SOP,p_value_SOP]=Perform_CW_test(actual,FC_HA,FC_SOP);
bias_SOP=mean(actual-FC_SOP);
decomp_SOP=[bias_SOP^2 MSFE_SOP-bias_SOP^2];
results_SOP=[MSFE_SOP R2OS_SOP MSFE_adjusted_SOP p_value_SOP decomp_SOP];

% Forecast encompassing tests

[lambda_ECON_TECH,MHLN_ECON_TECH,MHLN_p_value_ECON_TECH]=...
    Perform_HLN_test(actual,FC_ECON(:,end),FC_TECH(:,end));
[lambda_TECH_ECON,MHLN_TECH_ECON,MHLN_p_value_TECH_ECON]=...
    Perform_HLN_test(actual,FC_TECH(:,end),FC_ECON(:,end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Writing results to spreahdsheet');
output_file='Returns_econ_tech_results';
output_sheet='Out-of-sample results';

% Historical average

xlswrite(output_file,MSFE_HA,output_sheet,'b4');
xlswrite(output_file,decomp_HA,output_sheet,'f4');

% Forecasts based on macroeconomic variables

xlswrite(output_file,results_ECON,output_sheet,'b8');
xlswrite(output_file,results_ECON_exp,output_sheet,'m8');
xlswrite(output_file,results_ECON_rec,output_sheet,'r8');

% Forecasts based on technical indicators

xlswrite(output_file,results_TECH,output_sheet,'b27');
xlswrite(output_file,results_TECH_exp,output_sheet,'m27');
xlswrite(output_file,results_TECH_rec,output_sheet,'r27');

% Forecasts based on all variables taken together

xlswrite(output_file,results_ALL,output_sheet,'b44');
xlswrite(output_file,results_ALL_exp,output_sheet,'m44');
xlswrite(output_file,results_ALL_rec,output_sheet,'r44');

% Sum-of-the-parts forecast

xlswrite(output_file,results_SOP,output_sheet,'b59');

% Other results

xlswrite(output_file,[lambda_ECON_TECH ; MHLN_ECON_TECH ; ...
    MHLN_p_value_ECON_TECH],output_sheet,'b49');
xlswrite(output_file,[lambda_TECH_ECON ; MHLN_TECH_ECON ; ...
    MHLN_p_value_TECH_ECON],output_sheet,'b55');
xlswrite(output_file,MSFE_6,output_sheet,'b61');
