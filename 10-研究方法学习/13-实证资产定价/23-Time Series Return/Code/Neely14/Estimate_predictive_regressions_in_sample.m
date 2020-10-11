%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate_predictive_regressions_in_sample.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data/defining variables, 1950:12-2011:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='Equity premium';
y=xlsread(input_file,input_sheet,'b289:b1021');
y=y*100; % percent equity premium
input_sheet='Macroeconomic variables';
X_ECON=xlsread(input_file,input_sheet,'b289:o1021');
input_sheet='Technical indicators';
X_TECH=xlsread(input_file,input_sheet,'b289:o1021');

% Adjusting macroeconomic variables for positive expected slope

X_ECON(:,7)=-X_ECON(:,7); % net equity expansion
X_ECON(:,8)=-X_ECON(:,8); % Treasury bill rate
X_ECON(:,9)=-X_ECON(:,9); % long-term yield
X_ECON(:,14)=-X_ECON(:,14); % inflation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating predictive regressions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preliminaries

disp('Preliminaries');
T=length(y);
N_ECON=size(X_ECON,2);
N_TECH=size(X_TECH,2);
k_max_ECON=3;
k_max_TECH=3;
k_max_ALL=4;
IC=3; % adjusted R-squared to select # of principal components
input_file='Generate_wild_bootstrapped_pseudo_samples';
load(input_file);
B=size(y_star,2); % number of boostrap replications

% Principal component predictive regression: macroeconomic variables

disp('Computing PC predictive regression: macro variables');
[Lambda_hat_ECON,F_hat_ECON]=princomp(zscore(X_ECON));
[k_ECON]=Perform_selection_IC(y(2:end),...
    F_hat_ECON(1:end-1,1:k_max_ECON),IC);
F_hat_ECON=F_hat_ECON(:,1:k_ECON);
results_PC_ECON=nwest(y(2:T),[ones(T-1,1) F_hat_ECON(1:T-1,:)],0);

% Bivariate predictive regressions: macroeconomic variables

disp('Computing bivariate predictive regressions: macro variables');
results_ECON=nan(N_ECON+k_ECON,4);
for i=1:N_ECON;
    results_i=nwest(y(2:T),[ones(T-1,1) X_ECON(1:T-1,i)],0);
    results_ECON(i,[1 2 4])=[results_i.beta(2) results_i.tstat(2) ...
        results_i.rsqr];
    disp(i);
end;
results_ECON(N_ECON+1:end,1:2)=[results_PC_ECON.beta(2:end) ...
    results_PC_ECON.tstat(2:end)];
results_ECON(N_ECON+1,4)=results_PC_ECON.rsqr;

% Principal commponent predictive regression: technical indicators

disp('Computing PC predictive regression: technical indicators');
[Lambda_hat_TECH,F_hat_TECH]=princomp(zscore(X_TECH));
[k_TECH]=Perform_selection_IC(y(2:end),...
    F_hat_TECH(1:end-1,1:k_max_TECH),IC);
F_hat_TECH=F_hat_TECH(:,1:k_TECH);
results_PC_TECH=nwest(y(2:T),[ones(T-1,1) F_hat_TECH(1:T-1,:)],0);

% Bivariate predictive regressions: technical indicators

disp('Computing bivariate predictive regressions: technical indicators');
results_TECH=nan(N_TECH+k_TECH,4);
for i=1:N_TECH;
    results_i=nwest(y(2:T),[ones(T-1,1) X_TECH(1:T-1,i)],0);
    results_TECH(i,[1 2 4])=[results_i.beta(2) results_i.tstat(2) ...
        results_i.rsqr];
    disp(i);
end;
results_TECH(N_TECH+1:end,1:2)=[results_PC_TECH.beta(2:end) ...
    results_PC_TECH.tstat(2:end)];
results_TECH(N_TECH+1,4)=results_PC_TECH.rsqr;

% Principal commponent predictive regression: all predictors

disp('Computing PC predictive regression: all predictors');
[Lambda_hat_ALL, F_hat_ALL]=princomp(zscore([X_ECON X_TECH]));
[k_ALL]=Perform_selection_IC(y(2:end),...
    F_hat_ALL(1:end-1,1:k_max_ALL),IC);
F_hat_ALL=F_hat_ALL(:,1:k_ALL);
results_PC_ALL=nwest(y(2:T),[ones(T-1,1) F_hat_ALL(1:T-1,:)],0);
results_ALL=nan(k_ALL,4);
results_ALL(:,1:2)=[results_PC_ALL.beta(2:end) ...
    results_PC_ALL.tstat(2:end)];
results_ALL(1,4)=results_PC_ALL.rsqr;

% Collecting expected equity risk premium estimates

FC_PC=[results_PC_ECON.yhat results_PC_TECH.yhat results_PC_ALL.yhat];

% Computing wild bootstrapped p-values

disp('Computing wild bootstrapped p-values');
tstat_ECON_star=nan(B,N_ECON+k_ECON);
tstat_TECH_star=nan(B,N_TECH+k_TECH);
tstat_ALL_star=nan(B,k_ALL);
for b=1:B;

    % Predictive regressions: macroeconomic variables

    for i=1:N_ECON;
        results_i_b=nwest(y_star(2:T,b),...
            [ones(T-1,1) X_ECON_star(1:T-1,i,b)],0);
        tstat_ECON_star(b,i)=results_i_b.tstat(2);
    end;
    [Lambda_hat_ECON_star_b,F_hat_ECON_star_b]=...
        princomp(zscore(X_ECON_star(:,:,b)));
    F_hat_ECON_star_b=F_hat_ECON_star_b(:,1:k_ECON);
    results_PC_ECON_star_b=nwest(y_star(2:T,b),...
        [ones(T-1,1) F_hat_ECON_star_b(1:T-1,:)],0);
    tstat_ECON_star(b,N_ECON+1:end)=results_PC_ECON_star_b.tstat(2:end)';

    % Predictive regressions: technical indicators

    for i=1:N_TECH;
        results_i_b=nwest(y_star(2:T,b),...
            [ones(T-1,1) X_TECH_star(1:T-1,i,b)],0);
        tstat_TECH_star(b,i)=results_i_b.tstat(2);
    end;
    [Lambda_hat_TECH_star_b,F_hat_TECH_star_b]=...
        princomp(zscore(X_TECH_star(:,:,b)));
    F_hat_TECH_star_b=F_hat_TECH_star_b(:,1:k_TECH);
    results_PC_TECH_star_b=nwest(y_star(2:T,b),...
        [ones(T-1,1) F_hat_TECH_star_b(1:T-1,:)],0);
    tstat_TECH_star(b,N_TECH+1:end)=results_PC_TECH_star_b.tstat(2:end)';

    % Principal component predictive regression: all predictors

    [Lambda_hat_ALL_star_b,F_hat_ALL_star_b]=...
        princomp(zscore([X_ECON_star(:,:,b) X_TECH_star(:,:,b)]));
    F_hat_ALL_star_b=F_hat_ALL_star_b(:,1:k_ALL);
    results_PC_ALL_star_b=nwest(y_star(2:T,b),...
        [ones(T-1,1) F_hat_ALL_star_b(1:T-1,:)],0);
    tstat_ALL_star(b,:)=results_PC_ALL_star_b.tstat(2:end)';
    disp(b);
end;
for i=1:N_ECON+k_ECON;
    results_ECON(i,3)=mean(tstat_ECON_star(:,i)>results_ECON(i,2));
    disp(i);
end;
for i=1:N_TECH+k_TECH;
    results_TECH(i,3)=mean(tstat_TECH_star(:,i)>results_TECH(i,2));
    disp(i);
end;
for i=1:k_ALL;
    results_ALL(i,3)=mean(tstat_ALL_star(:,i)>results_ALL(i,2));
    disp(i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Writing results to spreadsheet');

% Predictive regressions: macroeconomic variables

output_file='Returns_econ_tech_results';
output_sheet='In-sample estimates';
xlswrite(output_file,results_ECON(1:N_ECON,:),output_sheet,'b9');
xlswrite(output_file,results_ECON(N_ECON+1:end,1:3),output_sheet,'b26');
xlswrite(output_file,results_ECON(N_ECON+1,4),output_sheet,'e26');
output_sheet='In-sample factor loadings';
xlswrite(output_file,Lambda_hat_ECON(:,1:k_ECON),output_sheet,'b6');
output_sheet='In-sample principal components';
xlswrite(output_file,F_hat_ECON(:,1:k_ECON),output_sheet,'b289');

% Predictive regressions: technical indicators

output_sheet='In-sample estimates';
xlswrite(output_file,results_TECH(1:N_TECH,:),output_sheet,'b34');
xlswrite(output_file,results_TECH(N_ECON+1:end,1:3),output_sheet,'b51');
xlswrite(output_file,results_TECH(N_ECON+1,4),output_sheet,'e51');
output_sheet='In-sample factor loadings';
xlswrite(output_file,Lambda_hat_TECH(:,1:k_max_TECH),output_sheet,'f23');
output_sheet='In-sample principal components';
xlswrite(output_file,F_hat_TECH(:,1:k_TECH),output_sheet,'f289');

% Principal component predictive regression: all predictors

output_sheet='In-sample estimates';
xlswrite(output_file,results_ALL(:,1:3),output_sheet,'b59');
xlswrite(output_file,results_ALL(1,4),output_sheet,'e59');
output_sheet='In-sample factor loadings';
xlswrite(output_file,Lambda_hat_ALL(1:N_ECON,1:k_ALL),...
    output_sheet,'j6');
xlswrite(output_file,Lambda_hat_ALL(N_ECON+1:end,1:k_max_ALL),...
    output_sheet,'j23');
output_sheet='In-sample principal components';
xlswrite(output_file,F_hat_ALL(:,1:k_ALL),output_sheet,'j289');

% Expected equity risk premium estimates

output_sheet='In-sample forecasts';
xlswrite(output_file,FC_PC,output_sheet,'b290');
