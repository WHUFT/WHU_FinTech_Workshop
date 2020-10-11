%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate_predictive_regressions_in_sample_dSENT_cycle.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data/defining variables, 1965:07-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='Sentiment';
y=xlsread(input_file,input_sheet,'e465:e1009'); % 1965:08-2010:12
y=[-999 ; y];
input_sheet='Equity premium';
recession=xlsread(input_file,input_sheet,'d465:d1009');
ind_exp=find(recession==0);
ind_rec=find(recession==1);
input_sheet='Macroeconomic variables';
X_ECON=xlsread(input_file,input_sheet,'b464:o1009');
input_sheet='Technical indicators';
X_TECH=xlsread(input_file,input_sheet,'b464:o1009');

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
k_ECON=3;
N_TECH=size(X_TECH,2);
k_TECH=1;
k_ALL=4;

% Macroeconomic variables

disp('Macroeconomic variables');
R2_ECON=nan(N_ECON+1,2);
for i=1:N_ECON;
    results_i=ols(y(2:T),[ones(T-1,1) X_ECON(1:T-1,i)]);
    ydev2_exp_i=(results_i.y(ind_exp)-mean(results_i.y)).^2;
    ydev2_rec_i=(results_i.y(ind_rec)-mean(results_i.y)).^2;
    e2_exp_i=results_i.resid(ind_exp).^2;
    e2_rec_i=results_i.resid(ind_rec).^2;
    R2_exp_i=1-(sum(e2_exp_i)/sum(ydev2_exp_i));
    R2_rec_i=1-(sum(e2_rec_i)/sum(ydev2_rec_i));
    R2_ECON(i,:)=[R2_exp_i R2_rec_i];
    disp([i R2_ECON(i,:)]);
end;
[Lambda_hat_ECON,F_hat_ECON]=princomp(zscore(X_ECON));
results=ols(y(2:T),[ones(T-1,1) F_hat_ECON(1:T-1,1:k_ECON)]);
ydev2_exp=(results.y(ind_exp)-mean(results.y)).^2;
ydev2_rec=(results.y(ind_rec)-mean(results.y)).^2;
e2_exp=results.resid(ind_exp).^2;
e2_rec=results.resid(ind_rec).^2;
R2_exp=1-(sum(e2_exp)/sum(ydev2_exp));
R2_rec=1-(sum(e2_rec)/sum(ydev2_rec));
R2_ECON(end,:)=[R2_exp R2_rec];
disp([N_ECON+1 R2_ECON(end,:)]);

% Technical indicators

disp('Technical indicators');
R2_TECH=nan(N_TECH+1,2);
for i=1:N_TECH;
    results_i=ols(y(2:T),[ones(T-1,1) X_TECH(1:T-1,i)]);
    ydev2_exp_i=(results_i.y(ind_exp)-mean(results_i.y)).^2;
    ydev2_rec_i=(results_i.y(ind_rec)-mean(results_i.y)).^2;
    e2_exp_i=results_i.resid(ind_exp).^2;
    e2_rec_i=results_i.resid(ind_rec).^2;
    R2_exp_i=1-(sum(e2_exp_i)/sum(ydev2_exp_i));
    R2_rec_i=1-(sum(e2_rec_i)/sum(ydev2_rec_i));
    R2_TECH(i,:)=[R2_exp_i R2_rec_i];
    disp([i R2_TECH(i,:)]);
end;
[Lambda_hat_TECH,F_hat_TECH]=princomp(zscore(X_TECH));
results=ols(y(2:T),[ones(T-1,1) F_hat_TECH(1:T-1,1:k_TECH)]);
ydev2_exp=(results.y(ind_exp)-mean(results.y)).^2;
ydev2_rec=(results.y(ind_rec)-mean(results.y)).^2;
e2_exp=results.resid(ind_exp).^2;
e2_rec=results.resid(ind_rec).^2;
R2_exp=1-(sum(e2_exp)/sum(ydev2_exp));
R2_rec=1-(sum(e2_rec)/sum(ydev2_rec));
R2_TECH(end,:)=[R2_exp R2_rec];
disp([N_TECH+1 R2_TECH(end,:)]);

% All predictors

disp('All predictors');
[Lambda_hat_ALL,F_hat_ALL]=princomp(zscore([X_ECON X_TECH]));
results=ols(y(2:T),[ones(T-1,1) F_hat_ALL(1:T-1,1:k_ALL)]);
ydev2_exp=(results.y(ind_exp)-mean(results.y)).^2;
ydev2_rec=(results.y(ind_rec)-mean(results.y)).^2;
e2_exp=results.resid(ind_exp).^2;
e2_rec=results.resid(ind_rec).^2;
R2_exp=1-(sum(e2_exp)/sum(ydev2_exp));
R2_rec=1-(sum(e2_rec)/sum(ydev2_rec));
R2_ALL=[R2_exp R2_rec];
disp(R2_ALL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Writing results to spreadsheet');
output_file='Returns_econ_tech_results';
output_sheet='In-sample estimates--sentiment';
xlswrite(output_file,R2_ECON(1:N_ECON,:),output_sheet,'f9');
xlswrite(output_file,R2_ECON(end,:),output_sheet,'f26');
xlswrite(output_file,R2_TECH(1:N_TECH,:),output_sheet,'f34');
xlswrite(output_file,R2_TECH(end,:),output_sheet,'f51');
xlswrite(output_file,R2_ALL,output_sheet,'f59');
