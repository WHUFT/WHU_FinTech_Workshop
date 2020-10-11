%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate_peak_trough_regressions_in_sample.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%
% Loading data, 1951:01-2011:12
%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='Equity premium';
actual=xlsread(input_file,input_sheet,'b290:b1021');
actual=100*actual;
FC_HA=mean(actual)*ones(size(actual,1),1);
input_sheet='In-sample forecasts';
FC_ECON=xlsread(input_file,input_sheet,'b290:b1021');
FC_TECH=xlsread(input_file,input_sheet,'c290:c1021');
FC_ALL=xlsread(input_file,input_sheet,'d290:d1021');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining dummy variables for cyclical peaks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Creating dummy variable for cyclical peaks');
index_peak=[31 ; 80 ; 112 ; 228 ; 275 ; 349 ; 367 ; 475 ; 603 ; 684];
max_back=4;
max_forward=2;
D_peak=zeros(size(actual,1),max_back+max_forward+1);
for i_peak=1:size(index_peak,1);
    for i_lag=1:max_back+max_forward+1;
        D_peak(index_peak(i_peak)+i_lag-max_back-1,i_lag)=1;
    end;
end;

% Defining dummy variables for cyclical troughs

disp('Creating dummy variable for cyclical troughs');
index_trough=[41 ; 88 ; 122 ; 239 ; 291 ; 355 ; 383 ; 483 ; 611 ; 702];
D_trough=zeros(size(actual,1),max_back+max_forward+1);
max_back=4;
max_forward=2;
for i_trough=1:size(index_trough,1);
    for i_lag=1:max_back+max_forward+1;
        D_trough(index_trough(i_trough)+i_lag-max_back-1,i_lag)=1;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating regression models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Actual equity premium

disp('Estimating regression for actual equity premium');
T=size(actual,1);
results_actual=ols(actual,[D_peak D_trough ones(T,1)]);
disp([results_actual.beta results_actual.bstd]);

% Forecast based on macroeconomic variables

disp('Estimating regression for forecast based on macro variables');
results_ECON=ols(FC_ECON,[D_peak D_trough ones(T,1)]);
disp([results_ECON.beta results_ECON.bstd]);

% Forecast based on technical indicators

disp('Estimating regression for forecast based on technical indicators');
results_TECH=ols(FC_TECH,[D_peak D_trough ones(T,1)]);
disp([results_TECH.beta results_TECH.bstd]);

% Forecast based on all predictors

disp('Estimating regression for forecast based on all predictors');
results_ALL=ols(FC_ALL,[D_peak D_trough ones(T,1)]);
disp([results_ALL.beta results_ALL.bstd]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving results for figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%

save('Estimate_peak_trough_regressions_in_sample','results_actual',...
    'results_ECON','results_TECH','results_ALL');
