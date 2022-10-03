%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform_reality_check.m
%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading equity premium/predictors, 1950:12-2011:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='Equity premium';
y=xlsread(input_file,input_sheet,'b289:b1021');
y=100*y; % percent equity premium
input_sheet='Macroeconomic variables';
X_ECON=xlsread(input_file,input_sheet,'b289:o1021');
input_sheet='Technical indicators';
X_TECH=xlsread(input_file,input_sheet,'b289:o1021');

% Adjusting economic variables where necessary for positive expected slope

X_ECON(:,7)=-X_ECON(:,7);
X_ECON(:,8)=-X_ECON(:,8);
X_ECON(:,9)=-X_ECON(:,9);
X_ECON(:,14)=-X_ECON(:,14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading actual/forecasts, 1966:01-2011:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_file='Estimate_predictive_regressions_out_of_sample';
load(input_file);
FC_PC_ECON=FC_ECON(:,end);
FC_ECON=FC_ECON(:,1:14);
FC_PC_TECH=FC_TECH(:,end);
FC_TECH=FC_TECH(:,1:14);
FC_PC_ALL=FC_ALL(:,end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clark and McCracken (2012) reality check for nested models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Computing maximum statistic

disp('Computing maxMSFE-F statistic');
P=size(actual,1);
e_HA=actual-FC_HA;
loss_HA=e_HA.^2;
FC_ALL=[FC_ECON FC_TECH FC_PC_ECON FC_PC_TECH FC_PC_ALL];
e_ALL=kron(actual,ones(1,size(FC_ALL,2)))-FC_ALL;
loss_ALL=e_ALL.^2;
d=kron(loss_HA,ones(1,size(loss_ALL,2)))-loss_ALL;
d_bar=mean(d)';
max_MSE_F=max(P*d_bar./mean(loss_ALL)');

% Computing OLS residuals for kitchen sink model (used for bootstrap)

T=size(y,1);
X_ECON_sink=X_ECON;
X_ECON_sink(:,[4 11])=[];
X_sink=[X_ECON_sink X_TECH];
results_sink=ols(y(2:T),[ones(T-1,1) X_sink(1:T-1,:)]);
e_sink=[0 ; results_sink.resid];

% Preliminaries

R=(1965-1950)*12+1; % 1950:12-1965:12 in-sample period
k_max=3;
k_max_ALL=4;
B=500;
mean_y=mean(y);
max_MSE_F_star=nan(B,1);
rng('default');
eta=randn(T,B);

% Wild fixed-regressor bootstrap

disp('Performing wild fixed-regressor bootstrap');
for b=1:B;
    y_star_b=mean_y*ones(T,1)+eta(:,b).*e_sink;
    FC_HA_star_b=nan(P,1);
    FC_ECON_star_b=nan(P,size(FC_ECON,2));
    FC_TECH_star_b=nan(P,size(FC_TECH,2));
    FC_PC_star_b=nan(P,3);
    for t=1:P;
        y_star_b_t=y_star_b(1:R+(t-1));
        FC_HA_star_b(t)=mean(y_star_b_t);
        X_ECON_t=X_ECON(1:R+(t-1),:);
        X_TECH_t=X_TECH(1:R+(t-1),:);
        for i=1:size(FC_ECON,2);
            results_star_b_t_i=ols(y_star_b_t(2:end),...
                [ones(R+(t-2),1) X_ECON_t(1:end-1,i)]);
            FC_ECON_star_b(t,i)=[1 X_ECON_t(end,i)]*...
                results_star_b_t_i.beta;
        end;
        for i=1:size(FC_TECH,2);
            results_star_b_t_i=ols(y_star_b_t(2:end),...
                [ones(R+(t-2),1) X_TECH_t(1:end-1,i)]);
            FC_TECH_star_b(t,i)=[1 X_TECH_t(end,i)]*...
                results_star_b_t_i.beta;
        end;
        [Lambda_ECON_t,PC_ECON_t]=princomp(zscore(X_ECON_t));
        R2bar_star_b_t=nan(k_max,1);
        for k=1:k_max;
            results_star_b_t_k=ols(y_star_b_t(2:end),...
                [ones(R+(t-2),1) PC_ECON_t(1:end-1,1:k)]);
            R2bar_star_b_t(k)=results_star_b_t_k.rbar;
        end;
        [R2bar_ECON_star_min_b_t,k_ECON_star_b_t]=min(R2bar_star_b_t);
        results_star_b_t=ols(y_star_b_t(2:end),...
            [ones(R+(t-2),1) PC_ECON_t(1:end-1,1:k_ECON_star_b_t)]);
        FC_PC_star_b(t,1)=[1 PC_ECON_t(end,1:k_ECON_star_b_t)]*...
            results_star_b_t.beta;
        [Lambda_TECH_t,PC_TECH_t]=princomp(zscore(X_TECH_t));
        R2bar_star_b_t=nan(k_max,1);
        for k=1:k_max;
            results_star_b_t_k=ols(y_star_b_t(2:end),...
                [ones(R+(t-2),1) PC_TECH_t(1:end-1,1:k)]);
            R2bar_star_b_t(k)=results_star_b_t_k.rbar;
        end;
        [R2bar_TECH_star_min_b_t,k_TECH_star_b_t]=min(R2bar_star_b_t);
        results_star_b_t=ols(y_star_b_t(2:end),...
            [ones(R+(t-2),1) PC_TECH_t(1:end-1,1:k_TECH_star_b_t)]);
        FC_PC_star_b(t,2)=[1 PC_TECH_t(end,1:k_TECH_star_b_t)]*...
            results_star_b_t.beta;
        [Lambda_ALL_t,PC_ALL_t]=princomp(zscore([X_ECON_t X_TECH_t]));
        R2bar_star_b_t=nan(k_max_ALL,1);
        for k=1:k_max_ALL;
            results_star_b_t_k=ols(y_star_b_t(2:end),...
                [ones(R+(t-2),1) PC_ALL_t(1:end-1,1:k)]);
            R2bar_star_b_t(k)=results_star_b_t_k.rbar;
        end;
        [R2bar_ALL_star_min_b_t,k_ALL_star_b_t]=min(R2bar_star_b_t);
        results_star_b_t=ols(y_star_b_t(2:end),...
            [ones(R+(t-2),1) PC_ALL_t(1:end-1,1:k_ALL_star_b_t)]);
        FC_PC_star_b(t,3)=[1 PC_TECH_t(end,1:k_ALL_star_b_t)]*...
            results_star_b_t.beta;
    end;
    actual_star_b=y_star_b(R+1:end);
    e_HA_star_b=actual_star_b-FC_HA_star_b;
    loss_HA_star_b=e_HA_star_b.^2;
    FC_ALL_star_b=[FC_ECON_star_b FC_TECH_star_b FC_PC_star_b];
    e_ALL_star_b=kron(actual_star_b,ones(1,size(FC_ALL_star_b,2)))-...
        FC_ALL_star_b;
    loss_ALL_star_b=e_ALL_star_b.^2;
    d_star_b=kron(loss_HA_star_b,ones(1,size(loss_ALL_star_b,2)))-...
        loss_ALL_star_b;
    d_bar_star_b=mean(d_star_b)';
    max_MSE_F_star(b)=max(P*d_bar_star_b./mean(loss_ALL_star_b)');
    disp([b max_MSE_F_star(b)]);
end;
p_value=mean(max_MSE_F_star>max_MSE_F);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Writing results to spreadsheet');
output_file='Returns_econ_tech_results';
output_sheet='Out-of-sample results';
xlswrite(output_file,[max_MSE_F ; p_value],output_sheet,'b63');
