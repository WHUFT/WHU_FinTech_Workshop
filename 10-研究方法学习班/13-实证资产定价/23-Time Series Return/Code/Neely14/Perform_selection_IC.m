function [k_star]=Perform_selection_IC(y,F_hat,IC)

% Last modified: 09-03-2013

% Selects the number of principal components to include
% in a regression model based on an information criterion
% or the adjusted R-squared.
%
% Input
%
% y     = T-vector of dependent variable observations
% F_hat = T-by-K matrix of principal component observations
% IC    = information criterion
%           = 1 for AIC
%           = 2 for SIC
%           = 3 for adjusted R-squared
%
%
% Output
%
% k_star = selected number of factors

T=size(y,1);
K=size(F_hat,2);
IC_value=zeros(K,1);
results_K=ols(y,[ones(T,1) F_hat]);
p_K=K+1;
for k=1:K;
    X_k=[ones(T,1) F_hat(:,1:k)];
    results_k=ols(y,X_k);
    p_k=size(X_k,2);
    if IC==1;
        IC_value(k)=log(results_k.resid'*results_k.resid/T)+...
            2*p_k/T;
    elseif IC==2;
        IC_value(k)=log(results_k.resid'*results_k.resid/T)+...
            p_k*log(T)/T;
    elseif IC==3;
        IC_value(k)=-100*results_k.rbar;
    end;
end;
[IC_value_min,k_star]=min(IC_value);
