function [MSPE_adjusted,p_value]=Perform_CW_test(actual,forecast_1,forecast_2)

% Last modified: 09-03-2013

% Performs the Clark and West (2007) test to compare forecasts
% from nested models.
%
% Input
%
% actual     = n-vector of actual values
% forecast_1 = n-vector of forecasts for restricted model
% forecast_2 = n-vector of forecasts for unrestricted model
%
% Output
%
% MSPE_adjusted = Clark and West (2007) statistic
% p_value       = corresponding p-value
%
% Reference
%
% T.E. Clark and K.D. West (2007). "Approximately Normal Tests
% for Equal Predictive Accuracy in Nested Models." Journal of
% Econometrics 138, 291-311

e_1=actual-forecast_1;
e_2=actual-forecast_2;
f_hat=e_1.^2-(e_2.^2-(forecast_1-forecast_2).^2);
Y_f=f_hat;
X_f=ones(size(f_hat,1),1);
beta_f=(inv(X_f'*X_f))*(X_f'*Y_f);
e_f=Y_f-X_f*beta_f;
sig2_e=(e_f'*e_f)/(size(Y_f,1)-1);
cov_beta_f=sig2_e*inv(X_f'*X_f);
MSPE_adjusted=beta_f/sqrt(cov_beta_f);
p_value=1-normcdf(MSPE_adjusted,0,1);
