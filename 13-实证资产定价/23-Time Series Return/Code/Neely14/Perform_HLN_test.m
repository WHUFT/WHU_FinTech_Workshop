function [lambda,MHLN,MHLN_pval]=Perform_HLN_test(actual,forecast_1,forecast_2)

% Last modified: 09-03-2013

% Performs the Harvey et al. (1998) test of whether forecast_1 encompasses
% forecast_2.
%
% Input:
%
% actual     = vector of actual values
% forecast_1 = vector of first forecasts
% forecast_2 = vector of second forecasts
%
% Output:
%
% lambda    = estimated combining weight on forecast_2
% MHLN      = modified HLN statistic
% MHLN_pval = one-sided (upper-tail) p-value for MHLN statistic
%
% Reference:
%
% D.I. Harvey, S.J. Leybourne, and P. Newbold (1998), "Tests for Forecast
% Encompassing," Journal of Economic and Business Statistics 16, 254-259

lambda=(forecast_2-forecast_1)\(actual-forecast_1);
u_1=actual-forecast_1;
u_2=actual-forecast_2;
d=(u_1-u_2).*u_1;
n=size(u_1,1);
d_bar=(1/n)*sum(d);
phi_0=(1/n)*(d-d_bar)'*(d-d_bar);
V_d_bar=(1/n)*phi_0;
HLN=((V_d_bar)^(-0.5))*d_bar;
HLN_pval=1-normcdf(HLN,0,1);
MHLN=((n-1)/n)*HLN;
MHLN_pval=1-tdis_cdf(MHLN,n-1);
