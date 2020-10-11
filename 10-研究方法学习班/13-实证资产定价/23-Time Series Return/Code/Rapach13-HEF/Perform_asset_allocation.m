function [avg_utility,weight_risky]=Perform_asset_allocation(actual,risk_free,forecast,volatility_forecast,gamma_MV)

% Last modified: 05-31-2012

% Performs asset allocation evaluation for allocation between risky
% asset and risk-free bill. The risky equity is restricted to lie
% between zero and 1.5.
%
% Input:
%
% actual              = T-vector of actual excess returns
% risk_free           = T-vector of risk-free rates
% forecast            = T-vector of excess return forecasts
% volatility_forecast = T-vector of volatility forecasts
% gamma_MV            = risk aversion parameter
%
% Output:
%
% avg_utility  = average utility
% weight_risky = T-vector of risky portfolio weights

T=size(actual,1);
weight_risky=zeros(T,1);
for iter=1:T;
    weight_risky(iter)=(1/gamma_MV)*(forecast(iter)/volatility_forecast(iter));
    if weight_risky(iter)<0;
        weight_risky(iter)=0;
    elseif weight_risky(iter)>1.5;
        weight_risky(iter)=1.5;
    end;
end;
return_portfolio=risk_free+weight_risky.*actual;
avg_utility=mean(return_portfolio)-0.5*gamma_MV*(std(return_portfolio))^2;
