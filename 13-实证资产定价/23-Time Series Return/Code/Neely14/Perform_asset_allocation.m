function [avg_utility,SR,weight_risky,cumulative_return,avg_turnover]=...
    Perform_asset_allocation(actual,risk_free,forecast,volatility_FC,gamma_MV,c_bp)

% Last modified: 09-03-2013

% Perform asset allocation evaluation.
%
% Input
%
% actual        = T-vector of actual excess returns
% risk_free     = T-vector of risk-free rates
% forecast      = T-vector of excess return forecasts
% volatility_FC = T-vector of volatility forecasts
% gamma_MV      = risk aversion parameter
% c_bp          = transaction cost in basis points
%
% Output
%
% avg_utility       = average utility
% SR                = Sharpe ratio
% weight_risky      = T-vector of risky portfolio weights
% cumulative_return = T-vector of cumulative portfolio returns
% avg_turnover      = average turnover
%
% References
%
% J.Y. Campbell and S.B. Thompson (2008), "Predicting Excess
% Returns Out of Sample: Can Anything Beat the Historical Average?"
% Review of Financial Studis 21(4), 1510-1531
%
% V. DeMiguel, L Garlappi, F.J. Nogales, R. Uppal (2009),
% "A Generalized Approach to Portfolio Optimization: Improving
% Performance by Constraining Portfolio Norms," Management
% Science 55(5), 798-812
%
% V. DeMiguel, L. Garlappi, R. Uppal (2009), "Optimal Versus
% Naive Diversification: How Inefficient is the 1/N Portfolio
% Strategy?" Review of Financial Studies 22(5), 1915-1953

c=c_bp/10000;
T=size(actual,1);
weight_risky=zeros(T,1);
turnover=zeros(T,1);
return_portfolio=zeros(T-1,1);
for iter=1:T;
    weight_risky(iter)=(1/gamma_MV)*...
        (forecast(iter)/volatility_FC(iter));
    if weight_risky(iter)<0;
        weight_risky(iter)=0;
    elseif weight_risky(iter)>1.5;
        weight_risky(iter)=1.5;
    end;
end;
for t=1:T;
    if t<T;
        wealth_total_end_t=1+risk_free(t)+weight_risky(t)*actual(t);
        wealth_risky_end_t=weight_risky(t)*(1+risk_free(t)+actual(t));
        target_risky_end_t=weight_risky(t+1)*wealth_total_end_t;
        turnover(t)=abs(target_risky_end_t-wealth_risky_end_t)/wealth_total_end_t;
        TC_t=c*turnover(t);
        return_portfolio(t)=(1+risk_free(t)+weight_risky(t)*actual(t))*...
            (1-c*turnover(t))-1;
    else
        return_portfolio(t)=risk_free(t)+weight_risky(t)*actual(t);
    end;
end;
avg_utility=mean(return_portfolio)-0.5*gamma_MV*(std(return_portfolio))^2;
excess_return_portfolio=return_portfolio-risk_free;
SR=mean(excess_return_portfolio)/std(excess_return_portfolio);
cumulative_return=cumprod(ones(T,1)+return_portfolio);
avg_turnover=mean(turnover);
