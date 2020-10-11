function [qLL]=Perform_EM_test(y,X,Z)

% Last modified: 09-03-2013

% Computes the Elliott and Muller (2006) qLL-hat statistic. The
% statistic is computed using the six steps given on page 914.
% Critical values are given in Table 1 on page 915.
%
% Input
%
% y = T-vector of dependent variable observations
% X = T-by-k data matrix for variables with potential breaks
% Z = T-by-k data matrix for variables with no breaks
%
% Output
%
% qLL = qLL-hat statistic
%
% Reference
%
% G. Elliott & U.K. Muller (2006), "Efficient Tests for General Persistent
% Time Variation in Regression Coefficients," Review of Economic Studies
% 73, 907-940

% Step 1

T=length(y);
k=size(X,2);
results=ols(y,[X Z]);

% Step 2

X_e=X.*kron(ones(1,k),results.resid);
V_X_hat=(1/T)*(X_e'*X_e);

% Step 3

U_hat=X_e*(V_X_hat^(-0.5));

% Step 4

r_bar=1-(10/T);
w_hat=nan(T,k);
for t=1:T;
    if t==1;
        w_hat(t,:)=U_hat(t,:);
   else
        w_hat(t,:)=r_bar*w_hat(t-1,:)+(U_hat(t,:)-U_hat(t-1,:));
    end;
end;

% Step 5

r_bar_t=(r_bar*ones(T,1)).^((1:1:T)');
SSR=nan(k,1);
SSU=nan(k,1);
for i=1:k;
    results_i=ols(w_hat(:,i),r_bar_t);
    SSR(i)=sum(results_i.resid.^2);
    SSU(i)=sum(U_hat(:,i).^2);
end;

% Step 6

qLL=r_bar*sum(SSR)-sum(SSU);
