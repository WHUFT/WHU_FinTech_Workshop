 function [bv,sebv,R2v,R2vadj,v,F] = olsWhite(lhv,rhv) ;
% Copyright: Fuwei JIANG
% function ols does ols regressions  
% Inputs:
%  lhv T x N vector, left hand variable data 
%  rhv T x K matrix, right hand variable data
%  If N > 1, this runs N regressions of the left hand columns on all the (same) right hand variables. 
 %  NOTE: you must make one column of rhv a vector of ones if you want a constant. 
%       
% Output:
%  b: regression coefficients K x 1 vector of coefficients
%  seb: K x N matrix standard errors of parameters. 
%      (Note this will be negative if variance comes out negative) 
%  v: variance covariance matrix of estimated parameters. If there are many y variables, the vcv are stacked vertically
%  R2v:    unadjusted
%  R2vadj: adjusted R2
%  F: [Chi squared statistic    degrees of freedom    pvalue] for all coeffs jointly zero. 
%   Note: program checks whether first is a constant and ignores that one for test

global Exxprim;
global inner;

if size(rhv,1) ~= size(lhv,1);
   disp('olsgmm: left and right sides must have same number of rows. Current rows are');
   size(lhv)
   size(rhv)
end;

T = size(lhv,1);
N = size(lhv,2);
K = size(rhv,2);
sebv = zeros(K,N);
Exxprim = inv((rhv'*rhv)/T);
bv = rhv\lhv;

    errv = lhv-rhv*bv;
    s2 = mean(errv.^2);
    vary = lhv - ones(T,1)*mean(lhv);
    vary = mean(vary.^2);

    R2v = (1-s2./vary)';
    R2vadj= (1 - (s2./vary)*(T-1)/(T-K))';
    
    % Compute White standard errors
    for indx = 1:N;
        err=errv(:,indx);
    	inner = (rhv.*(err*ones(1,K)))'*(rhv.*(err*ones(1,K)))/T;         
        varb = 1/T*Exxprim*inner*Exxprim;
        
    % F test for all coeffs (except constant) zero -- actually chi2 test
        if rhv(:,1) == ones(size(rhv,1),1); 
            chi2val = bv(2:end,indx)'*inv(varb(2:end,2:end))*bv(2:end,indx);
            dof = size(bv(2:end,1),1); 
            pval = 1-cdf('chi2',chi2val, dof); 
            F(indx,1:3) = [chi2val dof pval]; 
        else; 
            chi2val = bv(:,indx)'*inv(varb)*bv(:,indx);
            dof = size(bv(:,1),1); 
            pval = 1-cdf('chi2',chi2val, dof); 
            F(indx,1:3) = [chi2val dof pval]; 
        end; 
            
        if indx == 1; 
           v = varb;
        else;
           v = [v; varb ];
        end;
        
       seb = diag(varb);
       seb = sign(seb).*(abs(seb).^0.5);
       sebv(:,indx) = seb;
    end;
    
      
    