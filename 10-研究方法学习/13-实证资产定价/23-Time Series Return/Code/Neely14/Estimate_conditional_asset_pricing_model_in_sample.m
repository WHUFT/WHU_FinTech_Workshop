%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate_conditional_asset_pricing_model_in_sample.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data/defining variables, 1950:12-2011:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='In-sample principal components';
F_ECON=xlsread(input_file,input_sheet,'b289:d1021');
F_TECH=xlsread(input_file,input_sheet,'f289:f1021');
F_ALL=xlsread(input_file,input_sheet,'j289:m1021');
input_file='Returns_econ_tech_data';
input_sheet='Fama-French factors';
MKT=xlsread(input_file,input_sheet,'b289:b1021');
FF=xlsread(input_file,input_sheet,'b289:d1021');
UMD=xlsread(input_file,input_sheet,'e289:e1021');
R_f=xlsread(input_file,input_sheet,'f289:f1021');
input_sheet='10 momentum portfolios';
momentum=xlsread(input_file,input_sheet,'b289:k1021');
momentum=momentum-kron(ones(1,size(momentum,2)),R_f);
momentum=[momentum UMD];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conditional 3-factor model--time-varying alpha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Estimating conditional 3-factor model--time-varying alpha');
T=length(momentum);
X_TV=[ones(T-1,1) FF(2:end,:) F_ALL(1:end-1,:) ...
    F_ALL(1:end-1,:).*kron(ones(1,size(F_ALL,2)),FF(2:end,1)) ...
    F_ALL(1:end-1,:).*kron(ones(1,size(F_ALL,2)),FF(2:end,2)) ...
    F_ALL(1:end-1,:).*kron(ones(1,size(F_ALL,2)),FF(2:end,3))];
K=size(FF,2)+1;
N=size(momentum,2);
E=nan(T-1,N);
results_TV=nan(size(X_TV,2),2,N);
R2bar_TV=nan(N,1);
chi2_TV=nan(N,2,3);
R_1=zeros(K,size(X_TV,2));
R_1(1,5)=1;
R_1(2,9)=1;
R_1(3,13)=1;
R_1(4,17)=1;
R_2=zeros(3*K,size(X_TV,2));
R_2(1:3,6:8)=eye(3);
R_2(4:6,10:12)=eye(3);
R_2(7:9,14:16)=eye(3);
R_2(10:12,18:20)=eye(3);
R_3=[zeros(K*size(F_ALL,2),K) eye(K*size(F_ALL,2))];
for i=1:N;
    [beta_i,se_i,R2_i,R2bar_i,V_i,F_i]=olsWhite(momentum(2:end,i),X_TV);
    E(:,i)=momentum(2:end,i)-X_TV*beta_i;
    results_TV(:,:,i)=[beta_i beta_i./se_i];
    R2bar_TV(i)=R2bar_i;
    chi2_1_i=(R_1*beta_i)'*inv(R_1*V_i*R_1')*(R_1*beta_i);
    p_value_1_i=1-chis_cdf(chi2_1_i,size(R_1,1));
    chi2_TV(i,:,1)=[chi2_1_i p_value_1_i];
    chi2_2_i=(R_2*beta_i)'*inv(R_2*V_i*R_2')*(R_2*beta_i);
    p_value_2_i=1-chis_cdf(chi2_2_i,size(R_2,1));
    chi2_TV(i,:,2)=[chi2_2_i p_value_2_i];
    chi2_3_i=(R_3*beta_i)'*inv(R_3*V_i*R_3')*(R_3*beta_i);
    p_value_3_i=1-chis_cdf(chi2_3_i,size(R_3,1));
    chi2_TV(i,:,3)=[chi2_3_i p_value_3_i];
    disp(i);
end;
disp(results_TV);
disp(R2bar_TV);
disp(chi2_TV);

% Fixed-regressor wild bootstrap

X_0=[ones(T-1,1) FF(2:end,1:3)];
beta_0=nan(size(X_0,2),N);
for i=1:N;
    results_i=ols(momentum(2:end,i),X_0);
    beta_0(:,i)=results_i.beta;
end;
B=2000;
rng('default');
U=randn(T-1,B);
chi2_TV_bootstrap=nan(B,N,3);
for b=1:B;
    E_b=E.*kron(ones(1,N),U(:,b));
    r_b=X_0*beta_0+E_b;
    for i=1:N;
        [beta_b_i,se_b_i,R2_b_i,R2bar_b_i,V_b_i,F_b_i]=olsWhite(...
            r_b(:,i),X_TV);
        chi2_TV_bootstrap(b,i,1)=...
            (R_1*beta_b_i)'*inv(R_1*V_b_i*R_1')*(R_1*beta_b_i);
        chi2_TV_bootstrap(b,i,2)=...
            (R_2*beta_b_i)'*inv(R_2*V_b_i*R_2')*(R_2*beta_b_i);
        chi2_TV_bootstrap(b,i,3)=...
            (R_3*beta_b_i)'*inv(R_3*V_b_i*R_3')*(R_3*beta_b_i);
    end;
    disp(b);
end;
p_value_TV_bootstrap=nan(N,3);
for i=1:N;
    p_value_TV_bootstrap(i,1)=sum(chi2_TV_bootstrap(:,i,1)>...
        chi2_TV(i,1,1))/B;
    p_value_TV_bootstrap(i,2)=sum(chi2_TV_bootstrap(:,i,2)>...
        chi2_TV(i,1,2))/B;
    p_value_TV_bootstrap(i,3)=sum(chi2_TV_bootstrap(:,i,3)>...
        chi2_TV(i,1,3))/B;
end;

% Writing results to spreadsheet

disp('Writing results to spreadsheet');
output_file='Returns_econ_tech_results';
output_sheet='In-sample multifactor models';
beta_TV=nan(size(results_TV,3),size(results_TV,1));
tstat_TV=nan(size(results_TV,3),size(results_TV,1));
for i=1:size(results_TV,3);
    beta_TV(i,:)=results_TV(:,1,i)';
    tstat_TV(i,:)=results_TV(:,2,i)';
end;
xlswrite(output_file,beta_TV(:,1:4),output_sheet,'b8');
xlswrite(output_file,tstat_TV(:,1:4),output_sheet,'b22');
xlswrite(output_file,beta_TV(:,5:end),output_sheet,'f8');
xlswrite(output_file,tstat_TV(:,5:end),output_sheet,'f22');
xlswrite(output_file,R2bar_TV,output_sheet,'w8');
xlswrite(output_file,[chi2_TV(:,:,1) ...
    p_value_TV_bootstrap(:,1)],output_sheet,'y8');
xlswrite(output_file,[chi2_TV(:,:,2) ...
    p_value_TV_bootstrap(:,2)],output_sheet,'ac8');
xlswrite(output_file,[chi2_TV(:,:,3) ...
    p_value_TV_bootstrap(:,3)],output_sheet,'ag8');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conditional 3-factor model--constant alpha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Estimating conditional 3-factor model--constant alpha');
X_constant=[ones(T-1,1) FF(2:end,:) ...
    F_ALL(1:end-1,:).*kron(ones(1,size(F_ALL,2)),FF(2:end,1)) ...
    F_ALL(1:end-1,:).*kron(ones(1,size(F_ALL,2)),FF(2:end,2)) ...
    F_ALL(1:end-1,:).*kron(ones(1,size(F_ALL,2)),FF(2:end,3))];
K=size(FF,2);
N=size(momentum,2);
E=nan(T-1,N);
results_constant=nan(size(X_constant,2),2,N);
R2bar_constant=nan(N,1);
chi2_constant=nan(N,2,3);
R_1=zeros(K,size(X_constant,2));
R_1(1,5)=1;
R_1(2,9)=1;
R_1(3,13)=1;
R_2=zeros(3*K,size(X_constant,2));
R_2(1:3,6:8)=eye(3);
R_2(4:6,10:12)=eye(3);
R_2(7:9,14:16)=eye(3);
R_3=[zeros(K*size(F_ALL,2),K+1) eye(K*size(F_ALL,2))];
for i=1:N;
    [beta_i,se_i,R2_i,R2bar_i,V_i,F_i]=olsWhite(momentum(2:end,i),...
        X_constant);
    E(:,i)=momentum(2:end,i)-X_constant*beta_i;
    results_constant(:,:,i)=[beta_i beta_i./se_i];
    R2bar_constant(i)=R2bar_i;
    chi2_1_i=(R_1*beta_i)'*inv(R_1*V_i*R_1')*(R_1*beta_i);
    p_value_1_i=1-chis_cdf(chi2_1_i,size(R_1,1));
    chi2_constant(i,:,1)=[chi2_1_i p_value_1_i];
    chi2_2_i=(R_2*beta_i)'*inv(R_2*V_i*R_2')*(R_2*beta_i);
    p_value_2_i=1-chis_cdf(chi2_2_i,size(R_2,1));
    chi2_constant(i,:,2)=[chi2_2_i p_value_2_i];
    chi2_3_i=(R_3*beta_i)'*inv(R_3*V_i*R_3')*(R_3*beta_i);
    p_value_3_i=1-chis_cdf(chi2_3_i,size(R_3,1));
    chi2_constant(i,:,3)=[chi2_3_i p_value_3_i];
    disp(i);
end;
disp(results_constant);
disp(R2bar_constant);
disp(chi2_constant);

% Fixed-regressor wild bootstrap

chi2_constant_bootstrap=nan(B,N,3);
for b=1:B;
    E_b=E.*kron(ones(1,N),U(:,b));
    r_b=X_0*beta_0+E_b;
    for i=1:N;
        [beta_b_i,se_b_i,R2_b_i,R2bar_b_i,V_b_i,F_b_i]=olsWhite(...
            r_b(:,i),X_constant);
        chi2_constant_bootstrap(b,i,1)=...
            (R_1*beta_b_i)'*inv(R_1*V_b_i*R_1')*(R_1*beta_b_i);
        chi2_constant_bootstrap(b,i,2)=...
            (R_2*beta_b_i)'*inv(R_2*V_b_i*R_2')*(R_2*beta_b_i);
        chi2_constant_bootstrap(b,i,3)=...
            (R_3*beta_b_i)'*inv(R_3*V_b_i*R_3')*(R_3*beta_b_i);
    end;
    disp(b);
end;
p_value_constant_bootstrap=nan(N,3);
for i=1:N;
    p_value_constant_bootstrap(i,1)=sum(chi2_constant_bootstrap(:,i,1)>...
        chi2_constant(i,1,1))/B;
    p_value_constant_bootstrap(i,2)=sum(chi2_constant_bootstrap(:,i,2)>...
        chi2_constant(i,1,2))/B;
    p_value_constant_bootstrap(i,3)=sum(chi2_constant_bootstrap(:,i,3)>...
        chi2_constant(i,1,3))/B;
end;

% Writing results to spreadsheet

disp('Writing results to spreadsheet');
beta_constant=nan(size(results_constant,3),size(results_constant,1));
tstat_constant=nan(size(results_constant,3),size(results_constant,1));
for i=1:size(results_constant,3);
    beta_constant(i,:)=results_constant(:,1,i)';
    tstat_constant(i,:)=results_constant(:,2,i)';
end;
xlswrite(output_file,beta_constant(:,1:4),output_sheet,'b38');
xlswrite(output_file,tstat_constant(:,1:4),output_sheet,'b52');
xlswrite(output_file,beta_constant(:,5:end),output_sheet,'j38');
xlswrite(output_file,tstat_constant(:,5:end),output_sheet,'j52');
xlswrite(output_file,R2bar_constant,output_sheet,'w38');
xlswrite(output_file,[chi2_constant(:,:,1) ...
    p_value_constant_bootstrap(:,1)],output_sheet,'y38');
xlswrite(output_file,[chi2_constant(:,:,2) ...
    p_value_constant_bootstrap(:,2)],output_sheet,'ac38');
xlswrite(output_file,[chi2_constant(:,:,3) ...
    p_value_constant_bootstrap(:,3)],output_sheet,'ag38');
