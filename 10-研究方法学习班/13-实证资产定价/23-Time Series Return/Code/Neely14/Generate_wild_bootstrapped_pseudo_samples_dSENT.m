%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate_wild_bootstrapped_pseudo_samples_dSENT.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data/defining variables, 1965:07-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='Sentiment';
y=xlsread(input_file,input_sheet,'e465:e1009'); % 1968:08-2010:12
y=[-999 ; y];
input_sheet='Macroeconomic variables';
X_ECON=xlsread(input_file,input_sheet,'b464:o1009');
input_sheet='Technical indicators';
X_TECH=xlsread(input_file,input_sheet,'b464:o1009');

% Adjusting macroeconomic variables for positive expected slope

X_ECON(:,7)=-X_ECON(:,7); % net equity expansion
X_ECON(:,8)=-X_ECON(:,8); % Treasury bill yield
X_ECON(:,9)=-X_ECON(:,9); % long-term yield
X_ECON(:,14)=-X_ECON(:,14); % inflation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating wild bootstrapped pseudo samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preliminaries

disp('Wild bootstrap preliminaries');
B=2000;
T=size(y,1);
N_ECON=size(X_ECON,2);
N_TECH=size(X_TECH,2);
y_star=nan(T,B);
X_ECON_star=nan(T,N_ECON,B);
X_TECH_star=nan(T,N_TECH,B);
drop_index=[4 11];

% Estimating bias-corrected AR processes for economic variables

disp('Computing bias-adjusted AR parameters');
AR_coefficients=nan(N_ECON,2);
v_hat_c=nan(T-1,N_ECON);
for i=1:N_ECON;
    results_AR_i=ols(X_ECON(2:T,i),[ones(T-1,1) X_ECON(1:T-1,i)]);
    rho_hat_i=results_AR_i.beta(2);
    rho_hat_c_i=rho_hat_i+((1+3*rho_hat_i)/T)+(3*(1+3*rho_hat_i)/T^2);
    theta_hat_c_i=mean(X_ECON(2:T,i)-rho_hat_c_i*X_ECON(1:T-1,i));
    AR_coefficients(i,:)=[theta_hat_c_i rho_hat_c_i];
    v_hat_c(:,i)=X_ECON(2:T,i)-(theta_hat_c_i*ones(T-1,1)+...
        rho_hat_c_i*X_ECON(1:T-1,i));
end;

% Estimating persistence parameters for technical indicators

disp('Computing technical indicator persistence');
P=nan(N_TECH,2);
for i=1:N_TECH;
    S_i=X_TECH(:,i);
    S_t_minus_1_i=S_i(1:T-1);
    S_t_i=S_i(2:T);
    S_t_minus_1_0_i=sum(S_t_minus_1_i<1);
    S_t_minus_1_1_i=sum(S_t_minus_1_i>0);
    p_00_i=0;
    p_11_i=0;
    for t=1:T-1;
        if (S_t_minus_1_i(t)==0) && (S_t_i(t)==0);
            p_00_i=p_00_i+1;
        end;
        if (S_t_minus_1_i(t)==1) && (S_t_i(t)==1);
            p_11_i=p_11_i+1;
        end;
    end;
    p_00_i=p_00_i/S_t_minus_1_0_i;
    p_11_i=p_11_i/S_t_minus_1_1_i;
    P(i,:)=[p_00_i p_11_i];
end;

% Computing OLS return residuals for wild bootstrap

disp('Computing OLS residuals');
X_ECON_drop=X_ECON;
X_ECON_drop(:,drop_index)=[];
results_kitchen_sink=ols(y(2:T),...
    [ones(T-1,1) X_ECON_drop(1:T-1,:) X_TECH(1:T-1,:)]);
u_hat=results_kitchen_sink.resid;

% Generating wild bootstrapped draws

disp('Generating wild bootstrapped draws');
alpha_hat_restrict=mean(y(2:T));
rng('default');
w=randn(T-1,1,B);
e=randn(T,N_TECH,B);
for b=1:B;
    w_b=w(:,:,b);
    e_b=e(:,:,b);
    u_star_b=w_b.*u_hat;
    u_star_b=[0 ; u_star_b-mean(u_star_b)];
    v_star_b=kron(ones(1,N_ECON),w_b).*v_hat_c;
    v_star_b=[zeros(1,N_ECON) ; ...
        v_star_b-kron(mean(v_star_b),ones(T-1,1))];
    y_star_b=zeros(T,1);
    X_ECON_star_b=zeros(T,N_ECON);
    X_TECH_star_b=zeros(T,N_TECH);
    y_star_b(1)=-999;
    X_ECON_star_b(1,:)=X_ECON(1,:);
    X_TECH_star_b(1,:)=X_TECH(1,:);
    for t=2:T;
        y_star_b(t)=alpha_hat_restrict+u_star_b(t);
        X_ECON_star_b(t,:)=AR_coefficients(:,1)'+...
            AR_coefficients(:,2)'.*X_ECON_star_b(t-1,:)+v_star_b(t,:);
        for k=1:N_TECH;
            if X_TECH_star_b(t-1,k)==0;
                if e_b(t,k)<=P(k,1);
                    X_TECH_star_b(t,k)=0;
                else
                    X_TECH_star_b(t,k)=1;
                end;
            else
                if e_b(t,k)<=P(k,2);
                    X_TECH_star_b(t,k)=1;
                else
                    X_TECH_star_b(t,k)=0;
                end;
            end;
        end;
    end;
    y_star(:,b)=y_star_b;
    X_ECON_star(:,:,b)=X_ECON_star_b;
    X_TECH_star(:,:,b)=X_TECH_star_b;
    disp(b);
end;
output_file='Generate_wild_bootstrapped_pseudo_samples_dSENT';
save(output_file,'y_star','X_ECON_star','X_TECH_star');
