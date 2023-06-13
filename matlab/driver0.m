% Matrix Testing with Batched RandLUPP
% error estimates with Schur complement and residual U


%% Fast Decay

clear
close all
rng default

m = 1000;
n = 1000;
k = 200;

Sing_D = zeros(n,1);
beta = 1e-14;
for j = 1:n
    Sing_D(j) = beta^((j-1)/(n-1));
end

Sigma_D = diag(Sing_D);
Q_D = randn(m,n);
[Q_D,~] = qr(Q_D, 'econ');
V_D = randn(n,n);
[V_D,~] = qr(V_D,'econ');

D = Q_D*Sigma_D*V_D';
b = 20;
tau = 1e-8;

[I_s, W, U_crit_main, E_crit, L, U, Y_main, P] = BatchLUPP(D, b, tau);

l = length(E_crit);
for t = 1:l
    sv(t) = Sing_D(1+(t-1)*b);
end

for t = 2:l
    true_err(t-1) = norm(D(P,:)-L(:,1:(t-1)*b)*(L(1:(t-1)*b,1:(t-1)*b)\eye((t-1)*b,(t-1)*b))*D(P(1:(t-1)*b),:),"fro");
end

figure(1)
semilogy(0:b:length(I_s), E_crit, 'ob')
hold on
semilogy(0:b:length(I_s), U_crit_main, 'ok')
hold on
semilogy(0:b:length(I_s), sv, 'or')
hold on
semilogy(b:b:length(I_s),true_err,'*g')
legend('Schur','Res U','SVD','Actual')

%% S-shaped Decay

clear

m = 1000;
n = 1000;
k = 200;
s = 100;

beta = 1e-14;
Sing_D = zeros(n,1);
Sing_D(1:s) = sort(0.01*rand(s,1)+1,'descend');
for j = 1:n-s
    Sing_D(j+s) = beta^((j-1)/(n-s-1));
end
Sigma_D = diag(Sing_D);
Q_S = randn(m,n);
[Q_S,~] = qr(Q_S, 'econ');
V_S = randn(n,n);
[V_S,~] = qr(V_S,'econ');

D = Q_S*Sigma_D*V_S';

b = 20;
tau = 1e-8;

[I_s, W, U_crit_main, E_crit, L, U, Y_main, P] = BatchLUPP(D, b, tau);

l = length(E_crit);
for t = 1:l
    sv(t) = Sing_D(1+(t-1)*b);
end

for t = 2:l
    true_err(t-1) = norm(D(P,:)-L(:,1:(t-1)*b)*(L(1:(t-1)*b,1:(t-1)*b)\eye((t-1)*b,(t-1)*b))*D(P(1:(t-1)*b),:),"fro");
end

figure(2)
semilogy(0:b:length(I_s), E_crit, 'ob')
hold on
semilogy(0:b:length(I_s), U_crit_main, 'ok')
hold on
semilogy(0:b:length(I_s), sv, 'or')
hold on
semilogy(b:b:length(I_s),true_err,'*g')
legend('Schur','Res U','SVD','Actual')

%% Kahan
clear

m = 1000;
n = 1000;
k = 200;

zeta = 0.95;
phi = sqrt(1-zeta^2);
% Kahan construction
Sk = zeros(m,m);
Ku = -phi*ones(m,m);
for d = 1:m
    Sk(d,d) = zeta^(d-1);
end
K = eye(m,m) + triu(Ku,1);
D = Sk*K;
Sing_D = svd(D);

b = 20;
tau = 1e-8;

[I_s, W, U_crit_main, E_crit, L, U, Y_main, P] = BatchLUPP(D, b, tau);

l = length(E_crit);
for t = 1:l
    sv(t) = Sing_D(1+(t-1)*b);
end

for t = 2:l
    true_err(t-1) = norm(D(P,:)-L(:,1:(t-1)*b)*(L(1:(t-1)*b,1:(t-1)*b)\eye((t-1)*b,(t-1)*b))*D(P(1:(t-1)*b),:),"fro");
end

figure(3)
semilogy(0:b:length(I_s), E_crit, 'ob')
hold on
semilogy(0:b:length(I_s), U_crit_main, 'ok')
hold on
semilogy(0:b:length(I_s), sv, 'or')
hold on
semilogy(b:b:length(I_s),true_err,'*g')
legend('Schur','Res U','SVD','Actual')


