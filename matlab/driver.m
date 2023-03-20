
clear all
close all

%maxNumCompThreads(1);
fprintf('Number of threads: %i\n', maxNumCompThreads)


n = 2^11;
b = 128;
tol = 1e-8;

% n = 16;
% b = 4;
% tol = 1e-4;

% singular values decay fast
sv = 1e-16 .^ ((0:n-1)/(n-1));

[U,~] = qr(randn(n));
[V,~] = qr(randn(n));
A = U*diag(sv)*V';


tic 
[sk, rd, T, flops] = RandAdapLU(A, b, tol); 
t = toc;


fprintf("\nAdaptive LU\n") 
fprintf("time: %.2d s\n", t)
fprintf("flop/s: %.2d Gflop/s\n", flops/t/1e9)

e = norm(A(rd,:) - T*A(sk,:),'fro');
fprintf("rank: %i\n", length(sk))
fprintf("F error: %.2d\n", e)


%% Reference LUPP (known rank)

t = tic;
r = sum(sv>tol);
[sk, rd, T, flops] = randLUPP(A, r);
t = toc(t);


fprintf("\nReference LU (known rank)\n")
fprintf("time: %.2d s\n", t)
fprintf("flop/s: %.2d Gflop/s\n", flops/t/1e9)

e = norm(A(rd,:) - T*A(sk,:),'fro');
fprintf("rank: %i\n", length(sk))
fprintf("F error: %.2d\n", e)


%% Reference CPQR (known rank)

t = tic;
r = sum(sv>tol);
[sk, rd, T, flops] = randCPQR(A, r);
t = toc(t);


fprintf("\nReference QR (known rank)\n")
fprintf("time: %.2d s\n", t)
fprintf("flop/s: %.2d Gflop/s\n", flops/t/1e9)

e = norm(A(rd,:) - T*A(sk,:), 'fro');
fprintf("rank: %i\n", length(sk))
fprintf("F error: %.2d\n", e)


%% Performance of GEMM

tic
m = n;
A = rand(m);
B = rand(m);
C = A*B;
t = toc;
flops = 2*m*m*m;

fprintf("\nReference Matmul\n")
fprintf("time: %.2d s\n", t)
fprintf("flop/s: %.2d Gflop/s\n", flops/t/1e9)


