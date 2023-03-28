function [t1, t2, t3, t4, f1, f2, f3, f4] = benchmark_methods(A, b, tol)


m = size(A, 1);
X = rand(m);
Y = rand(m);

tic
Z = X*Y;
t4 = toc;
f4 = 2*m*m*m;

t = tic; 
[sk, ~, ~, f1] = RandAdapLUPP(A, b, tol); 
t1 = toc(t);

r = length(sk);

t = tic;
[~, ~, ~, f2] = randLUPP(A, r);
t2 = toc(t);

t = tic;
[~, ~, ~, f3] = randCPQR(A, r);
t3 = toc(t);


end
