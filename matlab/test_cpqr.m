
n = 2000;

%A = randn(n) * diag(2.^(-1:-1:-n));
A = randn(n) * diag(2.^(-n:-1));


%%

[Q, R, p] = qr(A, 'vector');
tic
for i=1:5
    [Q, R, p] = qr(A, 'vector');
end
toc


[~, R, p] = qr(A, 'vector');
tic
for i=1:5
    [~, R, p] = qr(A, 'vector');
end
toc



%%

[Q, R] = qr(A);
tic
for i=1:5
    [Q, R] = qr(A);
end
toc


[~, R] = qr(A);
tic
for i=1:5
    [~, R] = qr(A);
end
toc


