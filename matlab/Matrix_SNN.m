function A = Matrix_SNN(n)

m = 100;
sv = nan(1,n);
sv(1:m) = 10 ./ (1:m);
sv(m+1:end) = 1 ./ (m+1:n);

U = sprand(n, n, 0.1);
V = sprand(n, n, 0.1);

A = U*diag(sv)*V';

end