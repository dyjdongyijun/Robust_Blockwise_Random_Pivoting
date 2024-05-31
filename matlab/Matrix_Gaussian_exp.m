function A = Matrix_Gaussian_exp(n)

% singular values decay fast
%sv = 1e-16 .^ ((0:n-1)/(n-1));

m = 100;
sv = nan(1,n);
sv(1:m) = 1;
sv(m+1:end) = 0.8.^(1:n-m);
sv(sv<1e-5)=1e-5;

[U,~] = qr(randn(n));
[V,~] = qr(randn(n));
A = U*diag(sv)*V';

end