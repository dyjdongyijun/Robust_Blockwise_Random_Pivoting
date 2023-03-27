function A = Matrix_Fast_Decay(n)

% singular values decay fast
sv = 1e-16 .^ ((0:n-1)/(n-1));


[U,~] = qr(randn(n));
[V,~] = qr(randn(n));
A = U*diag(sv)*V';

end