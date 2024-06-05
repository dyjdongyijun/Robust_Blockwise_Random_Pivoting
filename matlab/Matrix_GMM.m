function A = Matrix_GMM(n, d)

k = 100;
m = n/k;

A = randn(n, d);

for i=1:k
    I = 1+(i-1)*m:i*m;
    A(I, i) = 10*i + A(I, i);
end

A = A';
end


