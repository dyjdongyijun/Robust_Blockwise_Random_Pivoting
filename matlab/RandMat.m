function R = RandMat(n, b)
    R = sqrt(1/b) * randn(n, b);
    %R = randn(n, b);
end