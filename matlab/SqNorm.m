function [sk, rd, T] = SqNorm(A, r)

    if isgpuarray(A)
      display('SqNorm on GPU')
    else
      fprintf('SqNorm on CPU with %d threads\n', maxNumCompThreads)
    end

[m, n] = size(A);
diags = vecnorm(A).^2;

if isgpuarray(A)
    diags = gather(diags);
    r = gather(r);
end
J = datasample(1:n, r, 'Replace', false, 'Weights', diags);

sk = J;
[Q, R] = qr(A(:,sk), 'econ');

% redundance
p = 1:n;
p(J) = p(1:r);
rd = p(r+1:end);

% interpolation matrix
A(:, J) = A(:, 1:r);
T = R \ (Q' * A(:, r+1:end));


end
