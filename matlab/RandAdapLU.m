
function [sk, rd, T, flops] = RandAdapLU(A, blk, tol)

[m, n] = size(A);
X = zeros(m, n);
P = 1:m;

Y = A*RandMat(n,blk);
flops = 2*m*n*blk;

p = min(m, n);
nb = ceil(p/blk);
err = nan(nb,1);
err(1) = norm(Y, 'fro');

for i=0:nb-1
    k = i*blk;
    if i < nb-1
        b = blk;
    else
        b = p-(nb-1)*blk;
    end
    
    % inplace LU
    [L,~,phat] = lu( Y(k+1:end,1:b), 'vector' );
    X(k+1:end,k+1:k+b) = L;
    flops = flops + 2*(m-k)*b*b/3;
    
    % global permutation
    tmp1 = P(k+1:end);
    P(k+1:end) = tmp1(phat);
    
    % apply local permutation
    if i>0
        tmp2 = X(k+1:end,1:k);
        X(k+1:end,1:k) = tmp2(phat,:);
    end
    
    if i==nb-1, break, end
    
    b = min(blk, p-(nb-1)*blk);
    Y = A*RandMat(n,b);
    Y = Y(P,:);
    flops = flops + 2*m*n*b;
    
    % Schur complement
    k = k + blk;
    Y(k:end,:) = Y(k:end,:) - X(k:end,1:k) * (X(1:k,1:k) \ Y(1:k,:));
    flops = flops + k*k*b + 2*(m-k)*k*b;
    
    err(i+2) = norm(Y(k:end,:), 'fro');
    %fprintf("Norm of Schur complement: %d\n", eSchur);
    if err(i+2) < tol, break, end
end

r = k;
sk = P(1:r);
rd = P(r+1:end);
T = X(r+1:end,1:r)/X(1:r,1:r);
flops = flops + r*r*(m-r);

end