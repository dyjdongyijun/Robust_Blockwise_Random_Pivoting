
function [sk, rd, T, flops] = RandAdapLUPP(A, blk, tol)

[m, n] = size(A);
X = zeros(m, n);
P = 1:m;

tic;
G = RandMat(n,blk);
t0 = toc;

tic
Y = A*G;
t1 = toc;
flops = 2*m*n*blk;

p = min(m, n);
nb = ceil(p/blk);
err = nan(nb,1);
err(1) = norm(Y, 'fro');

t2 = 0;
t4 = 0;
for i=0:nb-1
    k = i*blk;
    if i < nb-1
        b = blk;
    else
        b = p-(nb-1)*blk;
    end
    
    % inplace LU
    tic
    [L,~,phat] = lu( Y(k+1:end,1:b), 'vector' );
    t2 = t2 + toc;
    
    X(k+1:end,k+1:k+b) = L;
    flops = flops + 2*(m-k)*b*b/3;
    
    % global permutation
    tmp1 = P(k+1:end);
    P(k+1:end) = tmp1(phat);
    
    % apply local permutation
    if i>0
        tic
        tmp2 = k+1:m;
        X(k+1:end,1:k) = X( tmp2(phat), 1:k);
        t4 = t4 + toc;
    end
    
    if i==nb-1, break, end
    
    b = min(blk, p-(nb-1)*blk);
    
    tic
    G = RandMat(n,b);
    t0 = t0 + toc;
    
    tic
    Y = A*G;
    t1 = t1 + toc;
    flops = flops + 2*m*n*b;

    tic
    Y = Y(P,:);
    t4 = t4 + toc;
    
    % Schur complement
    tic
    k = k + blk;
    Y(k+1:end,:) = Y(k+1:end,:) - X(k+1:end,1:k) * (X(1:k,1:k) \ Y(1:k,:));
    t2 = t2 + toc;
    flops = flops + k*k*b + 2*(m-k)*k*b;
    
    
    %fprintf("Norm of Schur complement: %d\n", eSchur);
    err(i+2) = norm(Y(k+1:end,:), 'fro');
    if err(i+2) < tol, break, end
end

r = k;
sk = P(1:r);
rd = P(r+1:end);

tic
T = X(r+1:end,1:r)/X(1:r,1:r);
t3 = toc;

flops = flops + r*r*(m-r);

if 0   
    fprintf("\n------------------\n")
    fprintf("Profile of randAdapLUPP")
    fprintf("\n------------------\n")
    fprintf("Rand: %.3d\n", t0);
    fprintf("GEMM: %.3d\n", t1);
    fprintf("LUPP: %.3d\n", t2);
    fprintf("Perm: %.3d\n", t4);
    fprintf("Solve: %.3d\n", t3);
    fprintf("------------------\n")
    fprintf("Total: %.3d\n", t0+t1+t2+t3+t4);
    fprintf("------------------\n")
end
end