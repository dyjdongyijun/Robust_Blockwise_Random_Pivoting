function [sk, rd, T, flops] = randCPQR(A, r)
    [m, n] = size(A);
    
    tic
    G = RandMat(n,r);
    t0 = toc;
    
    tic
    Y = A*G;
    t1 = toc;
    
    Yt = Y';
    
    tic
    [~, R, p] = qr(Yt,0);
    t2 = toc;

    sk = p(1:r);
    rd = p(r+1:end);
    
    tic
    Tt  = R(1:r,1:r)\R(1:r,r+1:end);
    t3 = toc;
    
    T = Tt';
    flops = 2*m*n*r + 4*m*r*r/3 + r*r*(m-r);

    if 0    
        fprintf("\n------------------\n")
        fprintf("Profile of randCPQR")
        fprintf("\n------------------\n")
        fprintf("Rand: %.3d\n", t0);
        fprintf("GEMM: %.3d\n", t1);
        fprintf("CPQR: %.3d\n", t2);
        fprintf("Solve: %.3d\n", t3);
        fprintf("------------------\n")
        fprintf("Total: %.3d\n", t0+t1+t2+t3);
        fprintf("------------------\n")
    end
end