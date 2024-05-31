function [sk, rd, T, flops] = randCPQR(A, r)
    [m, n] = size(A);

    if isgpuarray(A)
      display('randCPQR on GPU')
    else
      fprintf('randCPQR on CPU with %d threads\n', maxNumCompThreads)
    end


    tic
    if isgpuarray(A)
      G = sqrt(1/m)*randn(r,m,"gpuArray");
    else
      G = RandMat(r,m);
    end
    t0 = toc;

    tic
    Y = G*A;
    t1 = toc;
    
    tic
    [~, R, p] = qr(Y, 'econ', 'vector');
    t2 = toc;

    sk = p(1:r);
    rd = p(r+1:end);
    
    tic
    T  = R(1:r,1:r)\R(1:r,r+1:end);
    t3 = toc;
    
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
