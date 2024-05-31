function [sk, rd, T, flops] = randCPQR_OS(A, r)
    [m, n] = size(A);
   
    if isgpuarray(A)
      display('randCPQR-OS on GPU')
    else
      fprintf('randCPQR-OS on CPU with %d threads\n', maxNumCompThreads)
    end

    tic
    if isgpuarray(A)
      G = sqrt(1/m)*randn(2*r,m,"gpuArray");
    else
      G = RandMat(2*r,m);
    end
    t0 = toc;

    tic
    Y = G*A;
    t1 = toc;
    
    tic
    [~, ~, p] = qr(Y(1:r, :), 'econ', 'vector');
    t2 = toc;

    sk = p(1:r);
    rd = p(r+1:end);
    
    % over-sampling for better accuracy
    tic
    Z = Y(:, p);
    [Q1, R1] = qr(Z(:, 1:r), 'econ');
    T = R1 \ (Q1' * Z(:, r+1:end));
    t3 = toc;


    if 1
        fprintf("\n------------------\n")
        fprintf("Profile of randCPQR-OS")
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
