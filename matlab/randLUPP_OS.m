function [sk, rd, T, flops] = randLUPP_OS(A, r, c)
    
    if nargin < 3
        c = 2;
    end

    [m, n] = size(A);

    if isgpuarray(A)
      display('randLUPP-OS on GPU')
    else
      fprintf('randLUPP-OS on CPU with %d threads\n', maxNumCompThreads)
    end
    
    tic
    if isgpuarray(A)
      G = sqrt(1/m)*randn(c*r,m,"gpuArray");
    else
      G = RandMat(c*r,m);
    end    
    t0 = toc;
    
    tic
    Y = G*A;
    t1 = toc;
    
    tic
    [~,~,p] = lu(Y(1:r,:)', 'vector');
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
        fprintf("Profile of randLUPP-OS")
        fprintf("\n------------------\n")
        fprintf("Rand: %.3d\n", t0);
        fprintf("GEMM: %.3d\n", t1);
        fprintf("LUPP: %.3d\n", t2);
        fprintf("Solve: %.3d\n", t3);
        fprintf("------------------\n")
        fprintf("Total: %.3d\n", t0+t1+t2+t3);
        fprintf("------------------\n")
    end

end
