
clear all

n = 2^12;
A = Matrix_Fast_Decay(n);

b = 128;
tol = 1e-8;


maxNumCompThreads(1);
[t1, t2, t3, t4, f1, f2, f3, f4] = benchmark_methods(A, b, tol);

fprintf("\n")
fprintf("------------------------------------------\n")
fprintf('# of threads: %i\n', maxNumCompThreads)
fprintf("------------------------------------------\n")
fprintf("\t\ttime(s)\t\tGflop/s\n")
fprintf("------------------------------------------\n")
fprintf("RandAdap\t%.2d\t%.2d\n", t1, f1/t1/1e9)
fprintf("RandLUPP\t%.2d\t%.2d\n", t2, f2/t2/1e9)
fprintf("RandCPQR\t%.2d\t%.2d\n", t3, f3/t3/1e9)
fprintf("GEMM\t\t%.2d\t%.2d\n", t4, f4/t4/1e9)
fprintf("------------------------------------------\n")


maxNumCompThreads('automatic');
[T1, T2, T3, T4, F1, F2, F3, F4] = benchmark_methods(A, b, tol);

fprintf("\n")
fprintf("----------------------------------------------------\n")
fprintf('# of threads: %i\n', maxNumCompThreads)
fprintf("----------------------------------------------------\n")
fprintf("\t\ttime(s)\t\tGflop/s\t\tSpeedup\n")
fprintf("----------------------------------------------------\n")
fprintf("RandAdap\t%.2d\t%.2d\t%.2d\n", T1, F1/T1/1e9, t1/T1)
fprintf("RandLUPP\t%.2d\t%.2d\t%.2d\n", T2, F2/T2/1e9, t2/T2)
fprintf("RandCPQR\t%.2d\t%.2d\t%.2d\n", T3, F3/T3/1e9, t3/T3)
fprintf("GEMM\t\t%.2d\t%.2d\t%.2d\n", T4, F4/T4/1e9, t4/T4)
fprintf("----------------------------------------------------\n")


