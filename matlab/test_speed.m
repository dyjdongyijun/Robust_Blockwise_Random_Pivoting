
%clear all


%maxNumCompThreads(4);



%load mnist.mat
%B = A(:, randperm(60000,20000));


% 4000
%load cifar10.mat
%B = A(:, randperm(60000, 66));


% 4000
B = Matrix_Gaussian_exp(2000);
%B = Matrix_SNN(4000);
%B = Matrix_GMM(8000, 2000);


size(B)
B = gpuArray(B);

tic; 
[Q2, R2, p2] = qr(B,'econ','vector'); 
toc


tic;
[Q1, R1, p1] = RBRP_left(B, 64, 'random');
%[Q1, R1, p1] = RBRP(B, 128, 'random');
toc


tic; 
[Q3, R3] = qr(B,'econ'); 
toc


% C = B';
% tic;
% %[Q1, R1, p1] = RBRP(B, 64, 'greedy');
% [Q3, R3, p3] = RBRP_row(C, 128, 'random');
% toc


