
clear all

K = 5; 

% dataset
load mnist.mat
B = A(:, randperm(60000, 1000));
X = B ./ vecnorm(B);


[~, R1, p1] = qr(X,'econ','vector');

tic;
for i=1:K
  tic
  [~, R1, p1] = qr(X,'econ','vector');
  toc
end
t_cpqr = toc;

fprintf('CPQR time: %1.3e s\n', t_cpqr/K);

