% filename: RandSketch.m
% Inputs: 
%   A: m x n input matrix
%   l: dimension for random embedding
%   embedtype: character string
%       'Gaussian': (default) random Gaussian matrix with iid N(0,1/b) entries
%       'SRTT': subsampled randomized trigonometric transform
%       'SparseSign': sparse sign matrices
%   num_power_iter: integer >= 0 for power method to compute row space approximator of A

% Outputs:
%   Y: l x n column sketch of A

function Y = RandColSketch(A, l, embedtype) 
    
    % set default values for embedtype, num_power_iter if not given

    [m,n] = size(A); 
    
    switch embedtype
        case 'Gaussian'
            Gamma = sqrt(1/l)*randn(l,n); % iid N(0, 1/b) entries

        case 'SRTT'
            tau = sqrt(n/l);
            D = diag(exp(2i*pi*rand(n,1)));
            F = zeros(n);
            for p = 1:n
                for q = 1:n
                    F(p,q) = sqrt(1/n)*exp(-2i*pi*(p-1)*(q-1)/n);
                end
            end
            R = eye(n); R = R(randperm(n,l),:);
            Gamma = tau*R*(F*D);

        case 'SRHT'
            tau = sqrt(n/l);
            D = diag(sign(-1+2*rand(n,1)));
            F = zeros(n);
            for p = 1:n
                for q = 1:n
                    F(p,q) = sqrt(1/n)*exp(-2i*pi*(p-1)*(q-1)/n);
                end
            end
            F = real(F) - imag(F);
            R = eye(n); R = R(randperm(n,l),:);
            Gamma = tau*R*(F*D);

        case 'SparseSign'
            Gamma = zeros(l,n);
            eta = min([8,l]); % sparsity parameter
            signvec = sign(-1+2*rand(eta,n));
            if eta < l
                for col = 1:n
                    r = randperm(l,eta);
                    Gamma(r,col) = signvec(:,col);
                end
            else
                Gamma = signvec;
            end
            Gamma = sqrt(n/eta)*Gamma;
    end
    
    Y = A*Gamma';

end