% version from paper

function [I_s, W, U_crit_main, E_crit, L, U, Y_main, P] = BatchLUPP(A, b, tau)
m = size(A,1);

P = 1:m;
I_s = [];
I_r = 1:m;

W = [];
L1 = [];
L2 = [];
U1 = [];
U12 = [];
Y_main = [];

Y = RandColSketch(A,b,'Gaussian');
E_schur = Y;
E_crit = norm(E_schur,'fro');
U_crit = 1+tau;
U_crit_main = U_crit;

t = 1;

while E_crit(t) > tau && U_crit > tau
    l = (t-1)*b;
    Y_main = [Y_main, Y];
    [Lhat, Uhat, Phat] = lu(E_schur, "vector");

    L1hat = Lhat(1:b,:);
    L2hat = Lhat(b+1:end,:);

    Ptemp = P(l+1:end);
    P(l+1:end) = Ptemp(Phat);

    What = zeros(m-b*t,b);
    What(Phat,:) = [eye(b); L2hat*(L1hat\eye(size(L1hat)))];

    Wtild = zeros(m,b);
    Wtild(I_r,:) = What;

    W = [W, Wtild];

    Ihat_s = I_r(Phat(1:b));
    I_s = [I_s, Ihat_s];
    I_r = I_r(Phat(b+1:end));
    
    if t == 1
        L2temp = [];
    else
        L2temp = L2(Phat,:);
    end


    L = [L1, zeros(size(L1,1),b); L2temp, Lhat];
    U = [U1, U12; zeros(size(Uhat,1),size(U1,2)), Uhat];
    
    U1 = U;
    su = size(U1,1);
    L1 = L(1:su,:);
    L2= L(su+1:end,:);
    
    Y = RandColSketch(A,b,'Gaussian');
    Y_perm = Y(P,:);
    U12 = L1\eye(size(L1))*Y_perm(1:su,:);

    E_schur = Y_perm(su+1:end,:) - L2*U12;
    E_crit = [E_crit, norm(E_schur,'fro')];
    U_crit = norm_Uest(U,b*(t-1));
    U_crit_main = [U_crit_main, U_crit];

    t = t+1;

end

end


function est = norm_Uest(U,ind)
    rows = U(ind+1:end,:);
    M = max(sqrt(sum(rows.^2,2)));
    %m = min(sqrt(sum(rows.^2,2)));
    est = M;
end