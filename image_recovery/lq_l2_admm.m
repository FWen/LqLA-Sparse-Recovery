function [x,Out] = lq_l2_admm(A, y, lamda, q, xtrue, x0)
% LqLS-admm solves
%
%   minimize || Ax - y ||_2^2 + \lambda || x ||_q^q
%
% Inputs
%	A,y,lambda: CS variables
%	0<=q<=1
%	xtrue: for debug, for calculation of errors
%   x0: intialization
% Outputs
%	x: the CS recovery
%	Out.e: the error with respect to the true
%	Out.et: time index

MAX_ITER = 1000; 
TOL = 1e-7;

CG_MAX_ITER = 10;
CG_TOL = 1e-5;

rho = 1;

if(isobject(A))
    m=A.m;
    n=A.n;
else
    [m,n]=size(A);
end

% save a matrix-vector multiply
Aty = A'*y;

if nargin<6
	x = zeros(n,1);
else
    x = x0;
end

z = x;
u = zeros(n,1);

Out.e  = [];
Out.et = [];
tic;

    
for iter = 1:MAX_ITER

    xm1 = x;
    zm1 = z;

    % x-update using the standard conjugate gradient method
    g_k  = zeros(n,1);
    d_k  = zeros(n,1);
    beta = 0;
    for iCG=1:CG_MAX_ITER
        g_km1 = g_k;
        d_km1 = d_k;
        g_k   = 2*(A'*(A*x) - Aty) + rho*(x-z+u); 
        if iCG>1
            beta   = norm(g_k)^2/norm(g_km1)^2;
        end
        d_k = beta*d_km1 - g_k;       
        x = x - g_k'*d_k / ( 2*norm(A*d_k)^2 + rho*d_k'*d_k ) * d_k;

        %terminate when the descent become samll
        if norm(g_k)<CG_TOL*sqrt(n)
            break;
        end
    end

    % z-update 
    z = shrinkage_Lq(x + u, q, lamda, rho);

    % u-update
    u = u + (x - z);

%     Out.e  = [Out.e norm(x-xtrue)/norm(xtrue)];
%     Out.e  = [Out.e norm(x-xm1)];
%     Out.et = [Out.et toc];

    %Check for convergence: when both primal and dual residuals become small
    if (norm(rho*(z-zm1))< sqrt(n)*TOL) && (norm(x-z) < sqrt(n)*TOL) 
        break;
    end
end
