function [x,out] = lq_l1_admm(A,y,lamda,q,xtrue,x0);
% LqLA-ADMM solves
%
%   minimize || Ax - y ||_{1,ep} + \lamda || x ||_q^q
%
% Inputs:
%	A: sensing matrix
%	y: CS data
%	lamda: regularization parameter 
%	0<=q<=1
%	x0: initialization 
%	xtrue: for debug, for calculation of errors
% Outputs
%	x: the CS recovery
%	out.e: the error with respect to the true
%	out.et: time index


%Convergence setup
max_iter = 1e4;
ABSTOL = 1e-7;

ep = 1e-3;
tao1 = 0.99; %for orthonornal A
tao2 = 2*ep;

if(isobject(A))
    m = A.m;
    n = A.n;
else
    [m,n]=size(A);
end

rho_cov = 3.2/ep/lamda;
rho  = 1e2;

%Initialize
if nargin<6
	x  = zeros(n,1);
else
	x = x0;
end;

w = zeros(m,1); 
v = zeros(m,1);

out.e  = [];
out.et = [];
tic;

for i = 1 : max_iter
    
    rho = rho*1.01;
    
    xm1 = x;
    
    %x-step
    z = x - tao1*(A'*(A*x - y - v - w/rho)); 
    x = shrinkage_Lq(z, q, tao1, rho);
    
    %v-step
    d2 = v./sqrt(v.^2+ep*ep);
    v  = 1/(1/tao2+rho*lamda) * (1/tao2*v-d2+rho*lamda*(A*x-y-w/rho));
    
	%u-step
    Ax = A*x;
	w = w - rho*(Ax - y - v);

    out.e  = [out.e norm(x-xm1)];
    out.et = [out.et toc];
    
    %terminate when residuals are small
    if (norm((x-xm1))< sqrt(n)*ABSTOL && norm(Ax-y-v)< sqrt(n)*ABSTOL) 
%         break;
    end
end

end
