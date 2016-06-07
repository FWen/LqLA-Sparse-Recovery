clear all;clc;%close all;

N = 512;
M = 200;
K = 30;

SNR = 30; %dB

A = randn(M,N);
A = orth(A')'; 

for l = 1:1
        
        t0 = tic;
        x = SparseVector(N, K);
        y0 = A*x;
        
        % --Two-component Gaussian mixture noise------
        epsilon = 0.1; kappa=1000;
        noise = gmrnd([0 0],[1 sqrt(kappa)],[1-epsilon epsilon],M,1);
        noise = noise/std(noise) *10^(-SNR/20)*std(y0);        
        y = y0 +  noise;

        lamda = 1;
        
        % run q=1 firstly to obtain an initialization for q<1
        [x_0] = lq_l1_admm(A, y, lamda, 1, x);
        
        % propsoed LqLA algorithm with smoothing, the {\bf v} subproblem is updated via (23)
        [x_rec1 Out1] = lq_l1_admm(A, y, lamda, 0.5, x, x_0);
        
        %  LqLA algorithm without smoothing, the {\bf v} subproblem is updated via (18)
        [x_rec2 Out2] = lq_l1_admm_nonsmoothing(A, y, lamda, 0.5, x, 1e3,x_0);

        figure(1);
        semilogy(1:length(Out2.e),Out2.e,'r--',1:length(Out1.e),Out1.e,'b--','linewidth',1); 
        xlabel('Iterations (k)'); grid on;ylim([1e-18 1]);
        legend('{\bf v} updated via (18)','{\bf v} updated via (23) with \epsilon=10^{-3}',1);
        ylabel('Changes in the {\bf x} update ||{\bf{x}}^{k+1}-{\bf{x}}^{k}||'); 

end


