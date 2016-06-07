clear all;clc;%close all;

N = 512;
M = 200;

A = randn(M,N);
A = orth(A')'; 

MC = 60;
Ks = [1 2 5:5:120];

for n=1: length(Ks)
    
    for l = 1:MC
        disp(['Sparsity: ', num2str(Ks(n)),'  Time: ', num2str(l)]);
        
        t0 = tic;
        x = SparseVector(N, Ks(n));
        y0 = A*x;
        
        % --Two-component Gaussian mixture noise------
%         SNR = 30; %dB
%         epsilon = 0.1; kappa=1000;
%         noise = gmrnd([0 0],[1 sqrt(kappa)],[1-epsilon epsilon],M,1);
%         noise = noise/std(noise) *10^(-SNR/20)*std(y0);     
        

        % --Gaussian noise-----------------------------
        SNR = 30; %dB
        noise = randn(M,1);
        noise = noise/std(noise) *10^(-SNR/20)*std(y0);  

        
        % --SaS noise-----------------------------------
%         noise = stblrnd(1, 0, 1e-4, 0, M, 1);
        

        y = y0 + noise;
        
        lamdas = logspace(log10(1e-5),log10(10),30);
       
        % YALL1
        for k = 1:length(lamdas)
            [x_rec1 Out1] = YALL1_admm(A, y, lamdas(k), x);
            relerr(1,k)   = norm(x_rec1 - x)/norm(x);
            xx(:,k)       = x_rec1;
        end
        [mv mi] = min(relerr(1,:));
        x_0 = xx(:,mi); 
        
        for k = 1:length(lamdas)
            % LqLA-ADMM (q=0.7)
            [x_rec1 Out2] = lq_l1_admm(A, y, lamdas(k), 0.7, x, x_0);
            relerr(2,k)   = norm(x_rec1 - x)/norm(x);            

            % LqLA-ADMM (q=0.5)
            [x_rec2 Out3] = lq_l1_admm(A, y, lamdas(k), 0.5, x, x_0);
            relerr(3,k)   = norm(x_rec2 - x)/norm(x);

            % LqLA-ADMM (q=0.2)
            [x_rec3 Out4] = lq_l1_admm(A, y, lamdas(k), 0.2, x, x_0);
            relerr(4,k)   = norm(x_rec3 - x)/norm(x);

            % L1LS-FISTA (q=1)
            [x_rec4 Out5] = lq_l2_fista(A, y, lamdas(k), 1, x);
            relerr(5,k)   = norm(x_rec4 - x)/norm(x);

            % LqLS-ADMM (q=0.5)
            [x_rec5 Out3] = lq_l2_admm(A, y, lamdas(k), 0.5, x, x_0);
            relerr(6,k)   = norm(x_rec5 - x)/norm(x);
            
        end
        RelErr(l,:) = min(relerr')';
       
        toc(t0)
    end

    aver_Err(n,:) = mean(RelErr);
    
    % successful rate of recovery
    for m=1:size(RelErr,2)
        i_Suc = find(RelErr(:,m)<=1e-2);
        Success_Rate(n,m) = length(i_Suc)/MC;
    end

end


figure(1);
plot(Ks,Success_Rate(:,5),'r-',Ks,Success_Rate(:,6),'r--',Ks,Success_Rate(:,1),'r:+',...
    Ks,Success_Rate(:,4),'b-',Ks,Success_Rate(:,3),'b--',Ks,Success_Rate(:,2),'b:+','linewidth',2);grid off;
legend('L1LS-FISTA','LqLS-ADMM (q=0.5)','YALL1','LqLA-ADMM (q=0.2)','LqLA-ADMM (q=0.5)','LqLA-ADMM (q=0.7)',1);
ylabel('Frequency of success'); xlabel('Sparsity K');
xlim([0 Ks(end)]);
