% Threshold to consider a skill learned
epsilon = 0.01;

% Fundamental complexity (AKA max number of episodes to learn any skill)
c0 = 100;
% Trial episodes
n  = 0:0.01:c0; 

% Total number of skills
N_S = 2*64;

% Number of clusters
K = 4;

% Number of skills per cluster
N_Z = N_S/K;

% Basic learning rate
alpha = -log(epsilon)/c0;%0.05;

delta = -log(epsilon)/N_Z;%0.05;

eta   = 0.1;

f = @(eta, N_zeta) eta*N_zeta+1;
g = @(delta, N_zeta) exp(-delta*N_zeta);

%% number of robots available
m = 8;
skill_labels = cell(N_Z/m,1);
for i=1:N_Z/m
    skill_labels{i,1} = ['s_{',num2str(i),'}'];
end


%% ************************************************************************
% Dynamics of incremental learning 
% *************************************************************************

clc
close all
fig = figure('color','w');


kappa = zeros(1,N_Z+1);
c_jk  = zeros(N_Z, K);
for k = 1:K
    subplot(1,K,k)
    for N_zeta = 0:(N_Z/m)-1
        j        = N_zeta+1;
        bsigma_0 = g(delta, N_zeta);
        bsigma   = ode4(@(n,bsigma) -alpha*f(eta, N_zeta)*bsigma, n, bsigma_0);
        plot(n,bsigma,'LineWidth',2)
        hold on
        %kappa(N_zeta + 1)  = sum(bsigma);
        c_jk(j, k) = -(log(epsilon) + delta*N_zeta)/(alpha*f(eta, N_zeta));
    end
    
    p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
    title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(I)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlim([0 n(end)])
    ylim([-0.1 1.1])
    legend(skill_labels)
end
% legend(skill_labels)
% fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 18, 3, 1)
c_jk_par_il = m*c_jk;
%%

fig = gcf;           % generate a figure
tightfig(fig)
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_incremental_learning'));
% disp('Printing done!')
%% ************************************************************************
% Dynamics of transfer + incremental learning 
% *************************************************************************

close all
figure('color','w')

c_jk  = zeros(N_Z, K);
kappa = zeros(1,N_Z+1);
for k = 1:K
    subplot(1,K,k)
    beta_k = (k-1)/K;
    for N_zeta = 0:(N_Z/m)-1
        j = N_zeta+1;
%         bsigma = exp(-alpha*(N_zeta+1)*n)*bsigma_0;
    %     bsigma = ode4(@(n,bsigma) -alpha*N_zeta*bsigma,n,1);

        bsigma_0 = (1- beta_k)*g(delta, N_zeta);
        bsigma   = ode4(@(n,bsigma) -alpha*(1- beta_k)^(-1)*f(eta, N_zeta)*bsigma, n, bsigma_0);    
        plot(n,bsigma,'LineWidth',2)
        hold on
        kappa(N_zeta + 1)  = sum(bsigma);
        %c_jk(j, k) = -(log(epsilon) - log(bsigma_0))/(alpha*(N_zeta+1));
        c_jk(j, k) = -(log(epsilon*(1- beta_k)^(-1)) + delta*N_zeta)/(alpha*(1- beta_k)^(-1)*f(eta, N_zeta));
    end
    p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
    title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(IT)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlim([0 n(end)])
    ylim([-0.1 1.1])
    legend(skill_labels)
end
c_jk_par_itl = m*c_jk;
%%
fig = gcf;           % generate a figure
tightfig(fig)
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
% print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_incremental_transfer_learning'));
% disp('Printing done!')
%% **************************************************************************
% Dynamics of collective learning 
% *************************************************************************

clc
close all

% Coupling (adjacency) matrix
A_full= ones(m) - eye(m);
A     = A_full;

% Inter cluster scaling matrix
beta_k = 1/K;
B      = beta_k*A;
clc
for i=1:K
   row = (2*(i-1) + 1);
   B(row, row+1) = 1;
end
B = triu(B) + transpose(triu(B));

figure('color','w')
c_jk_cl  = zeros(N_Z, K);
for k = 1:(N_S/m)  
    disp(['Batch ',num2str(k),' -------------------------'])
    subplot(2,K,k)        
    N_zeta   = k-1;
    bsigma_0 = (1- beta_k)*g(delta, r*N_zeta).*ones(m,1);
    a        = -alpha*f(eta, r*N_zeta)*(1- beta_k)^(-1);
    gamma    = 0.01;%*(-(a)/max(eig(abs(A))));
    F        = a*eye(m) + gamma*A.*B
    lambda_F = eig(F);
    bsigma   = transpose(ode4(@(n,bsigma) F*bsigma, n, bsigma_0));
    disp(lambda_F)        
    plot(n,bsigma,'LineWidth',2)
    hold on
    p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
    title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlim([0 n(end)])
    ylim([-0.1 1.1])    
end













%%

cond = true;
while cond
    A_partial = rand(m);
    A_partial = A_partial<rand(1);
    A_partial = triu(A_partial,1) + transpose(triu(A_partial,1));
    cond = all(A_partial(:)==0);
end
%%
A_full    = ones(m) - eye(m);
A = A_full;
%%
close all
figure('color','w')
c_jk_cl  = zeros(N_Z, K);
kappa = zeros(1,N_Z+1);
for k = 1:K
    disp(['Cluster ',num2str(k),' -------------------------'])
    subplot(1,K,k)
    beta_k = (k-1)/K;
    if k == 1
        r  = 1;
    else
        r = m;
    end    
    for N_zeta = 0:m:N_Z-m    
        j        = N_zeta+1;
        bsigma_0 = (1- beta_k)*g(delta, N_zeta).*ones(m,1);
        %B = TBD; (1- beta_k)^(-1)
%         a     = alpha*r*f(eta, N_zeta)*(1- beta_k)^(-1);
%         gamma = 1.1*(abs(a)/max(eig(abs(A))));
%         test = [a, gamma, abs(a)/max(eig(abs(A))), gamma > abs(a)/max(eig(abs(A)))]
%         F      = -a*eye(m) - gamma*A;

        a     = -alpha*r*f(eta, N_zeta)*(1- beta_k)^(-1);
        gamma = -0.3*(-(a)/max(eig(abs(A))));
%         test  = [a, gamma, -(a)/max(eig(abs(A))), gamma < -(a)/max(eig(abs(A)))]
        F     = a*eye(m) + gamma*A;
        lambda_F = eig(F)
%         if any(real(lambda_F)>0)
%             warning('Positive eigenvalue')
%             return
%         end
        bsigma = transpose(ode4(@(n,bsigma) F*bsigma, n, bsigma_0));
        

%         for i=0:m-1
%             c_jk_cl(j+i, k) = n(min(find(bsigma(i+1,:)<epsilon))); 
%         end
        plot(n,bsigma,'LineWidth',2)
        hold on
%         kappa(N_zeta + 1)  = sum(bsigma);
    end
    p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
    plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
    title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlabel('n','FontSize',25)
    ylabel(['$\bar{\boldmath{\sigma}}^{(C)}_{j,',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
    xlim([0 n(end)])
    ylim([-0.1 1.1])
end

%%
fig = gcf;           % generate a figure
tightfig(fig)
fig.Units = 'centimeters';        % set figure units to cm
% f.Position = [1203 646 478 174];
fig.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
fig.PaperSize = fig.Position(3:4);  % assign to the pdf printing paper the size of the figure
% print(fig,'-dpdf','-r600','-painters',fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'figures','dynamics_collective_learning'));
% disp('Printing done!')
%%
for k = 1:K
    A = G*A_i;
    disp('A:');
    disp(A)
    disp('Eigenvalues:')
    disp([(eig(A)),(eig(A_i)),abs(eig(A))-abs(eig(A_i))]) 
    
    
    disp('Det:')    
    disp(det(A))
%     close all
    learning_type_1 = 'collective_sat';
    % (t, x, A, alpha, s, c0, rate, threshold, type)
    if cycle ==1
        var = 1;
      
    else
        var = (cycle-1)*m;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha, s, c0, tau, epsilon, learning_type_1), ...
                                             n, ...
                                             1-(cycle-1)/(N/m)*ones(m,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(n,x,'LineWidth',2);
    hold on
end
plot(n,exp(A_i(1,1)*n),'k--','LineWidth',2)
p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 n(end)])
ylim([-0.1 1.1])
% set(gca, 'YScale', 'log')


%% **************************************************************************
% Dynamics of colletive learning 
% *************************************************************************
clc
close all
clear A

% This variable determines when knowlege collection is completed
epsilon = 0.01;

% Number of skills
N = 10;

% Number of robots
m = 2;

% Skill index
s         = transpose(1:N);

% Skill vector
Nj        = 0:N;

% Rate at which knowledge is depleted per skill acquired
% *NOTE: determined to satisfy <threshold = exp(-alpha*N)>
alpha     = -log(epsilon)/N;  % log(0.01)/N;%(rate/N);

% Initial values of the subsequent skills based on alpha
bsigma_0  = exp(-alpha*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:c0*N;
n  = 0:0.01:100; 

% Initial complexity
c0        = 100;

% Initial knowledge aquisition rate per episode
% *NOTE: determined to satisfy <threshold = exp(-(rate/c0)*c0)>
tau       = -log(epsilon)*ones(m,1);
A_i       = -diag(tau/c0);

close all
%theta          = 1:15;%&10*rand(N*(N+1)/2,1);
% theta          = normrnd(5*ones(N*(N+1)/2,1),0.33*ones(N*(N+1)/2,1));
theta          = 1*ones(m*(m+1)/2,1);
% theta(N+1:end) = rand(N*(N+1)/2 - N,1);
gamma           = 1;%rand(1);
theta(m+1:end) = gamma*ones(m*(m+1)/2 - m,1);
ind            = find(triu(ones(m,m),1));
L              = zeros(m) + diag(theta(1:m));
L(ind)         = theta(m+1:end);
G              = L + triu(L,1)';
eig(G);
% G = (L')*L


X = NaN(m,numel(n),N/m);
fig = figure('Color','w');
fig.WindowState = 'maximized';
for cycle = 1:N/m
    A = G*A_i;
    disp('A:');
    disp(A)
    disp('Eigenvalues:')
    disp([(eig(A)),(eig(A_i)),abs(eig(A))-abs(eig(A_i))]) 
    
    
    disp('Det:')    
    disp(det(A))
%     close all
    learning_type_1 = 'collective_sat';
    % (t, x, A, alpha, s, c0, rate, threshold, type)
    if cycle ==1
        var = 1;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha, s, c0, tau, epsilon, learning_type_1), ...
                                             n, ...
                                             ones(m,1));        
    else
        var = (cycle-1)*m;
        x = ode4(@(t,x) ...
              fcn_general_dynamics(t, x, var*A, alpha, s, c0, tau, epsilon, learning_type_1), ...
                                             n, ...
                                             1-(cycle-1)/(N/m)*ones(m,1));        
    end

    
    x = x';
    
    X(:,:,cycle) =x;
    plot(n,x,'LineWidth',2);
    hold on
end
plot(n,exp(A_i(1,1)*n),'k--','LineWidth',2)
p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 n(end)])
ylim([-0.1 1.1])
% set(gca, 'YScale', 'log')

%%
% -------------------------------------------------------------------------
learning_type_2 = 'collective';
% (t, x, A, alpha, s, c0, rate, threshold, type)
x_nonebounded = ode4(@(t,x) ...
      fcn_general_dynamics(t, x, A, alpha, s, c0, tau, epsilon, learning_type_2), ...
                                     n, ...
...
ones(N,1));

figure('Color','w')
p = plot(n,x_nonebounded,':','LineWidth',1.5);
hold on
set(gca,'ColorOrderIndex',1)
p = plot(n,x,'LineWidth',2);
plot(n,exp(A_i(1,1)*n),'k--','LineWidth',2)
p = area(n,  epsilon*ones(numel(n),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
plot(n,epsilon*ones(numel(n),1),'k:','LineWidth',2);
title('Learning comparison','FontSize',25)
xlabel('Episodes [n]','FontSize',25)
ylabel('$\bar{\boldmath{\sigma}}(N)$','FontSize',25,'Interpreter','latex')
xlim([0 n(end)])
ylim([-0.1 1.1])
H = gcf;% or H=gca
H.WindowState = 'maximized';

% figure('Color','w')
%     p   = bar3(abs(A));
%     colorbar off
%     colormap((jet))
%     for k = 1:length(p)
%         zdata = p(k).ZData;
%         p(k).CData = zdata;
%         p(k).FaceColor = 'interp';
%     end
%     xlabel('X')
%     ylabel('Y')
%     
%     xlim([1 size(A,1)])
%     ylim([1 size(A,2)])   
%%
 
close all
s         = [1:N]';

Nj        = 0:N;
alpha     = -log(epsilon)/N;%-log(0.01)/N;%(rate/N);
% alpha     = (rate/N);
bsigma_0  = exp(-alpha*Nj);

% Some solutions may need a smaller time step for the ODE solver
% n  = 0:0.01:c0*N;
n  = 0:0.01:1000; 
learning_type = 'incremental_3';
x = ode4(@(t,x) fcn_general_dynamics(t, x, A, alpha, s, c0, tau, epsilon,...
                learning_type), ...
                                     n, ...
                                     zeros(N,1) +1*bsigma_0(1:end-1)');
x = x';
figure('Color','w')
if strcmp(learning_type,'incremental_2')
%     p = plot(n,bsigma_0(1:end-1)'-x,'LineWidth',2);
    p = plot(n,x,'LineWidth',2);
else
    p = plot(n, x,'LineWidth',2);
end
hold on

% p = plot(n,(1-bsigma_0(1:end-1))' - x,'LineWidth',2);
% p = plot(n, - x,'LineWidth',2);
% hold on
plot(n(end),bsigma_0(1:end-1),'*')
xlabel('Episodes [n]','FontSize',25)
ylabel('Knowledge ','FontSize',25)
% xlim([0 sum(c0*bsigma_0)])
ylim([0 1])
%
title('Incremental')

% clear indx
% for j = 1:size(x,1)
%     delta  = x(j,:) - bsigma_0(j);
%     indx(j) = find(abs(delta) < threshold,1,'first');
% %     disp(n(indx(j)))
% end
% figure('Color','w')
% plot(1:N,c0*bsigma_0(1:end-1),'--','LineWidth',2)
% hold on
% plot(1:N,[n(indx(1)), diff(n(indx(:)))],'LineWidth',2)
% xlabel('Skills','FontSize',25)
% ylabel('Complexity ','FontSize',25)
% legend('Ideal','Actual','FontSize',10)
% xlim([1 N])


%%

sig_bar = @(Nj,alpha) exp(-alpha*Nj);
episodes = 0:c0;
clear indx
clc
close all
figure('Color','w')
% plot(Nj,sig_bar(Nj,(5/N)),'LineWidth',3)
% hold on
semilogy(episodes,0.01*ones(size(episodes)),'k:','LineWidth',3) % Threshold for skill completion
hold on
% The next two plots should be identical
% plot(Nj,0.3*sig_bar(Nj,(5/N)))
% stairs(Nj,sig_bar(Nj -log(0.3)/(5/N),(5/N)))
bsig = [];
for i=1:N
    bsig(i,:) = bsigma_0(i)*sig_bar(episodes,(tau/c0));
    indx(i) = find(bsig(i,:)<0.01,1,'first');
    disp(episodes(indx(i)))
    p = plot(episodes,bsigma_0(i)*sig_bar(episodes,(tau/c0)));
    semilogy(episodes(indx(i))*ones(numel(0:0.00001:1),1),0:0.00001:1,'--','Color',p.Color);
end

% Effects of changing the rate
% stairs(Nj,sig_bar(Nj,1.2*alpha))
xlabel('Episodes [n]','FontSize',25)

%%

sig_bar = @(Nj,alpha) exp(-alpha*Nj);
close all
semilogy(Nj,sig_bar(Nj,(5/N)))
hold on
plot(Nj,0.01*ones(size(Nj)),'k--')
semilogy(Nj,0.3*sig_bar(Nj,(5/N)))
semilogy(Nj,sig_bar(Nj -log(0.3)/(5/N),(5/N)))
semilogy(Nj,sig_bar(Nj,0.3))
%%
clc
close all

x = ode4(@(t,x) -A(1,1)*x,n,1);
figure
stairs(n,x)
xlabel('Episodes [n]','FontSize',25)

%% ************************************************************************

clearvars
close all
clc
clc
c0      = 100;
alpha_i = 10;
N_S     = 6;
N_K     = 2;
N       = N_S/N_K;
Spp     = sympositivedefinitefactory(N);

A     = Spp.rand();


% A     = A + abs(min(A,[],'all'));
% A     = A./(sum(A,'all'))

% Alternative construction for A
A = rand(N); % generate a random n x n matrix
A = 0.5*(A+A');
A = A.*(~eye(N)) + N*eye(N);


A
[V,D] = eig(-A)
c     = (V\ones(N,1));
%
close all
n = 0:0.001:c0;
% n=0;
f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
x = f(n);

f_i = @(t,x) -A.*eye(N)*x; 
f_c = @(t,x) -A*x; 
% x = ode4(f,n,sum(A,2));
x_i = ode4(f_i,n,ones(N,1));
x_c  = ode4(f_c,n,ones(N,1));


min(x_c(:))
figure('Color',[1 1 1])
plot(n,x_i,'k','LineWidth',1.5)
hold on
% plot(n,mean(x,2),'k--')
plot(n,x_c)
xlabel('Episodes n','FontSize',25)
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
xlim([0 5])
%%
close all
 plot(n,exp(-3*n) + exp(-8*n))
 plot(n,0.7*exp(-3*n))
 hold on
 plot(n,0.3*exp(-8*n))
 plot(n,0.7*exp(-3*n) + 0.3*exp(-8*n),'k--')
 xlim([0 10])
%% ************************************************************************
% SIMULATION OF THE COMPLETE SYSTEM
% *************************************************************************

clearvars
close all
clc
clc
c0      = 100;
alpha_i = 10;
N_S     = 4;
N_K     = 2;
N       = N_S/N_K;

Spp   = sympositivedefinitefactory(N);
A1     = Spp.rand();
A1     = A1 + abs(min(A1,[],'all'));

Spp   = sympositivedefinitefactory(N);
A2     = Spp.rand();
A2     = A2 + abs(min(A2,[],'all'));

B = 0.1/3*ones(N);

A = [A1, B; B, A2]

[V,D] = eig(-A/c0)
% c     = (V\ones(N,1));
%
close all
n = 0:0.001:c0;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
% f = @(t,x) -eye(size(A))*x; 
% x = ode4(f,n,sum(A,2));
x = ode4(f,n,ones(size(A,1),1));

min(x(:))
figure('Color',[1 1 1])
plot(n,x(:,1:2),'b--')
hold on
plot(n,x(:,3:4),'r')
% plot(n,sum(x,2),'k:')
xlabel('Episodes n','FontSize',25)
xlim([0 5])
ylabel('$\bar{\sigma}^{(C)}_{j,k}$','Interpreter','latex','FontSize',25)

h = zeros(N_K, 1);
h(1) = plot(NaN,NaN,'--b');
h(2) = plot(NaN,NaN,'-r');
leg = legend(h, 'k_1','k_1');
set(leg,'FontSize',15)

%% ************************************************************************
% SIMULATION WITH PRE DEFINED EIGENVECTORS
% *************************************************************************

clc
clearvars

% V = [0.2 0.1; 0.6 0.1];
% W = [0.2 0.1;0.1 0.6];
N = 3;
Spp   = sympositivedefinitefactory(N);
W     = Spp.rand();
W     = W + abs(min(W,[],'all'));
W     = W.*(~eye(N)) + eye(N);
W     = W./sum(W,2)

% W = [0.7 0.3;0.3 0.7];
% D = diag([5 5]);
D = diag(rand(N,1));%5*eye(N);
A = W*D*(W^-1)
[V_A,D_A] = eig(A)
c  = (W\sum(W,2));
%
close all
n = 0:0.01:50;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
x = ode4(f,n,sum(W,2));



min(x(:))
plot(n,x)
hold on
plot(n,mean(x,2),'k--')

%% ************************************************************************
% SIMULATION OF THE DIFFUSION PROCESS ON A GRAPH
% *************************************************************************
clearvars
clc
close all

N          = 3;
Spp        = sympositivedefinitefactory(N);
A          = Spp.rand();
A          = A + abs(min(A,[],'all'));
A          = A.*(~eye(size(A)))
D          = diag(sum(A,2))
L          = D - A
[U,lambda] = eig(L)

close all
n = 0:0.001:100;
f = @(t,x) -L*x; 
% x = ode4(f,n,sum(A,2));
% x = ode4(f,n,ones(N,1));
x0 = rand(N,1);
x0 = N*(x0/sum(x0));
x = ode4(f,n,x0);
plot(n,x)
xlabel('Episodes [n]','FontSize',25)
% xlim([0 10])
ylabel('$\bar{\sigma}^{(C)}_\mathcal{S}$','Interpreter','latex','FontSize',25)
