close all
k_bar = [0:0.001:1];
plot(x,exp(-5*k_bar))

%%
clc
close all
r = [0.001:0.001:1];
% r = [1000:-0.1:1E-3];
s = 1./r;
k = 1 - exp(-0.1*s);
% k = 1- exp(-s);

figure
subplot(2,1,1)
plot(r,s,'r','LineWidth',2)
xlabel('r','FontSize',15)
ylabel('1/r','FontSize',15)
subplot(2,1,2)
plot(r,k,'b','LineWidth',2)
xlabel('r','FontSize',15)
ylabel('k(1/r)','FontSize',15)

%%
% rng('default')
clc
close all
N_tasks = 100;
tau     = 1:N_tasks;


r       = normrnd(0.1,10,N_tasks,1);
r(r<0)  = rand(1);

r       = S(end,1:end-1)';
N_tasks = numel(r);
tau     = 1:N_tasks;


% r       = sort(rand(N_tasks,1),'descend');
r       = rand(N_tasks,1);
% r       = 0.01*ones(N_tasks,1);
k       = 1 - exp(-0.1*(1./r));
% d       = exp(-0.01*cumsum(k)./cumsum(ones(N_tasks,1)));
% d       = exp(-0.05*cumsum(k));
% k       = 1./r;
% d       = exp(-N_tasks*cumsum(k)./cumsum(ones(N_tasks,1)));
% d       = exp(-1/N_tasks*cumsum(k));
% k_bar   = cumsum(k)./cumsum(ones(N_tasks,1)); 
k_bar   = cumsum(k);
d       = exp(-0.1*k_bar);
% d       = exp(-0.1*k_bar);
figure
    subplot(2,2,1)
    plot(tau,r,'-')
    xlabel('$\tau$','interpreter','latex','FontSize',15)
    ylabel('$r(\tau)$','interpreter','latex','FontSize',15)
    
    subplot(2,2,2)
    plot(tau,k,'-')
    xlabel('$\tau$','interpreter','latex','FontSize',15)
    ylabel('$k(\tau)$','interpreter','latex','FontSize',15)
    
    subplot(2,2,3)
    plot(tau,cumsum(k)./cumsum(ones(N_tasks,1)),'-')
    xlabel('$\tau$','interpreter','latex','FontSize',15)
    ylabel('$\bar{k}$','interpreter','latex','FontSize',15)    
    
    subplot(2,2,4)
    plot(tau,d,'-')
    xlabel('$\tau$','interpreter','latex','FontSize',15)
    ylabel('$d(\tau)$','interpreter','latex','FontSize',15)     


% plot(tau,d)
% plot(tau,d)
% plot(tau,1./x)
% hold on
% plot(x,cumsum(d))

%% Incremental learning of tasks in T
clear d
clc
close all

rng('default')
N_tasks = 60;
% T       = rand(N_tasks,2);
T       = normrnd(0,0.5,N_tasks,2);

c_0 = 100;
e_0 = 10; %[Joules]
P   = 100;

lambda  = zeros(N_tasks,1);
d       = zeros(N_tasks,1);
c_i     = zeros(N_tasks,1);
E_i     = zeros(N_tasks,1);
t_i     = zeros(N_tasks,1);
D       = squareform(pdist(T));
S       = D./max(D,[],'all');
for tau = 1:N_tasks
    if tau == 1
        d(tau) = exp(0);
    else
        s_inv       = 1./S(tau,1:tau-1);
        %lambda(tau) = 1/N_tasks*mean(s_inv);
        %d(tau)      = exp(-lambda(tau)*(tau-1));
        k           = 1 - exp(-0.1*s_inv);
        d(tau)      = exp(-1*sum(k));%exp(-5*sum(k)./(tau-1));        
        
    end
    if tau == 60
        test = 1;
    end
    c_i(tau)    = ceil(d(tau)*c_0);
    E_i(tau)    = c_i(tau)*e_0;
    t_i(tau)    = E_i(tau)/P;
end

figure(1)
subplot(3,2,[1,3,5])
plot(T(:,1),T(:,2),'ro','LineWidth',3)
subplot(3,2,2)
plot(1:N_tasks,d,'LineWidth',2)
xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
ylabel('$d(\tau_i,\mathcal{T}_l)$','FontSize',15,'interpreter','latex')
subplot(3,2,4)
plot(1:N_tasks,c_i,'LineWidth',2)
xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
ylabel('$c_i(\tau_i)$ [iterations]','FontSize',15,'interpreter','latex')
subplot(3,2,6)
plot(1:N_tasks,E_i,'LineWidth',2)
xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
ylabel('$E_{tot}(\tau_i)$ [J]','FontSize',15,'interpreter','latex')
        
offset = 0;
f_mat = [];
x_mat = [];
for tau = 1:N_tasks
    x      = 0:c_i(tau);
    zeta   = 1/(0.11*x(end));
    cnst   = zeta*E_i(tau)/(1 - exp(-zeta*x(end)));
    f      = cnst*exp(-zeta*x);
%     plot(x + offset,f)
%     [E_i(tau), trapz(f)]
    f_mat  = [f_mat, f];
    x_mat  = [x_mat, x + offset];
    offset = offset + x(end);    
end

figure(2)
    subplot(2,1,1)
    area(x_mat,f_mat)
    xlabel('k [Iterations]','FontSize',15,'interpreter','latex')
    ylabel('$E_i(k) [J]$','FontSize',15,'interpreter','latex')
    grid minor

    subplot(2,1,2)
    bar(1:N_tasks,cumsum(E_i))
    hold on
    plot(sum(E_i)*ones(N_tasks,1),'k--','LineWidth',2)
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$E_{tot}$ [J]','FontSize',15,'interpreter','latex')

%% ************************************************************************
% Collective learning

rng('default')
clear d
clc


close all



N_tasks = 60;
% T       = rand(N_tasks,2);
T       = normrnd(0,0.5,N_tasks,2);


c_0 = 100;
e_0 = 10; %[Joules]
P   = 100;


%%


N_robot = 3;

lambda  = zeros(N_tasks,1);
d       = zeros(N_robot, N_tasks);
c_i     = zeros(N_robot, N_tasks);
E_i     = zeros(N_robot, N_tasks);
t_i     = zeros(N_robot, N_tasks);


task_log = zeros(N_tasks,1);
tau_record = [];

complete = 0;
iter     = 0;
D           = squareform(pdist(T));
S           = D./max(D(:));

% =========================================================================
% while complete ~= 1
%     if sum(task_log) == 0
%         tau_choice  = reshape(randperm(N_tasks,N_robot),N_robot,1);
%     else
%         task_pending = find(task_log  == 0);
%         tau_choice   = task_pending(randperm(numel(task_pending),N_robot));
%     end
%     tau_record = [tau_record, tau_choice];
%     if sum(task_log(tau_choice)) == 0 
%         task_pool = find(task_log  == 1);
%         numel(task_pool)
%         for r = 1:N_robot
%                 if isempty(task_pool)
%                     d(r,tau_choice(r))  = exp(0);
%                 else
%                     if tau_choice(r) == 2
%                         test = 1;
%                     end
%                     s_inv               = 1./S(tau_choice(r),task_pool);
%                     lambda(tau_choice(r)) = 1/N_tasks*mean(s_inv);
% %                     d(r,tau_asig(r))      = exp(-lambda(tau_asig(r))*numel(task_pool));
%                     d(r,tau_choice(r))      = exp(-lambda(tau_choice(r))*numel(task_pool));                    
%                 end
% %                 c_i(r,tau_asig(r))    = ceil(d(r,tau_asig(r))*c_0);
% %                 E_i(r,tau_asig(r))    = c_i(tau)*e_0;
% %                 t_i(r,tau_asig(r))    = E_i(tau)/P;
%                 c_i(r,tau_choice(r))    = ceil(d(r,tau_choice(r))*c_0);
%                 E_i(r,tau_choice(r))    = c_i(r, tau_choice(r))*e_0;
%                 t_i(r,tau_choice(r))    = E_i(r, tau_choice(r))/P;
%         end
%         task_log(tau_choice) = 1;
%         if sum(task_log) == N_tasks
%             complete = 1;
%         end
%     else
%         continue
%     end
%     iter = iter +1;
% end
% =========================================================================
while complete ~= 1
    if sum(task_log) == 0
        tau_choice  = reshape(randperm(N_tasks,N_robot),N_robot,1);
    else
        task_pending = find(task_log == 0);
        tau_choice   = task_pending(randperm(numel(task_pending),N_robot));
    end
    tau_record = [tau_record, tau_choice];
    if sum(task_log(tau_choice)) == 0 
        task_pool = find(task_log  == 1);
        numel(task_pool)
        for r = 1:N_robot
                if isempty(task_pool)
                    tau_pool_ex           = tau_choice(tau_choice ~= tau_choice(r));
                    s_inv                 = 1./S(tau_choice(r),tau_pool_ex);
                    lambda(tau_choice(r)) = 1/N_tasks*mean(s_inv);
                    d(r,tau_choice(r))    = exp(-lambda(tau_choice(r))*numel(tau_pool_ex)); 
                    
k                  = 1 - exp(-0.1*s_inv);
d(r,tau_choice(r)) = exp(-1*sum(k));       
                    
                else
                    tau_pool_ex           = tau_choice(tau_choice ~= tau_choice(r));
                    s_inv                 = 1./S(tau_choice(r),[task_pool;tau_pool_ex]);
                    lambda(tau_choice(r)) = 1/N_tasks*mean(s_inv);
                    d(r,tau_choice(r))    = exp(-lambda(tau_choice(r))*numel([task_pool;tau_pool_ex]));

k                  = 1 - exp(-0.1*s_inv);
d(r,tau_choice(r)) = exp(-1*sum(k));                      
                    
                end
                c_i(r,tau_choice(r))    = ceil(d(r,tau_choice(r))*c_0);
                E_i(r,tau_choice(r))    = c_i(r, tau_choice(r))*e_0;
                t_i(r,tau_choice(r))    = E_i(r, tau_choice(r))/P;
        end
        task_log(tau_choice) = 1;
        if sum(task_log) == N_tasks
            complete = 1;
        end
    else
        continue
    end
    iter = iter +1;
end
% close all
% tickLabel = {size(tau_record,2)};
% for i=1:size(tau_record,2)
%     tickLabel{i} = ['T', num2str(tau_record(1,i))];
% end
% plot(d(1,tau_record(1,:)))
% xticklabels(tickLabel)
% xtickangle(45)

figure(3)
subplot(3,2,[1 3 5])
    plot(T(:,1),T(:,2),'ro','LineWidth',3)
    title('$\tau_i$','FontSize',15,'interpreter','latex')
subplot(3,2,2)
    hold on
    legLabel = {N_robot};
    for r = 1:N_robot
        plot(d(r,tau_record(r,:)),'-','LineWidth',2)
        legLabel{r} = ['r_', num2str(r)];
    end
    leg = legend(legLabel);
    leg.Orientation = 'horizontal';
    leg.Location = 'northeast';
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$d(\tau_i,\mathcal{T}_l)$','FontSize',15,'interpreter','latex')
subplot(3,2,4)
    plot(c_i(1,tau_record(1,:)),'r','LineWidth',2)
    hold on
    plot(c_i(2,tau_record(2,:)),'b','LineWidth',2)
    leg = legend(legLabel);
    leg.Orientation = 'horizontal';
    leg.Location = 'northeast';
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$c_i(\tau_i)$ [iterations]','FontSize',15,'interpreter','latex')
subplot(3,2,6)
    plot(E_i(1,tau_record(1,:)),'r','LineWidth',2)
    hold on
    plot(E_i(2,tau_record(2,:)),'b','LineWidth',2)
    leg = legend(legLabel);
    leg.Orientation = 'horizontal';
    leg.Location = 'northeast';
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$E_i(\tau_i)$ [J]','FontSize',15,'interpreter','latex')


figure(4)
subplot(2,1,1)
    for r =1:N_robot
        offset = 0;
        f_mat  = [];
        x_mat  = [];
        for i = 1:size(tau_record,2)
            x      = 0:c_i(r,tau_record(r,i));
            zeta   = 1/(0.2*x(end));
            cnst   = zeta*E_i(r,tau_record(r,i))/(1 - exp(-zeta*x(end)));
            f      = cnst*exp(-zeta*x);
            f_mat  = [f_mat, f];
            x_mat  = [x_mat, x + offset];
            offset = offset + x(end);
        end
        p = area(x_mat,f_mat);

        hold on
        alpha(p,0.3)    
    end
    grid minor
%     xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$E_i$ [J]','FontSize',15,'interpreter','latex')
subplot(2,1,2)
    E_tot = zeros(N_robot,size(tau_record,2));
    c_tot = zeros(N_robot,size(tau_record,2));
    for r =1:N_robot
       E_tot(r,:) = E_i(r,tau_record(r,:));
       c_tot(r,:) = c_i(r,tau_record(r,:));
    end
    bar(cumsum(sum(E_tot,1)))
    hold on
    plot([0:N_tasks/r],sum(sum(E_tot,1))*ones(N_tasks/r+1,1),'k--','LineWidth',2)    
    ylabel('$E_{tot}$ [J]','FontSize',15,'interpreter','latex')
    grid minor



