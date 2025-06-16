x = 0:0.01:100;

c_j = 100;
k_max = 500;
k_min = 100;
 alpha = -(1/5)*log(k_min/k_max);


k = k_max*exp(-5*x*alpha/c_j);
c_str = c_0*sig;

close all
plot(x,k)

%%
% 
x = 1:1:20;
y = 1:1:100;

[X,Y] = meshgrid(x,y);%(1:1:100);
% z = x.*(1-exp(-0.1*y))./(1-exp(-0.1*x));
% Z = X.*(1-exp(-0.1*Y))./(1-exp(-0.1*X));

Z = X.*((exp(lambda*Y) - 1)*(exp(lambda*(X-Y))))./(exp(lambda*X)-1);

close all
surf(X,Y,Z)

xlabel('Robots','FontSize',20)
ylabel('Tasks','FontSize',20)
zlabel('Energy','FontSize',20)

%%
close all
N = 1:1:1000;
m = 1:10;
n = 1;
lambda = 0.01;
figure
hold on
for i = 1:numel(m)
   E_j = m(i).*exp(-lambda*m(i)*(N-1));
   if i == 1
       plot(N, cumsum(E_j),'k','LineWidth',2)
   else
       plot(N, cumsum(E_j))
   end
end
xlabel('Tasks','FontSize',20)
ylabel('Energy','FontSize',20)







%% ************************************************************************
%                    Knowledge according to similarity
% *************************************************************************
% knowledge = @(x) 1 - exp(-0.1*x);


knowledgeTrasferFactor = @(x) 1 - exp(-0.01*x);

%%
close all
k_bar = [0:0.001:1];
plot(x,exp(-5*k_bar))



%% Knowledge plot
close all

iter_end = 1000;
iter = [1:iter_end];
tconst = 200;%iter_end/10;



plot(iter,1 - exp(-(1/tconst)*iter),'LineWidth',2);
hold on
plot(5*tconst*ones(10,1),linspace(0,1,10),'k--','LineWidth',2)
plot(1:(5*tconst),(1 - exp(-(1/tconst)*5*tconst))*ones(5*tconst,1),'k--','LineWidth',2)
plot(iter, ones(numel(iter),1),'r.','LineWidth',2)
xlabel('Iterations','FontSize',15)
ylabel('Knowledge','FontSize',15)

%%
close all
plot(iter,1-exp(-(1/tcnst_1)*iter),'r','LineWidth',2);
hold on
plot(iter,1-exp(-(1/tcnst_2)*iter),'b','LineWidth',2);
plot(iter,1-exp(-(1/tcnst_1 + 1/tcnst_2)*iter),'k','LineWidth',2);
xlabel('Iterations','FontSize',15)
ylabel('Knowledge','FontSize',15)

%% ************************************************************************
%           Effects of knowledge transfer in different scenarios
% *************************************************************************
% NOTE: here it is considered that two agents i and j are learning tasks i and j 
clc
close all

simil    = 0.6;

iter_end = 1800;
iter     = [1:iter_end];

% The time constant of the task is related to the task complexity
tcnst_1  = 300;%iter_end/10;
tcnst_2  = 100;%iter_end/10;

a1 = 1;%1/(tcnst_1*(4 + exp(-5)));
a2 = 1;%1/(tcnst_2*(4 + exp(-5)));

k1  = 1 - exp(-(1/tcnst_1)*iter);
k1s = simil*(1 - exp(-(1/tcnst_1)*iter));
k2  = 1 - exp(-(1/tcnst_2)*iter);
% k2 = 1 - exp(-(1)*iter);
shift = -tcnst_1*log(1-simil);

% Notation:
% * kiaj -> knowledge aquired from i after aquiring j (transfer learning) 
% * kifj -> knowledge aquired from i receiving concurrently knowledge from
%           j. Note that i is only learning the remaining knowledge that is not coming from j 
% * kiwj -> knowledge aquired from i and j 

% k1a2 = 1 - exp(-(1/tcnst_1)*(iter + shift)) - 0*simil;
k1a2 = 1 - exp(-(1/tcnst_1)*(iter + shift));
k1f2 = (1 - simil)*(1 - exp(-(1/tcnst_1)*(iter))) + simil*k2;
k1w2 = (1 - simil)*(1 - exp(-(1/tcnst_1)*(iter))) + ...
                           simil*(1 - exp(-(1/tcnst_1 + 1/tcnst_2)*iter));
% k2to1 = 1 - exp(-((1 + simil*k2)./tcnst_1).*iter);
% k1to2 = 1 - exp(-(1/tcnst_1)*(iter + 1*shift)) - 0*simil;

shift = -tcnst_2*log(1-simil);
k2a1  = 1 - exp(-(1/tcnst_2)*(iter + shift));
k2f1 = (1 - simil)*k2 + simil*k1;
k2w1 = (1 - simil)*k2 + simil*(1 - exp(-(1/tcnst_1 + 1/tcnst_2)*iter));



figure(1)
hold on
p1 = plot(iter,k1,'b','LineWidth',2);
p2 = plot(iter,k1a2,'b:','LineWidth',2);
% p3 = plot(iter,k1f2,'b--','LineWidth',2);
p4 = plot(iter,k1w2,'b-.','LineWidth',2);

p5 = plot(iter,k2,'r','LineWidth',2);
p6 = plot(iter,k2a1,'r:','LineWidth',2);
% p7 = plot(iter,k2f1,'r--','LineWidth',2);
p8 = plot(iter,k2w1,'r-.','LineWidth',2);

plot(tcnst_1*ones(100,1),linspace(0,max(a1,a2),100),'b:')
plot(tcnst_2*ones(100,1),linspace(0,max(a1,a2),100),'r:')
plot(5*tcnst_1*ones(100,1),linspace(0,max(a1,a2),100),'b-.')
plot(5*tcnst_2*ones(100,1),linspace(0,max(a1,a2),100),'r-.')

leg = legend([p1,p2,p4,p5,p6,p8], ...
    '$\kappa_1$',...
    '$\kappa_{1\leftarrow 2}$',...
    '$\kappa_{1+2}$',...
    '$\kappa_2$',...
    '$\kappa_{2\leftarrow 1}$',...
    '$\kappa_{2+1}$');

% leg = legend([p1,p2,p3,p4,p5,p6,p7,p8], ...
%     '$\kappa_1$',...
%     '$\kappa_{1\leftarrow 2}$',...
%     '$\kappa_{1|2}$',...
%     '$\kappa_{1+2}$',...
%     '$\kappa_2$',...
%     '$\kappa_{2\leftarrow 1}$',...
%     '$\kappa_{2|1}$',...
%     '$\kappa_{2+1}$');

leg.Interpreter = 'latex';
leg.FontSize = 15;
% plot(5*tcnst_1*ones(10,1),linspace(0,1,10),'b--','LineWidth',2)
% plot(1:(5*tcnst_1),(1 - exp(-(1/tcnst_1)*5*tcnst_1))*ones(5*tcnst_1,1),'b--','LineWidth',2)
% plot(5*tcnst_2*ones(10,1),linspace(0,1,10),'r--','LineWidth',2)
% plot(1:(5*tcnst_2),(1 - exp(-(1/tcnst_2)*5*tcnst_2))*ones(5*tcnst_2,1),'r--','LineWidth',2)
% plot(iter, ones(numel(iter),1),'k.','LineWidth',2)
xlabel('Iterations','FontSize',15)
ylabel('Knowledge','FontSize',15)
title(['$\sigma = ',num2str(simil),'$'],'interpreter','latex','FontSize',15)
%%
close all
figure(1)
hold on
p1 = plot(iter,k1,'b','LineWidth',3);
p2 = plot(iter,k1a2,'b:','LineWidth',3);
% p3 = plot(iter,k1f2,'b--','LineWidth',3,'Color',[0.7 0.7 0.7]);
p3 = patchline(iter,k1f2,'linestyle','--','edgecolor','b','linewidth',3,'edgealpha',0.2);
p4 = plot(iter,k1w2,'b-.','LineWidth',3);

plot(tcnst_1*ones(100,1),linspace(0,max(a1,a2),100),'b:')
plot(tcnst_2*ones(100,1),linspace(0,max(a1,a2),100),'r:')
plot(5*tcnst_1*ones(100,1),linspace(0,max(a1,a2),100),'b-.')
plot(5*tcnst_2*ones(100,1),linspace(0,max(a1,a2),100),'r-.')

leg = legend([p1,p2,p3,p4], ...
    '$\kappa_1$',...
    '$\kappa_{1\leftarrow 2}$',...
    '$\kappa_{1|2}$',...
    '$\kappa_{1+2}$');
leg.Interpreter = 'latex';
leg.FontSize = 15;
% plot(5*tcnst_1*ones(10,1),linspace(0,1,10),'b--','LineWidth',2)
% plot(1:(5*tcnst_1),(1 - exp(-(1/tcnst_1)*5*tcnst_1))*ones(5*tcnst_1,1),'b--','LineWidth',2)
% plot(5*tcnst_2*ones(10,1),linspace(0,1,10),'r--','LineWidth',2)
% plot(1:(5*tcnst_2),(1 - exp(-(1/tcnst_2)*5*tcnst_2))*ones(5*tcnst_2,1),'r--','LineWidth',2)
% plot(iter, ones(numel(iter),1),'k.','LineWidth',2)
xlabel('Iterations','FontSize',15)
ylabel('Knowledge','FontSize',15)
title(['$\sigma = ',num2str(simil),'$'],'interpreter','latex','FontSize',15)
%% ************************************************************************
%                    Tasks and their knowledge areas
% *************************************************************************

close all
figure
hold on

num_sel_tasks = 10;
c = randperm(size(T,1),num_sel_tasks);
for i=1:num_sel_tasks
    p = c(i);
    plt1 = plot(T(p,1),T(p,2),'k.','LineWidth',3);
    text(1.01*T(p,1),1.01*T(p,2),['$\tau_' num2str(i),'$'],'interpreter','latex','FontSize',15)
    plt2 = circle(T(p,1),T(p,2),0.2);
%     plt2 = circle(T(p,1),T(p,2),1/sqrt(pi));
    
end
axis equal

title('$\tau_i$','FontSize',15,'interpreter','latex')

%%
close all
figure(2)
hold on
p1 = plot(iter,(simil)*k1,'b','LineWidth',2);
p2 = plot(iter,(simil)*k2,'r','LineWidth',2);
p3 = plot(iter,simil*(1 - exp(-(1/tcnst_1 + 1/tcnst_2)*iter)),'k--','LineWidth',2);
plot(iter,(1 - exp(-(1/100)*iter)),'m--','LineWidth',2);
plot(tcnst_1*ones(100,1),linspace(0,max(a1,a2),100),'b:')
plot(tcnst_2*ones(100,1),linspace(0,max(a1,a2),100),'r:')
plot(5*tcnst_1*ones(100,1),linspace(0,max(a1,a2),100),'b-.')
plot(5*tcnst_2*ones(100,1),linspace(0,max(a1,a2),100),'r-.')
leg = legend([p1,p2,p3],'$\sigma k_1$','$\sigma k_2$','$k_{12}$');
leg.Interpreter = 'latex';
leg.FontSize = 15;

%%
close all
plot(iter,k1w2)
hold on
plot(iter, 1 - exp(-(1/220)*(iter)),'r')

%%

close all


k11  = exp(-iter/tcnst_1)/tcnst_1;
plot(tcnst_1*k11)
hold on
plot(cumtrapz(k11))
% plot(1:5*tcnst_1,cumtrapz(1:5*tcnst_1,k11(1:5*tcnst_1)))

%%

close all

iter_end = 1000;
iter     = 1:iter_end;
tcnst_1  = 200;%iter_end/10;
tcnst_2  = 100;%iter_end/10;
k1       = zeros(iter_end,1);
k2       = zeros(iter_end,1);
ktf      = 0.7;
for i=1:iter_end
    if i == 1
        k1(i) = 1 - exp(-(1/tcnst_1)*i);
        k2(i) = 1 - exp(-(1/tcnst_2)*i);
    else
        k1(i) = 1 - exp(-((1 + ktf*k2(i-1))/tcnst_1)*i);
        k2(i) = 1 - exp(-((1 + ktf*k1(i-1))/tcnst_2)*i);
    end
end


% p1 = plot(iter,k1,'b:','LineWidth',2);
% p2 = plot(iter,k2,'r:','LineWidth',2);


figure(2)
hold on
p1 = plot(iter,k1,'b','LineWidth',2);
p2 = plot(iter,k2,'r','LineWidth',2);
p3 = plot(iter,1 - exp(-(1/tcnst_1)*iter),'b--','LineWidth',2);
p4 = plot(iter,1 - exp(-(1/tcnst_2)*iter),'r--','LineWidth',2);
leg = legend([p1,p2,p3,p4],'$k_{2\to 1}$','$k_{1\to 2}$','$k_1$','$k_2$');
leg.Interpreter = 'latex';
leg.FontSize    = 15;
xlabel('Iterations','FontSize',15)
ylabel('Knowledge','FontSize',15)

%%
clc
close all
r = [0.001:0.001:1];
% r = [1000:-0.1:1E-3];
s = 1./r;
k_factor = 1 - exp(-0.01*s);
% k = 1- exp(-s);

figure
subplot(3,1,1)
plot(r,s,'r','LineWidth',2)
xlabel('r','FontSize',15)
ylabel('1/r','FontSize',15)
subplot(3,1,2)
plot(r,k_factor,'b','LineWidth',2)
xlabel('r','FontSize',15)
ylabel('k_{factor}','FontSize',15)
subplot(3,1,3)
plot(r,k_factor,'b','LineWidth',2)
xlabel('r','FontSize',15)
ylabel('k_{factor}','FontSize',15)

%%
% rng('default')
clc
close all
N_tasks = 1000;
tau     = 1:N_tasks;


r       = normrnd(0.1,10,N_tasks,1);
r(r<0)  = rand(1);

r       = S(end,1:end-1)';
N_tasks = numel(r);
tau     = 1:N_tasks;
% r       = sort(rand(N_tasks,1),'descend');
r       = rand(N_tasks,1);
k       = 1 - exp(-0.01*(1./r));
% k       = knowledge(1./r);
k_bar   = cumsum(k);
d       = exp(-0.01*k_bar);
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
    

%%


close all
figure
hold on

num_sel_tasks = 5;
c = randperm(size(T,1),num_sel_tasks);
for i=1:num_sel_tasks
    p = c(i);
    plt1 = plot(T(p,1),T(p,2),'k.','LineWidth',3);
    text(1.01*T(p,1),1.01*T(p,2),['$\tau_' num2str(i),'$'],'interpreter','latex','FontSize',15)
    plt2 = circle(T(p,1),T(p,2),0.2);
%     plt2 = circle(T(p,1),T(p,2),1/sqrt(pi));
    
end
axis equal

title('$\tau_i$','FontSize',15,'interpreter','latex')

%% ************************************************************************
%              Incremental + Transfer learning of tasks in T              *
% *************************************************************************


clc
close all

rng('default')
N_tasks  = 10;
a        = 0;
b        = 1;
T_lims   = round(sort((b-a).*rand(N_tasks,2) + a,2),3);
T_center = mean(T_lims,2); 
D        = squareform(pdist([T_center,zeros(N_tasks,1)]));
T        = cell(N_tasks,1);
figure 
hold on
T_pool = [];
for i=1:N_tasks
    range  = T_lims(i,1):0.001:T_lims(i,2);
    T{i}   = range;
    T_pool = [T_pool, T{i}];
    plot(range,i*ones(numel(range),1),'LineWidth',2);
    plot(T_center(i),i,'ok','LineWidth',3);
end
% range = unique(T_pool);
% plot(range,(N_tasks+1)*ones(numel(range),1),'k--','LineWidth',2);

c_0 = 100;
e_0 = 100; %[Joules]
P   = 100;

d       = zeros(N_tasks,1);
c_i     = zeros(N_tasks,1);
E_i     = zeros(N_tasks,1);
t_i     = zeros(N_tasks,1);


k_gain = zeros(N_tasks,1);
sigma  = zeros(N_tasks,1);
T_pool = [];
A_pool = 0;
% [~,indices]=sort(D(1,:));
for tau = 1:N_tasks
    A_task      = numel(T{tau});        
    A_task_pool = numel(unique([T_pool, T{tau}]));
    k_gain(tau) = (A_task_pool - A_pool)/A_task;
    sigma(tau)  = 1-k_gain(tau);
    T_pool      = [T_pool, T{tau}];
    A_pool      = A_task_pool;
    c_i(tau)    = max(1,ceil(k_gain(tau)*c_0));
    E_i(tau)    = c_i(tau)*e_0;
    t_i(tau)    = E_i(tau)/P;    
end
figure(1)
subplot(3,1,1)
hold on
plot(k_gain)
plot(sigma)
xlabel('$\tau_j$','FontSize',25,'interpreter','latex')
ylabel('$\sigma(\tau_j,\mathcal{T}_l)$','FontSize',25,'interpreter','latex')
subplot(3,1,2)
stairs(c_i)
xlabel('$\tau_j$','FontSize',25,'interpreter','latex')
ylabel('$c_j^{\star}$','FontSize',25,'interpreter','latex')
subplot(3,1,3)
stairs(E_i)
xlabel('$\tau_j$','FontSize',25,'interpreter','latex')
ylabel('$E_j$ [J]','FontSize',25,'interpreter','latex')

% Learning episodes =======================================================
offset = 0;
f_mat = [];
x_mat = [];
for tau = 1:N_tasks
    x      = 0:c_i(tau);
    zeta   = 1/(0.11*x(end));
    cnst   = zeta*E_i(tau)/(1 - exp(-zeta*x(end)));
    f      = cnst*exp(-zeta*x);
%     plot(x + offset,f)
    [E_i(tau), trapz(f)]
    f_mat  = [f_mat, f];
    x_mat  = [x_mat, x + offset];
    offset = offset + x(end);    
end

figure(2)
    subplot(2,1,1)
    area(x_mat,f_mat)
    xlabel('k [Iterations]','FontSize',15,'interpreter','latex')
    ylabel('$E_i(k)$ [J]','FontSize',15,'interpreter','latex')
    grid minor

    subplot(2,1,2)
    bar(1:N_tasks,cumsum(E_i))
    hold on
    plot(sum(E_i)*ones(N_tasks,1),'k--','LineWidth',2)
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$E_{tot}$ [J]','FontSize',15,'interpreter','latex')

%% ************************************************************************
%                          Collective learning                            *
% *************************************************************************

clc
close all

rng('default')
N_tasks  = 600;
a        = 0;
b        = 1;
T_lims   = round(sort((b-a).*rand(N_tasks,2) + a,2),3);
T_center = mean(T_lims,2); 
D        = squareform(pdist([T_center,zeros(N_tasks,1)]));
T        = cell(N_tasks,1);
% figure 
% hold on
T_pool = [];
for i=1:N_tasks
    range  = T_lims(i,1):0.001:T_lims(i,2);
    T{i}   = range;
    T_pool = [T_pool, T{i}];
%     plot(range,i*ones(numel(range),1),'LineWidth',2);
%     plot(T_center(i),i,'ok','LineWidth',3);
end


N_robot = 3;
lambda  = zeros(N_tasks,1);
d       = zeros(N_robot, N_tasks);
k_gain  = zeros(N_robot, N_tasks);
sigma   = zeros(N_robot, N_tasks);
c_i     = zeros(N_robot, N_tasks);
E_i     = zeros(N_robot, N_tasks);
t_i     = zeros(N_robot, N_tasks);


task_log = zeros(N_tasks,1);
tau_record = [];

complete = 0;
iter     = 0;
rng('default')
T_pool = [];
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
            tau_pool_ex  = tau_choice(tau_choice ~= tau_choice(r));
            T_pool_r     = [];
            for i=1:numel(tau_pool_ex)
                T_pool_r = [T_pool_r, T{tau_pool_ex(i)}];
            end
            if NOT_COLLETIVE == 1
                T_pool_r =  [];
            end
            A_pool                  = numel(unique([T_pool, T_pool_r,]));
            A_task                  = numel(T{tau_choice(r)}); 
            A_task_pool             = numel(unique([T_pool, T_pool_r, T{tau_choice(r)}]));
            k_gain(r,tau_choice(r)) = (A_task_pool - A_pool)/A_task;
            sigma(r,tau_choice(r))  = 1 - k_gain(r,tau_choice(r));
            c_i(r,tau_choice(r))    = max(1,ceil(k_gain(r,tau_choice(r))*c_0));
            E_i(r,tau_choice(r))    = c_i(r, tau_choice(r))*e_0;
            t_i(r,tau_choice(r))    = E_i(r, tau_choice(r))/P;      
        end
        
        for i=1:numel(tau_choice)
            T_pool = [T_pool, T{tau_choice(i)}];
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

close all
figure
    hold on
    legLabel = {N_robot};
    temp = [];
    for r = 1:N_robot
        plot(sigma(r,tau_record(r,:)),'-','LineWidth',2)
        temp = [temp;sigma(r,tau_record(r,:))];
        legLabel{r} = ['r_', num2str(r)];
    end
    plot(1:200,mean(temp,1),'k.-','LineWidth',2)
    clear temp
    if N_robot<3
        leg = legend(legLabel);
        leg.Orientation = 'horizontal';
        leg.Location = 'northeast';
    end
    xlabel('$\tau_j$','FontSize',15,'interpreter','latex')
    ylabel('$\sigma(\tau_j,\mathcal{T}_l)$','FontSize',15,'interpreter','latex')
%%
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
    if N_robot<3
        leg = legend(legLabel);
        leg.Orientation = 'horizontal';
        leg.Location = 'northeast';
    end
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$d(\tau_i,\mathcal{T}_l)$','FontSize',15,'interpreter','latex')
subplot(3,2,4)
%     plot(c_i(1,tau_record(1,:)),'r','LineWidth',2)
%     hold on
%     plot(c_i(2,tau_record(2,:)),'b','LineWidth',2)
    
    hold on
    for r = 1:N_robot
        plot(c_i(r,tau_record(r,:)),'LineWidth',2)
    end
    
    
    
    if N_robot<3
        leg = legend(legLabel);
        leg.Orientation = 'horizontal';
        leg.Location = 'northeast';
    end
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$c_i(\tau_i)$ [iterations]','FontSize',15,'interpreter','latex')
subplot(3,2,6)
%     plot(E_i(1,tau_record(1,:)),'r','LineWidth',2)
%     hold on
%     plot(E_i(2,tau_record(2,:)),'b','LineWidth',2)

    hold on
    for r = 1:N_robot
        plot(E_i(r,tau_record(r,:)),'LineWidth',2)
    end
    
    
    if N_robot<3
        leg = legend(legLabel);
        leg.Orientation = 'horizontal';
        leg.Location = 'northeast';
    end
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$E_i(\tau_i)$ [J]','FontSize',15,'interpreter','latex')
%%
close all
figure(4)
% subplot(N_robot,1,1)
    for r =1:N_robot
subplot(N_robot+1,1,r)        
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
        p = area(x_mat,f_mat,'FaceColor', [0.7 0.7 0.7]);
        hold on
%         plot()
        alpha(p,0.3)    
        grid minor
%     xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
        ylabel('$E^{\rho_j}_i$ [J]','FontSize',15,'interpreter','latex')
    end
subplot(N_robot+1,1,N_robot+1)
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

    
%%
close all
x = 0:2000;
y1 = 1 - exp(-(1/300)*x);
y2 = 1 - exp(-(1/300 + 1/300)*x);
plot(x,y1)
hold on
plot(x,y2)

    
%% ************************************************************************
%                          Collective learning                            *
% *************************************************************************

rng('default')
N_tasks = 12;
% T       = normrnd(0,100,N_tasks,2);
T       = rand(N_tasks,2);
D       = squareform(pdist(T));
S       = D./max(D,[],'all');


N_robot = 3;
lambda  = zeros(N_tasks,1);
d       = zeros(N_robot, N_tasks);
k_gain  = zeros(N_robot, N_tasks);
c_i     = zeros(N_robot, N_tasks);
E_i     = zeros(N_robot, N_tasks);
t_i     = zeros(N_robot, N_tasks);


task_log = zeros(N_tasks,1);
tau_record = [];

complete = 0;
iter     = 0;
rng('default')
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
                else
                    tau_pool_ex           = tau_choice(tau_choice ~= tau_choice(r));
                    s_inv                 = 1./S(tau_choice(r),[task_pool;tau_pool_ex]);
                end
                ktf                  = knowledgeTrasferFactor(s_inv);
                d(r,tau_choice(r))   = exp(-1*sum(ktf));                  
                c_i(r,tau_choice(r)) = max(1,ceil(d(r,tau_choice(r))*c_0));
                E_i(r,tau_choice(r)) = c_i(r, tau_choice(r))*e_0;
                t_i(r,tau_choice(r)) = E_i(r, tau_choice(r))/P;
                   
                
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
    if N_robot<3
        leg = legend(legLabel);
        leg.Orientation = 'horizontal';
        leg.Location = 'northeast';
    end
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$d(\tau_i,\mathcal{T}_l)$','FontSize',15,'interpreter','latex')
subplot(3,2,4)
%     plot(c_i(1,tau_record(1,:)),'r','LineWidth',2)
%     hold on
%     plot(c_i(2,tau_record(2,:)),'b','LineWidth',2)
    
    hold on
    for r = 1:N_robot
        plot(c_i(r,tau_record(r,:)),'LineWidth',2)
    end
    
    
    
    if N_robot<3
        leg = legend(legLabel);
        leg.Orientation = 'horizontal';
        leg.Location = 'northeast';
    end
    xlabel('$\tau_i$','FontSize',15,'interpreter','latex')
    ylabel('$c_i(\tau_i)$ [iterations]','FontSize',15,'interpreter','latex')
subplot(3,2,6)
%     plot(E_i(1,tau_record(1,:)),'r','LineWidth',2)
%     hold on
%     plot(E_i(2,tau_record(2,:)),'b','LineWidth',2)

    hold on
    for r = 1:N_robot
        plot(E_i(r,tau_record(r,:)),'LineWidth',2)
    end
    
    
    if N_robot<3
        leg = legend(legLabel);
        leg.Orientation = 'horizontal';
        leg.Location = 'northeast';
    end
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

%%
figure
hold on
for r =1:N_robot
    plot((c_i(r,tau_record(r,:))))
end

