close all
clc

clear fig ax plt leg

fig = figure(1);
ax  = gca;
hold on
plt1 =  plot(data_servers(1,:),data_servers(2,:),'k--');
plt2 =  plot(data_servers(1,:),data_servers(3,:),'k-');
plt3 =  plot(data_servers(1,:),data_servers(4,:),'k:');
plt  = [plt1,plt2,plt3];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Energy [TWh]','interpreter','Latex','FontSize', 20);
leg = legend('Best','Expected','Worst', 'FontSize', 20);
set(leg,'Interpreter','latex')

% (fig, ax, plt, leg, tx, text_width, k_scaling, k_width_height)

fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 2, 0.5)
grid off
leg.Location    = 'northwest'; 
leg.Orientation = 'horizontal'; 
% leg.FontSize    = 50;
% ax.FontSize     = 40; 
% set(plt,'LineWidth',3)
   
% Figure storage
SAVE_FIG = 1;
if SAVE_FIG == 1
    export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\data_center_energy_consumption','-pdf')
    close(gcf);
end    

%% Installed base of industrial robots

clc
close all

irOpStock(irOpStock == 0) = NaN;

fig = figure(1);
ax  = gca;
hold on
% plt1 =  plot(ir_units(:,1),ir_units(:,2)/1000,'k-');
% plt2 =  plot(ir_installed(:,1),ir_installed(:,2),'k--');
% plt3 =  plot(ir_installed(:,1),ir_installed(:,3),'k:');
plt1 =  plot(irOpStock(:,1),irOpStock(:,2),'k-');
plt2 =  plot(irOpStock(:,1),irOpStock(:,3),'k--');
plt3 =  plot(irOpStock(:,1),irOpStock(:,4),'k:');
plt  = [plt1,plt2,plt3];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Operational stock [x$10^6$]','interpreter','Latex','FontSize', 20);
leg = legend('Actual','12\%','25\%', 'FontSize', 20);
set(leg,'Interpreter','latex')
xlim([2009 2025])
fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 2, 0.5)
grid off
leg.Location    = 'northwest'; 
leg.Orientation = 'horizontal'; 
% leg.FontSize    = 50;
% ax.FontSize     = 40; 
% set(plt,'LineWidth',3)
   
% Figure storage
SAVE_FIG = 1;
if SAVE_FIG == 1
    export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\ir_units_projections','-pdf')
    close(gcf);
end    
%% Energy projections for industrial robots
clc
close all

irEnDemand(irEnDemand == 0) = NaN;

fig = figure(1);
ax  = gca;
hold on
% plt =  bar(ir_energy(:,1),ir_energy(:,2:3));
plt1 =  plot(irEnDemand(:,1),irEnDemand(:,2),'k-');
plt2 =  plot(irEnDemand(:,1),irEnDemand(:,3),'k--');
plt3 =  plot(irEnDemand(:,1),irEnDemand(:,4),'k:');
plt  = [plt1,plt2,plt3];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Energy [$PJ$] (24/7)','interpreter','Latex','FontSize', 20);
leg = legend('Actual','12\%/year','25\%/year', 'FontSize', 20);
set(leg,'Interpreter','latex')
xlim([2009 2025])
fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 2, 0.5)
grid off
leg.Location    = 'northwest'; 
leg.Orientation = 'horizontal'; 
% leg.FontSize    = 50;
% ax.FontSize     = 40; 
% set(plt,'LineWidth',3)
   
% Figure storage
SAVE_FIG = 0;
if SAVE_FIG == 1
    export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\ir_energy_projections','-pdf')
    close(gcf);
end    

%% ************************************************************************
%                        COLLABORATIVE ROBOTS                             *
% *************************************************************************

%% Operational stock of cobots

clc
close all

fig = figure(1);
ax  = gca;
hold on
plt1 =  plot(cbOpStock(1:5,1),cbOpStock(1:5,2)/1000,'k-');
plt2 =  plot(cbOpStock(5:end,1),cbOpStock(5:end,2)/1000,'k--');
plt  = [plt1,plt2];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Operational stock [x$10^3$]','interpreter','Latex','FontSize', 20);
leg = legend('Reported','Projected', 'FontSize', 20);
set(leg,'Interpreter','latex')

fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 2, 0.5)
grid off
leg.Location    = 'northwest'; 
leg.Orientation = 'horizontal'; 
% leg.FontSize    = 50;
% ax.FontSize     = 40; 
% set(plt,'LineWidth',3)
   
% Figure storage
SAVE_FIG = 1;
if SAVE_FIG == 1
    export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\cb_units_projections','-pdf')
    close(gcf);
end 


%% Projected sales of cobots

clc
fig = figure(1);
ax  = gca;
hold on
plt =  bar(cb_sales(:,1),cb_sales(:,2:3));
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Unit sales [x$10^3$]','interpreter','Latex','FontSize', 20);
leg = legend('12\%/year','25\%/year', 'FontSize', 20);
set(leg,'Interpreter','latex')

fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 2, 0.5)
grid off
leg.Location    = 'northwest'; 
leg.Orientation = 'horizontal'; 
% leg.FontSize    = 50;
% ax.FontSize     = 40; 
% set(plt,'LineWidth',3)
   
% Figure storage
SAVE_FIG = 0;
if SAVE_FIG == 1
    export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\cb_sales_projections','-pdf')
    close(gcf);
end 
%% Energy projections for cobots
clc
close all 

fig = figure(1);
ax  = gca;
hold on
% plt =  bar(cb_energy(:,1),cb_energy(:,2:3));
plt =  plot(cbEnDemand(:,1),cbEnDemand(:,2),'k--');
% plt2 =  plot(data(1,:),data(3,:),'k-');
% plt3 =  plot(data(1,:),data(4,:),'k:');
% plt  = [plt1,plt2,plt3];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Energy [$PJ$] (24/7)','interpreter','Latex','FontSize', 20);
% leg = legend('60\%/year','80\%/year', 'FontSize', 20);
leg = legend('Estimate', 'FontSize', 20);
set(leg,'Interpreter','latex')

fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 2, 0.5)
grid off
leg.Location    = 'northwest'; 
leg.Orientation = 'horizontal'; 
% leg.FontSize    = 50;
% ax.FontSize     = 40; 
% set(plt,'LineWidth',3)
   
% Figure storage
SAVE_FIG = 1;
if SAVE_FIG == 1
    export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\cb_energy_projections','-pdf')
    close(gcf);
end

%%

for i = 1:5
    ir_units(8+i,3) = ir_units(8+i,2)*market_share(i,3)/market_share(i,2); 
end

close all
hold on
plot(ir_units(:,1),ir_units(:,2))
plot(ir_units(9:13,1),ir_units(9:13,3))
