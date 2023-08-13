close all
clc
clearvars

cd(fileparts(matlab.desktop.editor.getActiveFilename));
load('robot_plots_embodied_ai_paper_update.mat')
 

%% **************************************************************************
%                             DATA CENTERS                                *
% *************************************************************************
%% Data center energy consumption
close all
clc
clear fig ax plt leg

fig = figure(1);
ax  = gca;
hold on
plt1 =  plot(data_centers_table.year,data_centers_table.best,'k--');
plt2 =  plot(data_centers_table.year,data_centers_table.expected,'k-');
plt3 =  plot(data_centers_table.year,data_centers_table.worst,'k:');
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
  
% Figure saving
SAVE_FIG = 0;
if SAVE_FIG == 1
    %export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\data_center_energy_consumption','-pdf')
    exportgraphics(gcf,'/home/diaz/Documents/Knowledge-sharing-as-the-key-to-a-sustainable-energy-balance-in-embodied-AI/fig/data_center_energy_consumption.png','Resolution',600)
    close(gcf);
end    

%% **************************************************************************
%                        ROBOTS INSTALLATION SHARE                        *
% *************************************************************************

%% Industrial and cobot share
close all
clc
clear fig ax plt leg

fig = figure('color','w');
plt = bar(transpose(ir_cr_share_table.year), ...
          transpose([ir_cr_share_table.IR_share, ir_cr_share_table.CR_share]),"stacked");
ax = gca;
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Share [\%]','interpreter','Latex','FontSize', 20);
leg = legend('Industrial','Cobots','FontSize', 20);

fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 2, 0.5)
axis tight

% Figure saving
SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(fig,'/home/diaz/Documents/Knowledge-sharing-as-the-key-to-a-sustainable-energy-balance-in-embodied-AI/fig/share_industrial_and_cobots.png','Resolution',600)
    close(fig);
end  

%% **************************************************************************
%                        INDUSTRIAL ROBOTS                                *
% *************************************************************************

%% Installed base of industrial robots
close all
clc
clear fig ax plt leg

warning('Data from: ./miscellaneous/industrial_robots_energy_consumption_statistics_as_of_2021_UPDATE.xlsx')
fig = figure(1);
ax  = gca;
hold on
plt1 =  plot(irOpStock_table.year,irOpStock_table.actual,'k-');
plt2 =  plot(irOpStock_table.year,irOpStock_table.rate12ppy,'k--');
plt3 =  plot(irOpStock_table.year,irOpStock_table.rate25ppy,'k:');
plt  = [plt1,plt2,plt3];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Units/Year [x$10^6$]','interpreter','Latex','FontSize', 20);
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
   
% Figure saving
SAVE_FIG = 0;
if SAVE_FIG == 1
    %export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\ir_units_projections','-pdf')
    %exportgraphics(fig,'./figures/ir_units_projections.pdf','Resolution',300)
    exportgraphics(fig,'./figures/ir_units_projections.png','Resolution',600)
    close(fig);
end    
%% Energy projections for industrial robots
close all
clc
clear fig ax plt leg

warning('Data from: ./miscellaneous/industrial_robots_energy_consumption_statistics_as_of_2021_UPDATE.xlsx')
fig = figure(1);
ax  = gca;
hold on
% plt =  bar(ir_energy(:,1),ir_energy(:,2:3));
plt1 =  plot(irEnDemand_table.year,irEnDemand_table.actual,'k-');
plt2 =  plot(irEnDemand_table.year,irEnDemand_table.rate12ppy,'k--');
plt3 =  plot(irEnDemand_table.year,irEnDemand_table.rate25ppy,'k:');
plt  = [plt1,plt2,plt3];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('$PJ$/Year @ (24/7)','interpreter','Latex','FontSize', 20);
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
    %export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\ir_energy_projections','-pdf')
    %exportgraphics(fig,'./figures/ir_energy_projections.pdf','Resolution',300)
    exportgraphics(fig,'./figures/ir_energy_projections.png','Resolution',600)
    close(fig);
end    

%% **************************************************************************
%                        COLLABORATIVE ROBOTS                             *
% *************************************************************************

%% Operational stock of cobots
close all
clc
clear fig ax plt leg

warning('Data from: ./miscellaneous/industrial_robots_energy_consumption_statistics_as_of_2021_UPDATE.xlsx')
fig = figure(1);
ax  = gca;
hold on
plt1 =  plot(crStockEnergy_table.year(1:6),crStockEnergy_table.op_stock_mean(1:6),'k-');
plt2 =  plot(crStockEnergy_table.year(6:end),crStockEnergy_table.op_stock_mean(6:end),'k--');
plt  = [plt1,plt2];
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('Units/Year [x$10^3$]','interpreter','Latex','FontSize', 20);
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
SAVE_FIG = 0;
if SAVE_FIG == 1
    %export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\cb_units_projections','-pdf')
    %exportgraphics(fig,'./figures/cb_units_projections.pdf','Resolution',300)
    exportgraphics(fig,'./figures/cb_units_projections.png','Resolution',600)
    close(fig);
end 

%% Energy projections for cobots
% *NOTE: Assumed power 2 kW
close all
clc
clear fig ax plt leg
warning('Data from: ./miscellaneous/industrial_robots_energy_consumption_statistics_as_of_2021_UPDATE.xlsx')

fig = figure(1);
ax  = gca;
hold on
plt =  plot(crStockEnergy_table.year,crStockEnergy_table.energy,'k-');
xlabel('Year','interpreter','Latex','FontSize', 20);
ylabel('$PJ$/Year @ (24/7)','interpreter','Latex','FontSize', 20);
% leg = legend('60\%/year','80\%/year', 'FontSize', 20);
leg = [];% legend('Estimate', 'FontSize', 20);
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
    %export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\cb_energy_projections','-pdf')
    %exportgraphics(fig,'./figures/cb_energy_projections.pdf','Resolution',300)
    exportgraphics(fig,'./figures/cb_energy_projections.png','Resolution',600)
    close(gcf);
end

%% Power per payload for cobots
close all
clc
clear fig ax plt leg

warning('Data from: ./miscellaneous/industrial_robots_energy_consumption_statistics_as_of_2021_UPDATE.xlsx')
fig = figure('color','w');
p1 = bar( categorical(cbPower_table.Name),cbPower_table.PowPerLoad);
ax = gca;
ylabel('Power/payload [Watt/kg]','interpreter','Latex','FontSize', 20);
leg = [];
hold on
p2 = plot(categorical(cbPower_table.Name),mean(cbPower_table.PowPerLoad)*ones(size(cbPower_table.Name)),'r-');
leg = legend(p2, "Average power");
plt = [p1 p2];
fcn_scrpt_prepare_figure(fig, ax, plt, leg, [], 8.8, 1, 1)
axis tight
leg.Location = 'northeast';
% Figure saving
SAVE_FIG = 0;
if SAVE_FIG == 1
    %exportgraphics(fig,'/home/diaz/Documents/Knowledge-sharing-as-the-key-to-a-sustainable-energy-balance-in-embodied-AI/fig/cobot_watt_per_kg.png','Resolution',600)
    exportgraphics(fig,'./figures/cobot_watt_per_kg.png','Resolution',600)
    close(fig);
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
    %export_fig('C:\Users\ge73nuk\LRZ Sync+Share\xtras\cb_sales_projections','-pdf')
    exportgraphics(fig,'./figures/cb_sales_projections.pdf','Resolution',300)
    close(fig);
end 