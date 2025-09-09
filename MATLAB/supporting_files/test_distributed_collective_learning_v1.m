close all
clc
rng('default')
clear c_jk_cl_dist_episodes_ParamSweep learnedSkillsStorage
for iter = 1:1
    close all
    [c_jk_cl_dist_episodes_ParamSweep{iter}, learnedSkillsStorage{iter}] = runCollectiveLearningDistributed(parameters, 0.1);
end

%%

clc
close all
cmap = flip(distinguishable_colors(size(complexities{1},2),{'w','k'}),1);

pl = [];
clear learnedSkillsRate
x = [1:6];

% for index = 1:size(complexities{1},2)
for index = size(complexities{1},2):-1:1
    the_learned_skills           = learnedSkillsStorage{1}(:,index);
    learnedSkillsRate(:,index) = floor(100*(ceil(mean(the_learned_skills,2))./parameters.totalSkills));
    sh = plot(x,learnedSkillsRate(:,index),'Color',cmap(index,:));
    pl = [pl, sh];
    hold on

% bottom = 0; %identify bottom; or use min(sh.YData)
% xx = [sh.XData(1),repelem(sh.XData(2:end),2)];
% yy = [repelem(sh.YData(1:end-1),2),sh.YData(end)];
% % plot(x,y,'y:') %should match stair plot
% % Fill area
% fill([xx,fliplr(xx)],[yy,bottom*ones(size(yy))],cmap(index,:),'FaceAlpha', 0.1,'EdgeColor','none')



end
aux =round(linspace(-0.2,1,size(complexities{1},2)),2);
leg    = legend(pl,arrayfun(@(i) num2str(aux(i)),1:size(complexities{1},2),'UniformOutput',false));
ylim([0 100])
xticks([1:6])
xticklabels({'4','8','16','32','64','128'})
% yticks([1E+0 1E+1 1E+2 1E+3 1E+4])
% yticklabels({'10^0','10^1', '10^2', '10^3', '10^4'})
xlabel('Number of robots','FontSize',25)
ylabel('Success rate','FontSize',25)
% set(gca, 'YScale', 'log')

%%
close all
clc
fig = figure('Color','w');
p   = imagesc(learnedSkillsRate);
colormap hot
yticklabels({'4','8','16','32','64','128'})
aux =round(linspace(-0.2,1,size(complexities{1},2)),2);
xticks([1:size(complexities{1},2)])
xticklabels(arrayfun(@(i) num2str(aux(i)),1:size(complexities{1},2),'UniformOutput',false));
axis square
colorbar
title('Learned Skills Percentage ($\bar{\eta} = 0.1$)','FontSize',25,'Interpreter','latex')
ylabel('Number of robots','FontSize',25)
xlabel('$\bar{\gamma}$','FontSize',25,'Interpreter','latex')
fcn_scrpt_prepare_graph_science_std(fig, gca, p, [], [], 24, 1, 1)
SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','learned_skills_percentage_eta_0.1.png'),'Resolution',600)
    close(gcf);
end 

%%
figure
b =bar(x,the_mean);
b.FaceColor = 'flat';
hold on
the_std  = ceil(std(the_data,[],2))
uppeBound  = the_mean + the_std;
lowerBound = the_mean - the_std;    
%%

cmap = flip(distinguishable_colors(9,{'w','k'}),1);
legends = {...
    '$\bar{\eta}_+,\bar{\gamma}_-$',...
    '$\bar{\eta}_+,\bar{\gamma}_0$',...
    '$\bar{\eta}_+,\bar{\gamma}_+$',...
    '$\bar{\eta}_0,\bar{\gamma}_-$',...
    '$\bar{\eta}_0,\bar{\gamma}_0$',...
    '$\bar{\eta}_0,\bar{\gamma}_+$',...
    '$\bar{\eta}_-,\bar{\gamma}_-$',...
    '$\bar{\eta}_-,\bar{\gamma}_0$',...
    '$\bar{\eta}_-,\bar{\gamma}_+$'};
clc
close all
% fig = figure('color','w');
tl = tiledlayout(3,3);
tl.Title.String = "Total episodes for all skills";
t.Title.FontWeight = 'bold';

for index = 1:9
    nexttile
    upperBound  =  12800*ones(6,1);
    lowerBound= parameters.totalSkills./[4,8,16,32,64,128]'.*parameters.fundamentalComplexity;
    patch([x fliplr(x)], [lowerBound'  fliplr(upperBound')], [0.5  0.5  0.5],'FaceAlpha',0.25,'EdgeColor','w');
    hold on
    the_learned_skills = cell2mat(arrayfun(@(i) c_jk_cl_dist_episodes_ParamSweep{i}(:,index),1:10,'UniformOutput',false));
    the_mean = ceil(mean(the_learned_skills,2));
    the_std  = ceil(std(the_learned_skills,[],2));
    upperBound  = the_mean + the_std;
    lowerBound = the_mean - the_std;    
    x        = 2:7;
    patch([x fliplr(x)], [lowerBound'  fliplr(upperBound')], [0  0  0],'FaceColor',cmap(index,:),'FaceAlpha',0.1,'EdgeColor','w');
    p(index) = plot(x, the_mean,'LineStyle','-', 'LineWidth', 3,'Color','k');%,cmap(index,:));
    aux = scatter(x, the_mean,100,learnedSkillsRate(:,index),'filled','MarkerEdgeColor',cmap(index,:));
    
    cb = colorbar;
    clim(gca,[0, 100]);
    ylabel(cb,'Success rate')
    colormap jet
    ylim([1 12800])
    xticks([1:7])
    xticklabels({'2','4','8','16','32','64','128'})
    yticks([1E+0 1E+1 1E+2 1E+3 1E+4])
    yticklabels({'10^0','10^1', '10^2', '10^3', '10^4'})
    xlabel('Number of robots','FontSize',25)
    ylabel('Complexity','FontSize',25)
    set(gca, 'YScale', 'log')
    
    leg = legend(p(index),legends{index},'Interpreter','latex');
    % axis square

    fcn_scrpt_prepare_graph_science_std(gcf, gca, p(index), leg, [], 6, 1, 1)
    leg.Location = 'northeast';
    leg.Orientation = 'horizontal';
    leg.Interpreter = 'latex';
    leg.Box = 'on';
    leg.FontSize = 15;
end
%%
xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
xlabel('Number of robots','FontSize',25)
ylabel('Complexity [episodes for all skills]','FontSize',25)
leg = legend(p,...
    '$\bar{\eta}_+,\bar{\gamma}_-$',...
    '$\bar{\eta}_+,\bar{\gamma}_0$',...
    '$\bar{\eta}_+,\bar{\gamma}_+$',...
    '$\bar{\eta}_0,\bar{\gamma}_-$',...
    '$\bar{\eta}_0,\bar{\gamma}_0$',...
    '$\bar{\eta}_0,\bar{\gamma}_+$',...
    '$\bar{\eta}_-,\bar{\gamma}_-$',...
    '$\bar{\eta}_-,\bar{\gamma}_0$',...
    '$\bar{\eta}_-,\bar{\gamma}_+$');
fcn_scrpt_prepare_graph_science_std(fig, gca, p, leg, [], 18/2, 3, 1)
axis square
leg.Location = 'northeast';
leg.Orientation = 'horizontal';
leg.Interpreter = 'latex';
leg.Box = 'on';
ylim([1 12800])
xlim([2,7])
% fig = gcf;           % generate a figure
% tightfig(fig);
box on
set(gca, 'YScale', 'log')
pause(1)

SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','total_episodes_per_n_robots_v1.png'),'Resolution',600)
    close(gcf);
end 

%%

cmap = flip(distinguishable_colors(9,{'w','k'}),1);
% allMarkers = {'o','+','*','.','x','s','d','^','v','>','<','p','h'};
allMarkers = {'o','*','x','s','d','^','v','>','<','p','h','.','+'};
clc
close all
fig = figure('color','w');
p   = NaN(1,9);
% plot(x,parameters.totalSkills./[4,8,16,32,64,128].*parameters.fundamentalComplexity,'r:','LineWidth',5)
% pp  = patchline(x,parameters.totalSkills./[4,8,16,32,64,128].*parameters.fundamentalComplexity,'linewidth',300,'linestyle','-','edgecolor','k','linewidth',3,'edgealpha',0.2);

upperBound  =  12800*ones(6,1);
lowerBound= parameters.totalSkills./[4,8,16,32,64,128]'.*parameters.fundamentalComplexity;
 
% upperBound = parameters.totalSkills./[4,8,16,32,64,128]'.*parameters.fundamentalComplexity;
% lowerBound = upperBound - 300;
patch([x fliplr(x)], [lowerBound'  fliplr(upperBound')], [0.5  0.5  0.5],'FaceAlpha',0.25,'EdgeColor','w');
% fill([x fliplr(x)], [lowerBound'  fliplr(upperBound')],'r','FaceAlpha',0.25,'EdgeColor','w');

hold on
for index = 1:9
    the_learned_skills = cell2mat(arrayfun(@(i) c_jk_cl_dist_episodes_ParamSweep{i}(:,index),1:10,'UniformOutput',false));
    the_mean = ceil(mean(the_learned_skills,2));
    the_std  = ceil(std(the_learned_skills,[],2));
    upperBound  = the_mean + the_std;
    lowerBound = the_mean - the_std;    
    x        = 2:7;
    patch([x fliplr(x)], [lowerBound'  fliplr(upperBound')], [0  0  0],'FaceColor',cmap(index,:),'FaceAlpha',0.1,'EdgeColor','w');
    % p(index) = plot(x, the_mean, 'LineWidth', 3,'Color',cmap(index,:),'Marker',allMarkers(index),'MarkerFaceColor',cmap(index,:));
    p(index) = plot(x, the_mean,'LineStyle','-', 'LineWidth', 3,'Color',cmap(index,:));
    % aux = scatter(x, the_mean, 'Marker',allMarkers(index),'MarkerFaceColor',cmap(index,:));
    % aux = scatter(x, the_mean,50,learnedSkillsRate(:,index));
    aux = scatter(x, the_mean,100,learnedSkillsRate(:,index),'filled','MarkerEdgeColor',cmap(index,:));
end
cb = colorbar;
ylabel(cb,'Success rate')
colormap jet

% 
% x = linspace(0,2*pi,50);
% y = sin(x) + randi(50,1,50);
% c = linspace(1,10,length(x));
% scatter(x,y,[],c,'filled')
% colorbar
% colormap jet

% plot(5*ones(size(1E1:100:1E5)),1E0:100:1E5,'k--','LineWidth',3)

xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
xlabel('Number of robots','FontSize',25)
ylabel('Complexity [episodes for all skills]','FontSize',25)
leg = legend(p,...
    '$\bar{\eta}_+,\bar{\gamma}_-$',...
    '$\bar{\eta}_+,\bar{\gamma}_0$',...
    '$\bar{\eta}_+,\bar{\gamma}_+$',...
    '$\bar{\eta}_0,\bar{\gamma}_-$',...
    '$\bar{\eta}_0,\bar{\gamma}_0$',...
    '$\bar{\eta}_0,\bar{\gamma}_+$',...
    '$\bar{\eta}_-,\bar{\gamma}_-$',...
    '$\bar{\eta}_-,\bar{\gamma}_0$',...
    '$\bar{\eta}_-,\bar{\gamma}_+$');
fcn_scrpt_prepare_graph_science_std(fig, gca, p, leg, [], 18/2, 3, 1)
axis square
leg.Location = 'northeast';
leg.Orientation = 'horizontal';
leg.Interpreter = 'latex';
leg.Box = 'on';
ylim([1 12800])
xlim([2,7])
% fig = gcf;           % generate a figure
% tightfig(fig);
box on
set(gca, 'YScale', 'log')
pause(1)

SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','total_episodes_per_n_robots_v1.png'),'Resolution',600)
    close(gcf);
end 
%% Find the "sweet spot" for the gamma value to dominate learning

complexities = c_jk_cl_dist_episodes_ParamSweep;

% complexities = {c_jk_iso_parameters.episodes,c_jk_il_parameters.episodes,c_jk_til_parameters.episodes,c_jk_cl_parameters.episodes,c_jk_cl_dist_parameters.episodes};

% cmap = rand(size(complexities{1},2),3);
% cmap = flip(distinguishable_colors(size(complexities{1},2),{'w','k'}),1);

numColors = size(complexities{1},2);
% Choose a colormap (e.g., 'parula', 'jet', 'hot', 'cool', etc.)
% cmapName = 'autumn';

% Generate the colormap
% cmap = flipud(colormap(cmapName));
cmap = (colormap('turbo'));

% Create indices to pick colors evenly from the colormap
indices = linspace(1, size(cmap, 1), numColors);

% Interpolate the colormap to get the desired number of colors
cmap = interp1(1:size(cmap, 1), cmap, indices);







clc
close all
fig = figure('color','w');


upperBound  =  12800*ones(6,1);
lowerBound= parameters.totalSkills./[4,8,16,32,64,128]'.*parameters.fundamentalComplexity;
 
% upperBound = parameters.totalSkills./[4,8,16,32,64,128]'.*parameters.fundamentalComplexity;
% lowerBound = upperBound - 300;
patch([x fliplr(x)], [lowerBound'  fliplr(upperBound')], [0.5  0.5  0.5],'FaceAlpha',0.25,'EdgeColor','w');
p = zeros(1,size(complexities{1},2));
hold on
for index = 1:size(complexities{1},2)
    % if 0 %index<numel(complexities)
    %     the_mean   = ceil(mean(complexities{index},3));
    %     the_std    = ceil(std(complexities{index},[],3));
    %     uppeBound  = the_mean(:,end) + the_std(:,end);
    %     lowerBound = the_mean(:,end) - the_std(:,end);
    %     x = 1:7;
    %     patch([x fliplr(x)], [lowerBound'  fliplr(uppeBound')], [0.6  0.7  0.8],'FaceColor',cmap(index,:),'FaceAlpha',0.25,'EdgeColor','w');       
    % else
    %     the_mean   = ceil(mean(complexities{index},2));
    %     the_std    = ceil(std(complexities{index},[],2));
    %     uppeBound  = the_mean + the_std;
    %     lowerBound = the_mean - the_std;
    %     x = 2:7;
    %     patch([x fliplr(x)], [lowerBound'  fliplr(uppeBound')], [0.6  0.7  0.8],'FaceColor',cmap(index,:),'FaceAlpha',0.25,'EdgeColor','w');
    % end
    % % p = [p, plot(x, the_mean, 'o-', 'LineWidth', 1,'Color',cmap(index,:),'MarkerFaceColor',cmap(index,:))];        
    % % p(index) = plot(x, the_mean(:,end), 'o-', 'LineWidth', 1,'Color',cmap(index,:),'MarkerFaceColor',cmap(index,:));        
    
    x = 2:7;
    % tmp = plot(x, complexities{1}(:,index),'-','LineWidth', 1,'Color',cmap(index,:),'Marker','o','MarkerFaceColor',cmap(index,:));
    tmp = plot(x, complexities{1}(:,index),'--','LineWidth', 0.5,'Color',cmap(index,:));
    p(index) = tmp;
    clear tmp
end
% plot(5*ones(size(1E1:100:1E5)),1E0:100:1E5,'k--','LineWidth',3)
xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
xlabel('Number of robots','FontSize',25)
ylabel('Complexity (episodes for all skills)','FontSize',25)
title('$\bar{\eta} =-0.1 $','FontSize',25,'Interpreter','latex')
% leg = legend(p,...
%     '$\bar{\eta}_+,\bar{\gamma}_-$',...
%     '$\bar{\eta}_+,\bar{\gamma}_0$',...
%     '$\bar{\eta}_+,\bar{\gamma}_+$',...
%     '$\bar{\eta}_0,\bar{\gamma}_-$',...
%     '$\bar{\eta}_0,\bar{\gamma}_0$',...
%     '$\bar{\eta}_0,\bar{\gamma}_+$',...
%     '$\bar{\eta}_-,\bar{\gamma}_-$',...
%     '$\bar{\eta}_-,\bar{\gamma}_0$',...
%     '$\bar{\eta}_-,\bar{\gamma}_+$');
aux =round(linspace(-0.2,1,size(complexities{1},2)),2);
leg    = legend(p,arrayfun(@(i) num2str(aux(i)),1:size(complexities{1},2),'UniformOutput',false));
fcn_scrpt_prepare_graph_science_std(fig, gca, p, leg, [], 18/2, 3, 1)
% axis square
% leg.Location = 'northeast';
% leg.Interpreter = 'latex';
% leg.Box = 'on';
% leg.Interpreter = 'latex';
% leg.Box = 'on';
% ylim([1 12800])
% % fig = gcf;           % generate a figure
% % tightfig(fig);
% box on
% set(gca, 'YScale', 'log')
% grid off
% pause(1)
% leg.FontSize = 10;



sweetPoints = cell2mat(arrayfun(@(i) find(learnedSkillsRate(i,:)==100,1,'first'),1:6,'UniformOutput',false));

sweetPoints = [NaN sweetPoints];

% scatter([1:6],sweetPoints,100,'r','filled','hexagram')
ax = gca;

% colormap(ax,'hot')
bla = flip(ax.Children(1:end-1));
for i=1:numel(sweetPoints)
    if ~isnan(sweetPoints(i))
        % ax.Children(sweetPoints(i)).Color = [1 0 0];
        % theLine = 51 - (sweetPoints(i)+0);
        % ax.Children(theLine).LineWidth = 4;
        % scatter(ax.Children(theLine).XData(i), ...
        %         ax.Children(theLine).YData(i),300,'r','filled','hexagram')
        theLine = sweetPoints(i);
        bla(theLine).LineStyle = '-';
        bla(theLine).LineWidth = 4;
        bla(theLine).Color = cmap(theLine,:);

        scatter(bla(theLine).XData(i), ...
                bla(theLine).YData(i),100,cmap(theLine,:),'filled','o','MarkerEdgeColor','k')        
    end
end

axis square
leg.Location = 'northeast';
leg.Interpreter = 'latex';
leg.Box = 'on';
leg.Interpreter = 'latex';
leg.Box = 'on';
ylim([1 12800])
% fig = gcf;           % generate a figure
% tightfig(fig);
box on
set(gca, 'YScale', 'log')
grid off
pause(1)
leg.FontSize = 10;
ax.XGrid = 'on'; 
leg.String =  leg.String(1:50);
SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','total_episodes_for_all_skills_eta_n0.1.png'),'Resolution',600)
    close(gcf);
end  

%%
% Define the number of colors you want
numColors = 10;

% Choose a colormap (e.g., 'parula', 'jet', 'hot', 'cool', etc.)
cmapName = 'parula';

% Generate the colormap
cmap = colormap(cmapName);

% Create indices to pick colors evenly from the colormap
indices = linspace(1, size(cmap, 1), numColors);

% Interpolate the colormap to get the desired number of colors
selectedColors = interp1(1:size(cmap, 1), cmap, indices);

% Display the colors
figure;
hold on;
for i = 1:numColors
    plot(i, 1, 'o', 'MarkerSize', 10, 'MarkerFaceColor', selectedColors(i, :), 'MarkerEdgeColor', 'none');
end
%%

function [c_jk_cl_dist_episodes, learnedSkillsStorage] = runCollectiveLearningDistributed(parameters, etaValue)

% f = @(eta, N_zeta) eta.*N_zeta+1;
    g = @(delta, N_zeta) exp(-delta*N_zeta);
    % Dynamics of COLLECTIVE LEARNING (distributed) 
    % * NOTE: EQUAL number of robots per cluster
    clc
    close all
    % rng('default')
    warning("<<EQUAL>> number of robots per cluster")
    pause(2)
    % clear c_jk_cl_dist_episodes
    parameterSweepCases       = 50;
    theGammas                 = linspace(-0.2,1,parameterSweepCases);
    learnedSkillsStorage      = zeros(6,9);
    parameters.enableSharing  = 1;
    parameters.cl_distributed = 1;
    for scenario = 1:parameterSweepCases
        close all
        % switch scenario
        %     case 1
        %         parameters.eta_0   = 0.1;
        %         parameters.gamma_0 =-0.2;
        %     case 2
        %         parameters.eta_0   = 0.1;
        %         parameters.gamma_0 = 0;            
        %     case 3
        %         parameters.eta_0   = 0.1;
        %         parameters.gamma_0 = 0.2;  
        % 
        %     case 4
        %         parameters.eta_0   = 0;
        %         parameters.gamma_0 =-0.25;
        %     case 5
        %         parameters.eta_0   = 0;
        %         parameters.gamma_0 = 0;            
        %     case 6
        %         parameters.eta_0   = 0;
        %         parameters.gamma_0 = 0.2;                  
        % 
        %     case 7
        %         parameters.eta_0   = -0.1;
        %         parameters.gamma_0 = -0.2;
        %     case 8
        %         parameters.eta_0   =-0.1;
        %         parameters.gamma_0 = 0;
        %     case 9
        %         parameters.eta_0   =-0.1;
        %         parameters.gamma_0 = 0.2;  
        % end
        parameters.eta_0   = etaValue;
        parameters.gamma_0 = theGammas(scenario); 
    
        robotBatchIndex = 1;
        % Loop over robots
        % * NOTE: index starts in 4 to have at least one robot per cluster
        for numberOfRobots = [4,8,16,32,64,128]          
            fig = figure('color','w');
            % set(fig, 'WindowStyle', 'Docked');
            c_jk_cl_dist  = zeros(parameters.totalSkills/numberOfRobots, 1);%zeros(N_Z, N_K);
            
            % New set of skills, new learning rates
            Alpha = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));
            learnedSkills = 0;
            for skillsBatch = 1:(parameters.totalSkills/numberOfRobots)
                % if (skillsBatch==1)
                %     learnedSkills = 0;
                % else
                %     learnedSkills = learnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
                % end                
                % Beta adapts at every cycle
                % NOTE: From each cluster, only a fraction of knowlede can be transfered                
                % beta_k = numberOfRobots*learnedSkills/parameters.totalSkills;
                beta_k = learnedSkills/parameters.totalSkills;
                j      = skillsBatch;%learnedSkills + 1;
                fprintf('Gamma scenario: %1i |Robots: %3i | Skills batch: %2i | Cluster knowledge: %1.3f \n',scenario,numberOfRobots, skillsBatch, beta_k)
                % initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, numberOfRobots*learnedSkills).*ones(numberOfRobots,1);
                initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
                
                % Run remaining knowledge dynamics
                % remainingKnowledge = transpose( ...
                %         ode4_sat(@(n,sigmaBar) ...
                %             fcn_collectiveKnowledgeSharingDynamicsEpisodic(...
                %             numberOfRobots*learnedSkills,...
                %             sigmaBar, ...
                %             numberOfRobots, ...
                %             Alpha, ...
                %             beta_k, ...
                %             parameters), parameters.episodes, initialRemainingKnowedge)); 
                remainingKnowledge = transpose( ...
                        ode4_sat(@(n,sigmaBar) ...
                            fcn_collectiveKnowledgeSharingDynamicsEpisodic_mex(...
                            learnedSkills,...
                            sigmaBar, ...
                            numberOfRobots, ...
                            Alpha, ...
                            beta_k, ...
                            parameters), parameters.episodes, initialRemainingKnowedge));       

                if numberOfRobots == 8
                    test = 1
                end
                learnedSkills = learnedSkills + sum(remainingKnowledge(:,end)<parameters.knowledgeLowerBound);
                if any(remainingKnowledge(:,end)==1)
                    warning('There are unlearned skills')
                end                
        
                c_jk_cl_dist(j) = ...
                    ceil( ...
                        mean( ...
                            parameters.episodes( ...
                                cell2mat( ...
                                    arrayfun( ...
                                        @(i) min( ...
                                            find( ...
                                                remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:numberOfRobots,'UniformOutput',false)))));
                
                if isnan(c_jk_cl_dist(j))
                    % disp('here')
                    c_jk_cl_dist(j) = parameters.fundamentalComplexity;
                end


                loglog(parameters.episodes,mean(remainingKnowledge,1),'LineStyle','-','color',rand(1,3),'LineWidth',1)
                xlim([0 parameters.episodes(end)])
                ylim([0.008 1E0])
                xticks([10^0 10^1 10^2])
                xticklabels({'1','10', '$c_0$'})
                yticks([1E-3 1E-2 1E-1 1E0])
                yticklabels({'','$\epsilon$', '', '1'})
                set(gca,'TickLabelInterpreter','latex')
                hold on
            end
        
            p = area(parameters.episodes,  parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'FaceColor',[0 1 0],'FaceAlpha',0.25,'EdgeColor','w');
            plot(parameters.episodes,parameters.knowledgeLowerBound*ones(numel(parameters.episodes),1),'k:','LineWidth',2);
            % title(['$\mathcal{Z}_{',num2str(k),'}$'],'FontSize',25,'Interpreter','latex')
            xlabel('Episodes','FontSize',25)
            ylabel(['$\bar{\boldmath{\sigma}}^{(\mathrm{CLd})}_{j,k}$'],'FontSize',25,'Interpreter','latex')
            fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 18/2, 3, 1)
            pause(1)
            tightfig(fig);
            c_jk_cl_dist_episodes(robotBatchIndex, scenario) = sum(c_jk_cl_dist);
            learnedSkillsStorage(robotBatchIndex, scenario) = learnedSkills

            robotBatchIndex                        = robotBatchIndex + 1; 
        end
    end
end
%%