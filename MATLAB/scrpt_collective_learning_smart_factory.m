clearvars
clc
close all
cd(fileparts(matlab.desktop.editor.getActiveFilename));

%%
clc
theCollective = RobotCollective(numberOfRobots = 12, repeatingSkills = false);

% Set flags
theCollective.RAND_COMM_INTERRUPT         = true; 
theCollective.ENABLE_INCREMENTAL_LEARNING = 1; 
theCollective.ENABLE_TRANSFER_LEARNING    = 1; 
theCollective.ENABLE_COLLECTIVE_LEARNING  = 1; 
% theCollective.REPEATING_SKILLS            = 0;

% Define simulation episodes
theCollective.Episodes            = 1:0.1:(2*theCollective.FundamentalComplexity);

% Define agent(s) learning factors
% theCollective.Eta_0       = +0.01;
% theCollective.Gamma_0     = +0.01;
theCollective.Eta_0       = -0.1;
theCollective.Gamma_0     = +0.2;

% Max. nummber of skills per product
% theCollective.MaxNumberOfSkillsPerProduct = 8;
theCollective.MaxNumberOfSkillsPerProduct = theCollective.NumberOfRobots;
%% Simulate
clc
close all
[totalLearningEpisodes, learnedSkills, clusterKnowledge, results] = ...
    theCollective.simulateDistributedCollectiveKnowledgeDynamics(maxNumberOfProducts = 1000);


%%
close all
clc


robotArray = [8,16,32,64,128];

learningCases = [zeros(1,3);  % Isolated
                 1 0 0;       % Incremental
                 1 1 0;       % Trasnfer + Incremental
                 1 1 1];      % Collective
learningCasesLabels = {'IsL','IL','TIL','DCL'};
cl_results = cell(size(learningCases,1),numel(robotArray));


%%
clc
for learningCaseIndex = 1%:size(learningCases,1)
    for robotArrayIndex = 1:numel(robotArray)
        cprintf('green',sprintf('[INFO] %s with collective of size %2i...\n', learningCasesLabels{learningCaseIndex},robotArray(robotArrayIndex)))
        pause(3)
        
        theCollective = RobotCollective(numberOfRobots = robotArray(robotArrayIndex));
        
        % Set flags
        theCollective.RAND_COMM_INTERRUPT         = true; 
        theCollective.ENABLE_INCREMENTAL_LEARNING = learningCases(learningCaseIndex,1); 
        theCollective.ENABLE_TRANSFER_LEARNING    = learningCases(learningCaseIndex,2); 
        theCollective.ENABLE_COLLECTIVE_LEARNING  = learningCases(learningCaseIndex,3); 
        theCollective.REPEATING_SKILLS            = true;
        
        % Define simulation episodes
        theCollective.Episodes            = 1:.1:2*theCollective.FundamentalComplexity;
        
        % Define agent(s) learning factors
        theCollective.Eta_0       = +0.01;
        theCollective.Gamma_0     = +0.01;
        
        % Max. nummber of skills per product
        theCollective.MaxNumberOfSkillsPerProduct = 8;
    
    
    
        [~, ~, ~, cl_results{learningCaseIndex, robotArrayIndex}] = ...
        theCollective.simulateDistributedCollectiveKnowledgeDynamics(maxNumberOfProducts = 1000);
    end
end
%%
clc
close all
figure('Color','w')
xlim([1, 500])
ylim([1, 100])
set(gca,'XScale','linear')
set(gca,'YScale','log')
xlabel('Items','FontSize',11)
ylabel('Complexity','FontSize',11)
hold on
for learningCaseIndex = 1:size(learningCases,1)
    for robotArrayIndex = 1:numel(robotArray)
        loglog(cl_results{learningCaseIndex,robotArrayIndex}.c_jk_cl_dist_episodes)
    end
end

%%
clc
close all
figure('Color','w')
xlim([1, 500])
ylim([1, 100])
set(gca,'XScale','linear')
set(gca,'YScale','linear')
xlabel('Items','FontSize',11)
ylabel('Complexity (episodes per item)','FontSize',11)
hold on

cmap = colormap('lines');
% Create indices to pick colors evenly from the colormap
colorIndices = linspace(1, size(cmap, 1), size(learningCases,1));
% Interpolate the colormap to get the desired number of colors
selectedColors = interp1(1:size(cmap, 1), cmap, colorIndices);

for learningCaseIndex = 1:size(learningCases,1)
    remainingKnowledge = cell2mat(arrayfun(@(robotArrayIndex) cl_results{learningCaseIndex,robotArrayIndex}.c_jk_cl_dist_episodes,1:5,'UniformOutput',false))';
    
    meanCurve  = max(1,mean(remainingKnowledge,1));
    stdCurve   = std(remainingKnowledge, 0, 1);
    upperBound = meanCurve + stdCurve;
    lowerBound = max(1,meanCurve - stdCurve);
    
    % Create the plot
    % Plot the shaded region for standard deviation
    
    patch(gca, [1:500, fliplr(1:500)], [upperBound, fliplr(lowerBound)], selectedColors(learningCaseIndex,:), ...
    'EdgeColor', 'none', 'FaceAlpha', 0.3);
    
    % Plot the mean curve
    plot(gca, 1:500, meanCurve, 'b-', 'LineWidth', 1,'Color',selectedColors(learningCaseIndex,:));

end
p1 = plot(NaN,'Color',selectedColors(1,:));
p2 = plot(NaN,'Color',selectedColors(2,:));
p3 = plot(NaN,'Color',selectedColors(3,:));
p4 = plot(NaN,'Color',selectedColors(4,:));
legend([p1 p2 p3 p4], learningCasesLabels)


%%
close all
clc
figure('Color', 'w')
plot(1:theCollective.MaxNumberOfProducts, theCollective.NumberOfNewSkills,'r')
hold on
plot(1:theCollective.MaxNumberOfProducts, theCollective.NumberOfSeenSkills,'b')
plot(1:theCollective.MaxNumberOfProducts, cumsum(theCollective.NumberOfNewSkills),'m--')
plot(1:theCollective.MaxNumberOfProducts, theCollective.SkillsInCluster,'k')
legend('No. new skills','No. learned skills','Skills per cluster')
xlabel('No. of products')
ylabel('No. of skills learned')
axis square
%%
clc
clearvars

LOAD_RESULTS = 1;
if LOAD_RESULTS == 1
    load('simulation_cases_distributed_cl_eta_and_gamma.mat')
end

%%
close all
clc
rng('default')
clear c_jk_cl_dist_episodes_ParamSweep learnedSkillsStorage
for iter = 1:1
    close all
    [c_jk_cl_dist_episodes_ParamSweep{iter}, learnedSkillsStorage{iter}] = ...
        runCollectiveLearningDistributedParameterSweep(parameters, 8)
end

%%

% totalGammaCandidates
% totalGammaCandidates
clc
close all
Z = c_jk_cl_dist_episodes_ParamSweep{iter};
Z = (100*(learnedSkillsStorage{iter}./parameters.totalSkills));
aux = linspace(-0.2,1,10);
[X,Y] = meshgrid(aux,aux);
x = X(:);
y = Y(:);
z = Z(:);

% Define a grid of query points and interpolate the scattered data over the grid.

[xq,yq] = meshgrid(-0.2:0.01:1,-0.2:0.01:1);
zq = griddata(x,y,z,xq,yq,'linear');

figure(Color='w')
imagesc(zq)
axis square
colorbar

aux =round(-0.2:0.01:1,2);
tickText    = arrayfun(@(i) num2str(aux(i)),1:10:121,'UniformOutput',false);
xticks([1:10:121])
yticks([1:10:121])
xticklabels(tickText)
yticklabels(tickText)
xlabel('$\bar{\gamma}$','Interpreter','latex','FontSize',20)
ylabel('$\bar{\eta}$','Interpreter','latex','FontSize',20)
title('Success rate for $m=8$ robots','Interpreter','latex','FontSize',15)
axis square

%%
clc
close all
Z = c_jk_cl_dist_episodes_ParamSweep{iter};
% Z = (100*(learnedSkillsStorage{iter}./parameters.totalSkills));
aux = linspace(-0.2,1,10);
[X,Y] = meshgrid(aux,aux);
x = X(:);
y = Y(:);
z = Z(:);

% Define a grid of query points and interpolate the scattered data over the grid.

[xq,yq] = meshgrid(-0.2:0.01:1,-0.2:0.01:1);
zq = griddata(x,y,z,xq,yq,'linear');

figure(Color='w')
imagesc(zq)
axis square
colorbar

aux =round(-0.2:0.01:1,2);
tickText    = arrayfun(@(i) num2str(aux(i)),1:10:121,'UniformOutput',false);
xticks([1:10:121])
yticks([1:10:121])
xticklabels(tickText)
yticklabels(tickText)
xlabel('$\bar{\gamma}$','Interpreter','latex','FontSize',15)
ylabel('$\bar{\eta}$','Interpreter','latex','FontSize',15)
title('Total complexity $m=8$ robots','Interpreter','latex','FontSize',15)
axis square

%%

parameters.knowledgeLowerBound   = 0.01;
parameters.fundamentalComplexity = 100;
parameters.episodes              = 0:0.1:500;
parameters.totalSkills           = 512;
parameters.totalSkillClusters    = 4;
parameters.skillsPerCluster      = 128;
parameters.alpha_min             = 0.0461;
parameters.alpha_max             = 0.0691;
parameters.delta                 = 0.0360;
parameters.eta_0                 =-0.1000;
parameters.eta_std               = 0.1;
parameters.gamma_0               = 10;
parameters.gamma_std             = 0.1;
parameters.maxNumberOfRobots     = 128;
parameters.totalSimulationScenarios = 5;
parameters.cl_distributed        = 1;
parameters.enableSharing         = 1;
parameters.randCommInterrupt     = 1;
parameters.enableTransfer        = 1;

%%
close all 
clc
skillIndices  = randperm(parameters.totalSkills,parameters.totalSkills);
skillClusters = reshape(skillIndices,parameters.skillsPerCluster,parameters.totalSkillClusters);


for k = 1:parameters.totalSkillClusters
    sample_indices = randperm(parameters.skillsPerCluster,10);
    skillClusters(sample_indices,k);
end
parameters.skillClusters = skillClusters;
%%
clc
productSkills = randi(parameters.totalSkills,1,numberOfRobots);

% intersect(skillClusters(:,1),productSkills) 
% intersect(skillClusters(:,2),productSkills) 
% intersect(skillClusters(:,3),productSkills) 
% intersect(skillClusters(:,4),productSkills)

sum(ismember(skillClusters(:,1),productSkills))
sum(ismember(skillClusters(:,2),productSkills))
sum(ismember(skillClusters(:,3),productSkills))
sum(ismember(skillClusters(:,4),productSkills))
%%
clc
close all

seenSkills = [];
% numberOfSeenSkills = 0;
% for iter=1:5
% for iter=1:1000

clear numberOfNewSkills numberOfSeenSkills  skillsInCluster1 skillsInCluster2  skillsInCluster3 skillsInCluster4
product = 1;
while numel(seenSkills) < parameters.totalSkills
    productSkills = randi(parameters.totalSkills,1,32);
    % numberOfSeenSkills  = numberOfSeenSkills + sum(~ismember(productSkills,seenSkills));
    % seenSkills = [seenSkills, productSkills(~ismember(productSkills,seenSkills))];
    numberOfNewSkills(product) = numel(unique([seenSkills, productSkills])) - numel(seenSkills);
    seenSkills = unique([seenSkills, productSkills]);
    numberOfSeenSkills(product) = numel(seenSkills);



skillsInCluster1(product) = numel(intersect(seenSkills,skillClusters(:,1)))
skillsInCluster2(product) = numel(intersect(seenSkills,skillClusters(:,2)))
skillsInCluster3(product) = numel(intersect(seenSkills,skillClusters(:,3)))
skillsInCluster4(product) = numel(intersect(seenSkills,skillClusters(:,4)))

    % if numberOfSeenSkills ~= numel(seenSkillsAux)
    %     disp('HERE')
    % else
    %    seenSkills = seenSkillsAux; 
    % end
    % if numel(seenSkills) == parameters.totalSkills
    %     break
    % end
    product = product + 1;
% numel(seenSkills)
% numel(intersect(1:parameters.totalSkills,productSkills))
% storedSklls = sum(ismember(productSkills,1:parameters.totalSkills));
end
figure('Color', 'w')
plot(numberOfNewSkills,'r')
hold on
plot(numberOfSeenSkills,'b')
plot(skillsInCluster1,'k')
plot(skillsInCluster2,'k')
plot(skillsInCluster3,'k')
plot(skillsInCluster4,'k')
legend('# new skills','# seen skills')
xlabel('No. of products')
ylabel('No. of skills learned')
axis square

%%

B = unique(seenSkills); % which will give you the unique elements of A in array B
Ncount = histc(seenSkills, B); % this willgive the number of occurences of each unique element

%%
clc
close all

parameters.randCommInterrupt   = 1;
parameters.enableTransfer      = 1;
parameters.enableSharing       = 1;
parameters.episodes            = 1:.1:2*parameters.fundamentalComplexity;
parameters.repeatingSkillsFlag = 1;


% eta_mean       = -0.1;
% gamma_mean     = +0.4;

eta_mean       = +0.01;
gamma_mean     = +0.10;

parameters.numberOfRobots           = 8;
parameters.numberOfRobotsPerCluster = repmat(parameters.numberOfRobots/parameters.totalSkillClusters,parameters.totalSkillClusters,1);

% parameters.numberOfRobotsPerCluster = [2,4,8,2];

parameters.maxNumberOfSkillsPerProduct = 8;

% b = @(clusterTrasferrableKnowledgeFraction,parameters.numberOfRobots) (clusterTrasferrableKnowledgeFraction-1).*((parameters.numberOfRobots/parameters.totalSkillClusters-1) + (parameters.totalSkillClusters-1)*parameters.numberOfRobots/parameters.totalSkillClusters.*clusterTrasferrableKnowledgeFraction);
% a = mean([parameters.alpha_min,parameters.alpha_max]);
% gamma_mean     = (a/b(1 - numberOfRobots/parameters.totalSkills,numberOfRobots))*(parameters.eta_0*(parameters.totalSkills-numberOfRobots) + 1);

% [totalComplexity, learnedSkills, clusterKnowledge] = ...
%     simulateDistributedCollectiveKnowledgeDynamicsGammaSweep(eta_mean, gamma_mean, parameters, numberOfRobots);
[totalLearningEpisodes, learnedSkills, clusterKnowledge, results] = ...
    simulateDistributedCollectiveKnowledgeDynamics(eta_0               = eta_mean, ...
                                                   gamma_0             = gamma_mean, ...
                                                   parameters          = parameters, ...
                                                   maxNumberOfProducts = 1000);


% title(gca,['$\bar{\eta}=',num2str(eta_mean),'~|~\bar{\gamma} \in [-0.2, 10.4]$'],'Interpreter','latex','FontSize',15)
% title(gca,['$m =',num2str(numberOfRobots),'~|~\bar{\eta}=',num2str(eta_mean),'~|~\bar{\gamma} =', num2str(gamma_mean),'$'],'Interpreter','latex','FontSize',11)
%%

clc
close all

parameters.randCommInterrupt = 1;
parameters.enableTransfer    = 1;
parameters.enableSharing     = 1;
parameters.episodes          = 1:1:2*parameters.fundamentalComplexity;

% eta_mean       = -0.1;
% gamma_mean     = +0.4;

eta_mean       = +0.1;
gamma_mean     = +0.1;

numberOfRobots = 8;
b = @(clusterTrasferrableKnowledgeFraction,numberOfRobots) (clusterTrasferrableKnowledgeFraction-1).*((numberOfRobots/parameters.totalSkillClusters-1) + (parameters.totalSkillClusters-1)*numberOfRobots/parameters.totalSkillClusters.*clusterTrasferrableKnowledgeFraction);
a = mean([parameters.alpha_min,parameters.alpha_max]);
% gamma_mean     = (a/b(1 - numberOfRobots/parameters.totalSkills,numberOfRobots))*(parameters.eta_0*(parameters.totalSkills-numberOfRobots) + 1);

% [totalComplexity, learnedSkills, clusterKnowledge] = ...
%     simulateDistributedCollectiveKnowledgeDynamicsGammaSweep(eta_mean, gamma_mean, parameters, numberOfRobots);
totalLearningEpisodes ={};
learnedSkills   ={};
clusterKnowledge = {};
for m = [4,8,16,32,64,128]
    numberOfRobots = m;   
    [totalLearningEpisodes{m}, learnedSkills{m}, clusterKnowledge{m}] = ...
        simulateDistributedCollectiveKnowledgeDynamics(eta_mean, gamma_mean, parameters, numberOfRobots);
    pause
end
%%
clc
close all
figure('Color','w')
for m = [4,8,16]
    plot(totalLearningEpisodes{m})
    hold on
end
%%
clc
close all
[skillsGrid, betaGrid]=meshgrid(0:1:parameters.totalSkills-numberOfRobots,0:0.01:(1-numberOfRobots/parameters.totalSkills));

gammaBoundary = (a./b(betaGrid(:),numberOfRobots)).*(eta_mean.*skillsGrid(:) + 1);
gammaBoundary = reshape(gammaBoundary,size(skillsGrid,1),size(skillsGrid,2));


for gammaIter = 0.1:0.1:10
close all
clc
figure('Color','w');
surf(skillsGrid, betaGrid, gammaBoundary,EdgeColor = 'none',FaceAlpha=1,FaceColor='interp');
hold on
surf(skillsGrid, betaGrid, gammaIter*ones(size(gammaBoundary)),EdgeColor = 'none',FaceAlpha=1,FaceColor='r')
% contour(skillsGrid, betaGrid, gammaBoundary)
xlabel('$\kappa$',Interpreter='latex',FontSize=15)
ylabel('$\beta$',Interpreter='latex',FontSize=15)
zlabel('$\gamma_\mathrm{min}$',Interpreter='latex',FontSize=15)
title(['$\gamma=',num2str(gammaIter),'$'],Interpreter='latex',FontSize=15)
axis tight;                              % Adjust to fit the data
axis vis3d;                              % Lock aspect ratio for proper visualization

view(0,90)
drawnow
exportgraphics(gcf,'stability_region.gif','Append',true,'Resolution',300)
pause(1)
end
%% Plot the gridded data as a mesh and the scattered data as dots.

mesh(xq,yq,zq,'FaceColor','interp')
xlabel('$\bar{\gamma}$','Interpreter','latex','FontSize',15)
ylabel('$\bar{\eta}$','Interpreter','latex','FontSize',15)
axis square
% hold on
% plot3(x,y,v,"o")
% xlim([-2.7 2.7])
% ylim([-2.7 2.7])
%%

clc
close all
% figure('color','w')
clear learnedSkillsRate
for index = 1:size(c_jk_cl_dist_episodes_ParamSweep{1},2)
    the_learned_skills = cell2mat(arrayfun(@(i) learnedSkillsStorage{i}(:,index),1:10,'UniformOutput',false));
    learnedSkillsRate(:,index) = floor(100*(ceil(mean(the_learned_skills,2))./parameters.totalSkills));
    % b =plot(x,the_mean,'color',cmap(index,:),'Marker',allMarkers(index));
    % figure
    % b =bar(x,the_mean);
    % b.FaceColor = 'flat';
    % hold on
    % the_std  = ceil(std(the_data,[],2))
    % uppeBound  = the_mean + the_std;
    % lowerBound = the_mean - the_std;    
end
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

for index = 1:size(c_jk_cl_dist_episodes_ParamSweep{1},2)
    nexttile
    upperBound  =  12800*ones(6,1);
    lowerBound= parameters.totalSkills./[4,8,16,32,64,128]'.*parameters.fundamentalComplexity;
    patch([x fliplr(x)], [lowerBound'  fliplr(upperBound')], [0.5  0.5  0.5],'FaceAlpha',0.25,'EdgeColor','w');
    hold on
    the_learned_skills = cell2mat(arrayfun(@(i) c_jk_cl_dist_episodes_ParamSweep{i}(:,index),1:10,'UniformOutput',false));
    the_mean   = ceil(mean(the_learned_skills,2));
    the_std    = ceil(std(the_learned_skills,[],2));
    upperBound = the_mean + the_std;
    lowerBound = the_mean - the_std;    
    x          = 2:7;
    patch([x fliplr(x)], [lowerBound'  fliplr(upperBound')], [0  0  0],'FaceColor',cmap(index,:),'FaceAlpha',0.1,'EdgeColor','w');
    p(index) = plot(x, the_mean,'LineStyle','-', 'LineWidth', 3,'Color','k');%,cmap(index,:));
    aux = scatter(x, the_mean, 100, learnedSkillsRate(:,index), 'filled', 'MarkerEdgeColor', cmap(index,:));
    
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

    fcn_scrpt_prepare_graph_science_std(gcf, gca, p(index), leg, [], 18, 1, 1)
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
    % aux = scatter(x, the_mean,50,mean_learned_skills(:,index));
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
%%

complexities = {c_jk_cl_dist_episodes_ParamSweep};

% complexities = {c_jk_iso_parameters.episodes,c_jk_il_parameters.episodes,c_jk_til_parameters.episodes,c_jk_cl_parameters.episodes,c_jk_cl_dist_parameters.episodes};

cmap = rand(size(complexities{1},2),3);
clc
close all
fig = figure('color','w');
p = NaN(1,size(complexities{1},2));
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
    p(index) = plot(x, complexities{1}(:,index), 'LineWidth', 1,'Color',cmap(index,:),'MarkerFaceColor',cmap(index,:));        
end
% plot(5*ones(size(1E1:100:1E5)),1E0:100:1E5,'k--','LineWidth',3)
xticks([1:7])
xticklabels({'2','4','8','16','32','64','128'})
xlabel('Number of robots','FontSize',25)
ylabel('Complexity (episodes per skill)','FontSize',25)
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
leg.Interpreter = 'latex';
leg.Box = 'on';
% fig = gcf;           % generate a figure
% tightfig(fig);
box on
set(gca, 'YScale', 'log')
pause(1)

SAVE_FIG = 0;
if SAVE_FIG == 1
    exportgraphics(gcf, fullfile(fileparts(matlab.desktop.editor.getActiveFilename), ...
        'figures','total_parameters.episodes_per_n_robots_v1.png'),'Resolution',600)
    close(gcf);
end  
%%

function [c_jk_cl_dist_episodes, learnedSkillsStorage] = runCollectiveLearningDistributed(parameters)

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
    parameterSweepCases = 9;
    
    learnedSkillsStorage = zeros(6,9);
    parameters.enableSharing  = 1;
    parameters.cl_distributed = 1;
    for scenario = 1:parameterSweepCases
    close all
        switch scenario
            case 1
                parameters.eta_0   = 0.1;
                parameters.gamma_0 =-0.2;
            case 2
                parameters.eta_0   = 0.1;
                parameters.gamma_0 = 0;            
            case 3
                parameters.eta_0   = 0.1;
                parameters.gamma_0 = 0.2;  

            case 4
                parameters.eta_0   = 0;
                parameters.gamma_0 =-0.25;
            case 5
                parameters.eta_0   = 0;
                parameters.gamma_0 = 0;            
            case 6
                parameters.eta_0   = 0;
                parameters.gamma_0 = 0.2;                  

            case 7
                parameters.eta_0   = -0.1;
                parameters.gamma_0 = -0.2;
            case 8
                parameters.eta_0   =-0.1;
                parameters.gamma_0 = 0;
            case 9
                parameters.eta_0   =-0.1;
                parameters.gamma_0 = 0.2;  
        end
    
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
                fprintf('Scenario: %1i |Robots: %3i | Skills batch: %2i | Cluster knowledge: %1.3f \n',scenario,numberOfRobots, skillsBatch, beta_k)
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

%% Dynamics of DISTRIBUTED collective learning
% * NOTE: EQUAL number of robots per cluster (when possible)
function [c_jk_cl_dist_episodes, learnedSkillsStorage] = runCollectiveLearningDistributedParameterSweep(parameters, numberOfRobots)
    g = @(delta, N_zeta) exp(-delta*N_zeta);

    clc
    close all
    % rng('default')
    warning("<<EQUAL>> number of robots per cluster")
    pause(2)
    % clear c_jk_cl_dist_episodes
    totalGammaCandidates      = 10;
    totalEtaCandidates        = 10;
    theGammas                 = linspace(-0.2,1,totalGammaCandidates);
    theEtas                   = linspace(-0.2,1,totalEtaCandidates);
    learnedSkillsStorage      = zeros(numel(theGammas),numel(theEtas));
    parameters.enableSharing  = 1;
    parameters.cl_distributed = 1;
    for gammaScenario = 1:totalGammaCandidates
        close all
        % parameters.eta_0   =theEtas(etaScenario);%-0.1;
        parameters.gamma_0 =theGammas(gammaScenario); 
    
        robotBatchIndex = 1;
        % Loop over robots
        % * NOTE: index starts in 4 to have at least one robot per cluster
        % for numberOfRobots = [4,8,16,32,64,128]          
        for etaScenario = 1:totalEtaCandidates
            parameters.eta_0   =theEtas(etaScenario);%-0.1
            fig = figure('color','w');
            % Subindex jk means skill j cluster k
            % c_jk_cl_dist  = zeros(parameters.totalSkills/numberOfRobots, 1);
            if mod(parameters.totalSkills,numberOfRobots)==0
                numberOfSkillBatches = parameters.totalSkills/numberOfRobots;
                FLAG = 0;
            else
                numberOfSkillBatches = floor(parameters.totalSkills/numberOfRobots) + 1;
                FLAG = 1;
            end
            c_jk_cl_dist  = zeros(numberOfSkillBatches, 1);
            
            % New set of skills, new learning rates
            Alpha = diag(parameters.alpha_min + (parameters.alpha_max-parameters.alpha_min).*rand(numberOfRobots,1));
            learnedSkills = 0;
            for skillsBatch = 1:numberOfSkillBatches
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
                fprintf('Gamma scenario: %2i | Eta scenario: %2i | Skills batch: %2i | Cluster knowledge: %1.3f \n',gammaScenario, etaScenario, skillsBatch, beta_k)
                % initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, numberOfRobots*learnedSkills).*ones(numberOfRobots,1);
                
                if FLAG == 0
                    initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
                    remainingKnowledge = transpose( ...
                        ode4_sat(@(n,sigmaBar) ...
                            fcn_collectiveKnowledgeSharingDynamicsEpisodic_mex(...
                            learnedSkills,...
                            sigmaBar, ...
                            numberOfRobots, ...
                            Alpha, ...
                            beta_k, ...
                            parameters), parameters.episodes, initialRemainingKnowedge));       

                elseif (FLAG == 1) 
                    if (skillsBatch < numberOfSkillBatches)
                        initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(numberOfRobots,1);
                        remainingKnowledge = transpose( ...
                            ode4_sat(@(n,sigmaBar) ...
                                fcn_collectiveKnowledgeSharingDynamicsEpisodic_mex(...
                                learnedSkills,...
                                sigmaBar, ...
                                numberOfRobots, ...
                                Alpha, ...
                                beta_k, ...
                                parameters), parameters.episodes, initialRemainingKnowedge));       

                    elseif (skillsBatch == numberOfSkillBatches)
                        initialRemainingKnowedge = (1 - beta_k)*g(parameters.delta, learnedSkills).*ones(parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1),1);
                        remainingKnowledge = transpose( ...
                            ode4_sat(@(n,sigmaBar) ...
                                fcn_collectiveKnowledgeSharingDynamicsEpisodic_mex(...
                                learnedSkills,...
                                sigmaBar, ...
                                parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1), ...
                                Alpha(1:parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1),1:parameters.totalSkills-numberOfRobots*(numberOfSkillBatches-1)), ...
                                beta_k, ...
                                parameters), parameters.episodes, initialRemainingKnowedge));  
                    end
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
                                                remainingKnowledge(i,:)<parameters.knowledgeLowerBound)),1:size(remainingKnowledge,1),'UniformOutput',false)))));
                
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
            c_jk_cl_dist_episodes(etaScenario, gammaScenario) = sum(c_jk_cl_dist);
            learnedSkillsStorage(etaScenario, gammaScenario) = learnedSkills;

            % robotBatchIndex                        = robotBatchIndex + 1; 
        end
    end
end
%%