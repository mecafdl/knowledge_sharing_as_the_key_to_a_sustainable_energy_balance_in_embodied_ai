clearvars
clc
close all

mainPath = fileparts(matlab.desktop.editor.getActiveFilename);
cd(mainPath);
addpath(genpath(fullfile(mainPath,'supporting_files')));

%% ************************************************************************
% Run collective learning smart factory scenario
% *************************************************************************

try
    clc
    load("./paper_results/collective_learning_smart_factory_results.mat")
    cprintf('g', 'Successfully loaded file\n')
    RUN_CL = 0;
catch
    warning('File not found')
    RUN_CL = 1;
end
%%

RUN_CL = 1;
if RUN_CL == 1
    close all
    clc
    
    
    robotArray = [8,16,32,64,128];
    
    learningCases = [zeros(1,3);  % Isolated
                     1 0 0;       % Incremental
                     1 1 0;       % Trasnfer + Incremental
                     1 1 1];      % Collective
    learningCasesLabels = {'IsL','IL','TIL','CL'};
    
    numberOfScenarios = 50;
    cl_results = cell(size(learningCases,1), numel(robotArray), numberOfScenarios);
    
    clc
    
    seeds = randperm(numberOfScenarios,numberOfScenarios);
    % NOTE: Goal of the many scenarios is to find the mean values across
    %       simulations
    %for scenario = 1%1:numberOfScenarios
    for scenario = 1:numberOfScenarios
        theSeed = randi(100);
        % for learningCaseIndex = 4%1:size(learningCases,1)
        for learningCaseIndex = 1:size(learningCases,1)
            % for robotArrayIndex = 1%1:numel(robotArray)
            for robotArrayIndex = 1:numel(robotArray)
                close all
                cprintf('green',sprintf('[INFO] %s with collective of size %2i...\n', learningCasesLabels{learningCaseIndex},robotArray(robotArrayIndex)))
                pause(3)
                
                % Initialize a robot collective
                theCollective = RobotCollective(numberOfRobots = robotArray(robotArrayIndex), ...
                                                repeatingSkills = true, ...
                                                MaxNumberOfSkillsPerProduct = 8, ...
                                                maxNumberOfProducts = 500);
                
                % Set flags
                theCollective.RAND_COMM_INTERRUPT         = true; 
                theCollective.ENABLE_INCREMENTAL_LEARNING = learningCases(learningCaseIndex,1); 
                theCollective.ENABLE_TRANSFER_LEARNING    = learningCases(learningCaseIndex,2); 
                theCollective.ENABLE_COLLECTIVE_LEARNING  = learningCases(learningCaseIndex,3);
                
                % Define simulation episodes
                theCollective.Episodes            = 1:.1:2*theCollective.FundamentalComplexity;
                
                % Define agent(s) and collective learning factors
                theCollective.Eta_0       = +0.01;
                theCollective.Gamma_0     = +0.01;
                
                % theCollective.productSeed = seeds(scenario);
                theCollective.productSeed = theSeed;
                % Run the scenario
                [~, ~, ~, cl_results{learningCaseIndex, robotArrayIndex, scenario}] = ...
                theCollective.simulateDistributedCollectiveKnowledgeDynamics(maxNumberOfProducts = 1000);
            end
        end
    end
end
%% ************************************************************************
%  Create plot of the episodes per item for all learning paradigms
% *************************************************************************
clc

cmap = colormap('lines');
% Create indices to pick colors evenly from the colormap
colorIndices = linspace(1, size(cmap, 1), size(learningCases,1));
% Interpolate the colormap to get the desired number of colors
selectedColors = interp1(1:size(cmap, 1), cmap, colorIndices);

close all
fig = figure('Color','w');
ax = gca;

xlim([1, 500])
ylim([1, 100])
set(ax,'XScale','log')
set(ax,'YScale','log')
xlabel('Items','FontSize',11)
ylabel('Episodes / Item','FontSize',11)
axis square
hold on
p = [];
for learningCaseIndex = 1:size(learningCases,1)
    for robotArrayIndex = 1%:numel(robotArray)
        aux      = (cell2mat(arrayfun(@(k) (cl_results{learningCaseIndex,robotArrayIndex,k}.c_jk_cl_dist_episodes),1:50,'UniformOutput',false)));
        
        learningEpisodes      = aux;

        [~,mu,sigma] = zscore(learningEpisodes');
        meanCurve = mu;
        stdCurve  = sigma;
        
        % Define the upper and lower bounds of the shaded region
        upperBound = meanCurve + stdCurve;
        lowerBound = max(1,meanCurve - stdCurve);
        patch(ax, [1:500, fliplr(1:500)], [upperBound, fliplr(lowerBound)], selectedColors(learningCaseIndex,:), ...
            'EdgeColor', 'none', 'FaceAlpha', 0.1);
        if learningCaseIndex<4
            plot(ax, smooth(meanCurve,10), 'b-', 'LineWidth', 1,'Color',selectedColors(learningCaseIndex,:));
        else
            plot(ax, meanCurve, 'b-', 'LineWidth', 1,'Color',selectedColors(learningCaseIndex,:));
        end
        totalLearningEpisodes(learningCaseIndex) = sum(mu);


        % loglog(ax, mu + sigma,'k')
        % loglog(ax, mu - sigma,'k')
        % p = [p loglog(ax, mu,'-','Color',selectedColors(learningCaseIndex,:))];

        % p = [p loglog(ax, smooth(mu,10))];

        % learningEpisodes      = mean(aux,2);
        % totalLearningEpisodes(learningCaseIndex) = sum(learningEpisodes);
        % p = [p loglog(ax, smooth(learningEpisodes,10))];
    end
end
p1  = plot(NaN,'-','Color',selectedColors(1,:));
p2  = plot(NaN,'-','Color',selectedColors(2,:));
p3  = plot(NaN,'-','Color',selectedColors(3,:));
p4  = plot(NaN,'-','Color',selectedColors(4,:));
leg = legend([p1 p2 p3 p4], learningCasesLabels);
grid off
ax1 = gca;
fcn_scrpt_prepare_graph_science_std(fig, ax, p, leg, [], 6, 1, 1)
box on

%% ************************************************************************
% Bar plot of the energy consumed by each learning paradigm
% *************************************************************************

close all
fig = figure('Color','w');
    learningEnergy  = 105000*totalLearningEpisodes;

    learningEnergy(end) = learningEnergy(end) + theCollective.NumberOfRobots*(theCollective.NumberOfRobots-1)*0.8;
    bp = bar(learningEnergy,'facecolor', 'flat');
    bp.CData = selectedColors(1:4,:);
    set(gca,'xticklabel',learningCasesLabels)
    ylim([1 1.1*max(learningEnergy)])
    ylabel("Total learning energy [J]")
    fcn_scrpt_prepare_graph_science_std(fig, gca, bp, [], [], 6, 1, 1)
    grid off
    axis square
    box on