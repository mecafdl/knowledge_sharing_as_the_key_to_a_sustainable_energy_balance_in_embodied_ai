clearvars
clc
close all
cd(fileparts(matlab.desktop.editor.getActiveFilename));

%% Run (distributed) collective learning scenarios for different values of eta and gamma
close all
clc

theEtas           = [-0.1 0 0.1];
theGammas         = [-0.2 0 0.2];
[X, Y]            = meshgrid(theEtas,theGammas);
scenarios         = [X(:) Y(:)];
numberOfScenarios = size(scenarios,1);
   
% robotCollectivesSize = 4:4:128;
robotCollectivesSize = [4 8 16 32 64 128];
cl_scenarios_results = cell(numberOfScenarios,numel(robotCollectivesSize));

tStart = tic;
for scenarioIndex = 1:numberOfScenarios
    for robotCollectiveIndex = 1:numel(robotCollectivesSize)  
        close all
        % Initialize a robot collective
        theCollective = RobotCollective(numberOfRobots  = robotCollectivesSize(robotCollectiveIndex), repeatingSkills = false);
        theCollective.RAND_COMM_INTERRUPT         = true; 
        theCollective.ENABLE_INCREMENTAL_LEARNING = true; 
        theCollective.ENABLE_TRANSFER_LEARNING    = true; 
        theCollective.ENABLE_COLLECTIVE_LEARNING  = true; 
        
        % Define simulation episodes
        theCollective.Episodes = 1:0.1:2*theCollective.FundamentalComplexity;
        
        % Define agent(s) learning factors
        theCollective.Eta_0   = scenarios(scenarioIndex,1);
        theCollective.Gamma_0 = scenarios(scenarioIndex,2);    
        
        % Max. number of skills per product
        theCollective.MaxNumberOfSkillsPerProduct = robotCollectivesSize(robotCollectiveIndex);

        % Run the scenario
        cprintf('green',sprintf('[INFO] DCL: collective size %2i | eta_0 = %0.2f | gamma_0 = %0.2f ...\n', robotCollectivesSize(robotCollectiveIndex),theCollective.Eta_0,theCollective.Gamma_0))
        pause(1)
        [~, ~, ~, cl_scenarios_results{scenarioIndex, robotCollectiveIndex}] = ...
        theCollective.simulateDistributedCollectiveKnowledgeDynamics(maxNumberOfProducts = 1000);
    end
end
tEnd = toc(tStart);

%% Create plot
close all
clc



scenarioLegends = {...
    '$\bar{\eta}>0,\bar{\gamma}<0$',...
    '$\bar{\eta}>0,\bar{\gamma}_0$',...
    '$\bar{\eta}>0,\bar{\gamma}>0$',...
    '$\bar{\eta}_0,\bar{\gamma}<0$',...
    '$\bar{\eta}_0,\bar{\gamma}_0$',...
    '$\bar{\eta}_0,\bar{\gamma}>0$',...
    '$\bar{\eta}<0,\bar{\gamma}<0$',...
    '$\bar{\eta}<0,\bar{\gamma}_0$',...
    '$\bar{\eta}<0,\bar{\gamma}>0$'};

fig = figure('color','w','Renderer','opengl');

success_cmap = colormap('jet');
% Create indices to pick colors evenly from the colormap
success_colorIndices = linspace(0, size(success_cmap, 1), 100);


aux_index = 1;
for index = [7,8,9,4,5,6,1,2,3] % The indices match the order of the above legends
    ax = subplot(3,3,aux_index);
    
    % Color the N/A area
    upperBound = 12800*ones(numel(robotCollectivesSize),1);
    lowerBound = theCollective.TotalSkills./robotCollectivesSize'.*theCollective.FundamentalComplexity;
    patch([robotCollectivesSize fliplr(robotCollectivesSize)], [lowerBound'  fliplr(upperBound')], [0.5  0.5  0.5],'FaceAlpha',0.25,'EdgeColor','w');
    hold on

    % Compute the total number of episodes used for learning the skills (even if unsuccessful)
    totalLearningEpisodes   = arrayfun(@(r) sum(cl_scenarios_results{index,r}.c_jk_cl_dist_episodes),1:numel(robotCollectivesSize));
    totalLearnedSkills      = arrayfun(@(r) mean(cl_scenarios_results{index,r}.learnedSkillsStorage),1:numel(robotCollectivesSize));
    learnedSkillsPercentage = 100*totalLearnedSkills./theCollective.TotalSkills;
    
    % Auxiliary code to color the line with a smooth color gradient corresponding to the success rate
    xx                       = 4:1:128;
    yy                       = interp1(robotCollectivesSize, totalLearningEpisodes, xx,"pchip");
    learnedSkillsPercentage2 = interp1(robotCollectivesSize, learnedSkillsPercentage, xx);


    upperBound  =  12800*ones(size(xx))';
    upperBound  = upperBound(:);
    lowerBound  = theCollective.TotalSkills./xx'.*theCollective.FundamentalComplexity;
    lowerBound  = lowerBound(:);
    patch([xx fliplr(xx)], [lowerBound'  fliplr(upperBound')], [0.7  0.7  0.7],'FaceAlpha',0.25,'EdgeColor','w');

    X = [xx(:) xx(:)];  % Create a 2D matrix based on "X" column
    Y = [yy(:) yy(:)];  % Same for Y
    Z = zeros(size(X)); % Everything in the Z=0 plane

% Z = [learnedSkillsPercentage2(:) learnedSkillsPercentage2(:)];

    C =[learnedSkillsPercentage2(:) learnedSkillsPercentage2(:)] ;  %Matrix for "CData"

    % Draw the surface (actually a line)
    hs=surf(X,Y,Z,C,'EdgeColor','interp','FaceColor','interp',LineWidth=2) ;

    % Add a horizontal colorbar to annotate the success rate
    c = colorbar('southoutside');
    c.Box = "off";
    % cbh = colorbar('YTickLabel', num2cell(1:8)) ;
    % Get the current positions
    axPos = ax.Position;
    c.Position(3) = 0.1;
    cPos = c.Position;

    % Adjust the colorbar's position to be inside the axes
    cPos(1) = axPos(1) + 0.06; % Move up
    cPos(2) = axPos(2) + 0.04; % Move up
    cPos(4) = 0.005;           % Reduce height
    c.Position = cPos;
    clim(gca,[0, 100]);
    ylabel(c,'Success rate')
    
    colormap jet

    % Set axes limits
    xlim([4 128])
    ylim([1 12800])
    yticks([1E+0 1E+1 1E+2 1E+3 1E+4])
    yticklabels({'10^0','10^1', '10^2', '10^3', '10^4'})
    xlabel('Number of agents','FontSize',11)
    ylabel('Total learning episodes','FontSize',11)
    set(gca, 'YScale', 'log')
    
    p       = plot(NaN,NaN, 'Linestyle', 'none', 'Marker', 'none', 'Color', 'none');
    leg     = legend(p, scenarioLegends{aux_index}, 'Interpreter', 'latex', fontsize=11);
    leg.Box = "off";

    aux_index = aux_index + 1;
    axis square


            % fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 6, 1, 1)
            % % pause(1)
            % tightfig(fig);

end

%% If desired export the figure
EXPORT_FIG = false
if(EXPORT_FIG)
    f = gcf;
    exportgraphics(f,'Peppers300.png','Resolution',600)
end

%%

%% Create plot
close all
clc



% scenarioLegends = {...
%     '$\bar{\eta}>0,\bar{\gamma}<0$',...
%     '$\bar{\eta}>0,\bar{\gamma}_0$',...
%     '$\bar{\eta}>0,\bar{\gamma}>0$',...
%     '$\bar{\eta}_0,\bar{\gamma}<0$',...
%     '$\bar{\eta}_0,\bar{\gamma}_0$',...
%     '$\bar{\eta}_0,\bar{\gamma}>0$',...
%     '$\bar{\eta}<0,\bar{\gamma}<0$',...
%     '$\bar{\eta}<0,\bar{\gamma}_0$',...
%     '$\bar{\eta}<0,\bar{\gamma}>0$'};


scenarioLegends = {...
    '$\bar{\eta}>0,\bar{\gamma}<0$',...
    '$\bar{\eta}>0,\bar{\gamma}=0$',...
    '$\bar{\eta}>0,\bar{\gamma}>0$',...
    '$\bar{\eta}=0,\bar{\gamma}<0$',...
    '$\bar{\eta}=0,\bar{\gamma}=0$',...
    '$\bar{\eta}=0,\bar{\gamma}>0$',...
    '$\bar{\eta}<0,\bar{\gamma}<0$',...
    '$\bar{\eta}<0,\bar{\gamma}=0$',...
    '$\bar{\eta}<0,\bar{\gamma}>0$'};

fig = figure('color','w','Renderer','opengl');

success_cmap = colormap('jet');
% Create indices to pick colors evenly from the colormap
success_colorIndices = linspace(0, size(success_cmap, 1), 100);


aux_index = 1;
for index = [7,8,9,4,5,6,1,2,3] % The indices match the order of the above legends
    clf
    % Color the N/A area
    % upperBound = 12800*ones(numel(robotCollectivesSize),1);
    % lowerBound = theCollective.TotalSkills./robotCollectivesSize'.*theCollective.FundamentalComplexity;
    % patch([robotCollectivesSize fliplr(robotCollectivesSize)], [lowerBound'  fliplr(upperBound')], [0.5  0.5  0.5],'FaceAlpha',0.25,'EdgeColor','w');
    hold on

    % Compute the total number of episodes used for learning the skills (even if unsuccessful)
    totalLearningEpisodes   = arrayfun(@(r) sum(cl_scenarios_results{index,r}.c_jk_cl_dist_episodes),1:numel(robotCollectivesSize));
    totalLearnedSkills      = arrayfun(@(r) mean(cl_scenarios_results{index,r}.learnedSkillsStorage),1:numel(robotCollectivesSize));
    learnedSkillsPercentage = 100*totalLearnedSkills./theCollective.TotalSkills;
    
    % Auxiliary code to color the line with a smooth color gradient corresponding to the success rate
    xx                       = 4:1:128;
    yy                       = interp1(robotCollectivesSize, totalLearningEpisodes, xx,"pchip");
    learnedSkillsPercentage2 = interp1(robotCollectivesSize, learnedSkillsPercentage, xx);


    upperBound  =  12800*ones(size(xx))';
    upperBound  = upperBound(:);
    lowerBound  = theCollective.TotalSkills./xx'.*theCollective.FundamentalComplexity;
    lowerBound  = lowerBound(:);
    
    
    yy(yy(:)>lowerBound) = lowerBound(yy(:)>lowerBound);
    
    patch([xx fliplr(xx)], [lowerBound'  fliplr(upperBound')], [0.7  0.7  0.7],'FaceAlpha',0.25,'EdgeColor','w');

% plot(robotCollectivesSize,totalLearningEpisodes,LineWidth=2) ;

    X = [xx(:) xx(:)];  % Create a 2D matrix based on "X" column
    Y = [yy(:) yy(:)];  % Same for Y
    Z = zeros(size(X)); % Everything in the Z=0 plane

% Z = [learnedSkillsPercentage2(:) learnedSkillsPercentage2(:)];

    C =[learnedSkillsPercentage2(:) learnedSkillsPercentage2(:)] ;  %Matrix for "CData"

    % Draw the surface (actually a line)
    hs=surf(X,Y,Z,C,'EdgeColor','interp','FaceColor','interp',LineWidth=2) ;

    % Add a horizontal colorbar to annotate the success rate
    c = colorbar('southoutside');
    c.Box = "off";
    % cbh = colorbar('YTickLabel', num2cell(1:8)) ;
    % Get the current positions
    ax = gca;
    axPos = ax.Position;
    c.Position(3) = 0.1;
    cPos = c.Position;

    % Adjust the colorbar's position to be inside the axes
    cPos(1) = axPos(1) + 0.06; % Move up
    cPos(2) = axPos(2) + 0.04; % Move up
    cPos(4) = 0.005;           % Reduce height
    c.Position = cPos;
    clim(gca,[0, 100]);
    ylabel(c,'Success rate')
    
    colormap jet
    delete(c)

    % Set axes limits
    xlim([4 128])
    ylim([1 12800])
    yticks([1E+0 1E+1 1E+2 1E+3 1E+4])
    yticklabels({'10^0','10^1', '10^2', '10^3', '10^4'})
    xlabel('Number of agents','FontSize',11)
    ylabel('Total learning episodes','FontSize',11)
    set(gca, 'YScale', 'log')
    
    p       = plot(NaN,NaN, 'Linestyle', 'none', 'Marker', 'none', 'Color', 'none');
    leg     = legend(p, scenarioLegends{aux_index}, 'Interpreter', 'latex', fontsize=11);
    leg.Box = "off";

    aux_index = aux_index + 1;
    axis square
    
    % Format the figure
    
    fcn_scrpt_prepare_graph_science_std(fig, gca, [], [], [], 6, 1, 1)
    grid off
    pause(1)
    % tightfig(fig);
    EXPORT_FIG = true;
    if(EXPORT_FIG)
        exportgraphics(fig,sprintf('cl_case_%i.png',index),'Resolution',600);
    end
end