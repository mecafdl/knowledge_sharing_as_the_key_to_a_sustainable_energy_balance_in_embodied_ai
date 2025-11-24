clearvars
clc
close all

mainPath = fileparts(matlab.desktop.editor.getActiveFilename);
cd(mainPath);
addpath(genpath(fullfile(mainPath,'supporting_files')));

%% ************************************************************************
% Run collective learning scenarios for different values of eta and gamma
% *************************************************************************

try
    clc
    load("./paper_results/collective_learning_regimes_results.mat")
    cprintf('g', 'Successfully loaded file\n')
    RUN_CL = 0;
catch
    warning('File not found')
    RUN_CL = 1;
end

if RUN_CL == 1
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
            theCollective = RobotCollective(numberOfRobots              = robotCollectivesSize(robotCollectiveIndex), ...
                                            repeatingSkills             = false, ...
                                            maxNumberOfSkillsPerProduct = robotCollectivesSize(robotCollectiveIndex));
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
            % theCollective.MaxNumberOfSkillsPerProduct = robotCollectivesSize(robotCollectiveIndex);
    
            % Run the scenario
            cprintf('green',sprintf('[INFO] DCL: collective size %2i | eta_0 = %0.2f | gamma_0 = %0.2f ...\n', robotCollectivesSize(robotCollectiveIndex),theCollective.Eta_0,theCollective.Gamma_0))
            pause(1)
            [~, ~, ~, cl_scenarios_results{scenarioIndex, robotCollectiveIndex}] = ...
            theCollective.simulateDistributedCollectiveKnowledgeDynamics(maxNumberOfProducts = 1000);
        end
    end
    tEnd = toc(tStart);
end

%% ************************************************************************
%  Create plot of the nine CL regimes 
% *************************************************************************

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

% myVar = NaN(6,2,9);
myVar = NaN(125,3,9);
aux_index = 1;
energyPerEpisode = 105000;
for index = [7,8,9,4,5,6,1,2,3] % The indices match the order of the above legends
    ax = subplot(3,3,aux_index);
    
    % Color the N/A area
    upperBound = 12800*ones(numel(robotCollectivesSize),1);
    lowerBound = theCollective.TotalSkills./robotCollectivesSize'.*theCollective.FundamentalComplexity;
    patch([robotCollectivesSize fliplr(robotCollectivesSize)], [lowerBound'  fliplr(upperBound')], [0.5  0.5  0.5],'FaceAlpha',0.25,'EdgeColor','w');
    hold on

    % Compute the total number of episodes used for learning the skills (even if unsuccessful)
    totalLearningEpisodes   = arrayfun(@(r) sum(cl_scenarios_results{index,r}.c_jk_cl_dist_episodes),1:numel(robotCollectivesSize));
    
    
    totalLearningEnergy     = (energyPerEpisode + 0.8*robotCollectivesSize.*(robotCollectivesSize-1)).*totalLearningEpisodes;
    
    
    totalLearnedSkills      = arrayfun(@(r) mean(cl_scenarios_results{index,r}.learnedSkillsStorage),1:numel(robotCollectivesSize));
    learnedSkillsPercentage = 100*totalLearnedSkills./theCollective.TotalSkills;
    
% myVar(:,:,index) = [totalLearningEpisodes', learnedSkillsPercentage'];

    % Auxiliary code to color the line with a smooth color gradient corresponding to the success rate
    xx                       = 4:1:128;
    yy                       = interp1(robotCollectivesSize, totalLearningEpisodes, xx,"pchip");
    zz                       = interp1(robotCollectivesSize, totalLearningEnergy, xx,"pchip");
    learnedSkillsPercentage2 = interp1(robotCollectivesSize, learnedSkillsPercentage, xx);

myVar(:,:,index) = [yy', learnedSkillsPercentage2', zz'];

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
EXPORT_FIG = false;
if(EXPORT_FIG)
    f = gcf;
    exportgraphics(f,'Peppers300.png','Resolution',600)
end

%% ************************************************************************
% Plot of the CL greater regimes (ALT 1: y-axis = episodes)
% *************************************************************************

clc
close all

theMarkers = {"o","square","diamond","^","v",">","<","pentagram","hexagram"};
alphaRange = linspace(0.5,1,125);
clc
theColors  = distinguishable_colors(9);
close all

for regime = 1:4
    fig = figure('Color','w');
    ax = gca;
    xlim(ax,[0,100])
    ylim(ax,[1,15000])
    axis square
    xlabel('Success rate [%]','FontSize',11)
    ylabel('Total learning episodes','FontSize',11)
    hold on
    
    p = [];
    
    theCases  = [7,8,9,4,5,6,1,2,3];
    
    supraRegimes = {[8,9],[1,4,7],[2,5],[3,6]};

    for k = theCases(supraRegimes{regime})
        interval = 20;
        plot(myVar(:,2,k),myVar(:,1,k),'-','Color',theColors(k,:),'LineWidth',1);
        aux = scatter(myVar(1:interval:end,2,k),myVar(1:interval:end,1,k),50,'Marker',theMarkers{k},'MarkerFaceColor',theColors(k,:),'MarkerEdgeColor','k');
        p = [p,aux];
    end
    
    ax.YAxis.Exponent = 3;
    theCasesLabels = {'Case 7','Case 8','Case 9','Case 4','Case 5','Case 6','Case 1','Case 2','Case 3'};
    leg = legend(p,theCasesLabels{theCases(supraRegimes{regime})});
    % set(gca,'XScale','log')
    set(gca,'YScale','log')
    % ax = gca;
    ax.YTick = [1E0 1E1 1E2 1E3 1E4];
    ax.YTickLabel = {'10^0', '10^1' '10^2', '10^3', '10^4'};
    % ax.YTicklabel  = {'10^0','10^1', '10^2', '10^3', '10^4'};
    fcn_scrpt_prepare_graph_science_std(fig, ax, p, leg, [], 6, 1, 1)
    % yticks(ax,[1E0 1E1 1E2 1E3 1E4])   
    % ax.YTickLabel = {'1E0', '1E1' '1E2', '1E3', '1E4'};
    grid off
    % yticklabels({'','$\epsilon$', '', '1'})
    axis square
    box on
    
    % Export figure
    % EXPORT_FIG = true;
    % if(EXPORT_FIG)
    %     f = gcf;
    %     name =  sprintf("supra_regimes_%s.pdf",strrep(num2str(supraRegimes{regime}),' ',''));
    %     exportgraphics(f,name,'Resolution',600)
    % end
    % pause(1)
end

%% ************************************************************************
% Plot of the CL greater regimes (ALT 2: y-axis = energy)
% *************************************************************************


%     yy                       = interp1(robotCollectivesSize, totalLearningEpisodes, xx,"pchip");
%     learnedSkillsPercentage2 = interp1(robotCollectivesSize, learnedSkillsPercentage, xx);
% 
% myVar(:,:,index) = [yy', learnedSkillsPercentage2'];
% 
% 
% % totalLearningEpisodes = (105000 + 0.8*robotCollectivesSize.*(robotCollectivesSize-1)).*totalLearningEpisodes;      

clc
close all

theMarkers = {"o","square","diamond","^","v",">","<","pentagram","hexagram"};
alphaRange = linspace(0.5,1,125);
clc
theColors  = distinguishable_colors(9);
close all

for regime = 1:4
    fig = figure('Color','w');
    ax = gca;
    xlim(ax,[0,100])
    ylim(ax,[1,10^10])
    axis square
    xlabel('Success rate [%]','FontSize',11)
    ylabel('Total learning energy (J)','FontSize',11)
    % zlabel('Size of the collective','FontSize',11)
    % view([45 35])
    hold on
    
    p = [];
    
    theCases  = [7,8,9,4,5,6,1,2,3];
    
    supraRegimes = {[8,9],[1,4,7],[2,5],[3,6]};

    for k = theCases(supraRegimes{regime})
        interval = 20;
        plot(myVar(:,2,k),myVar(:,3,k),'-','Color',theColors(k,:),'LineWidth',1);
        aux = scatter(myVar(1:interval:end,2,k),myVar(1:interval:end,3,k),50,'Marker',theMarkers{k},'MarkerFaceColor',theColors(k,:),'MarkerEdgeColor','k');
        p = [p,aux];

% p   = plot3(myVar(:,2,k),myVar(:,1,k),xx,'-','Color',theColors(k,:),'LineWidth',1);
% aux = scatter3(myVar(1:interval:end,2,k),myVar(1:interval:end,1,k),xx(1:interval:end),50,'Marker',theMarkers{k},'MarkerFaceColor',theColors(k,:),'MarkerEdgeColor','k');        
% p = [p,aux];

    end
    
    ax.YAxis.Exponent = 3;
    theCasesLabels = {'Case 7','Case 8','Case 9','Case 4','Case 5','Case 6','Case 1','Case 2','Case 3'};
    leg = legend(p,theCasesLabels{theCases(supraRegimes{regime})});
    % set(gca,'XScale','log')
    set(gca,'YScale','log')
    % ax = gca;
ax.YTick = [1E0 1E3 1E6 1E10];
ax.YTickLabel = {'10^0', '10^3' '10^6', '10^{10}'};
    % ax.YTicklabel  = {'10^0','10^1', '10^2', '10^3', '10^4'};
    fcn_scrpt_prepare_graph_science_std(fig, ax, p, leg, [], 8.5, 1, 1)
    % yticks(ax,[1E0 1E1 1E2 1E3 1E4])   
    % ax.YTickLabel = {'1E0', '1E1' '1E2', '1E3', '1E4'};
    grid off
    % yticklabels({'','$\epsilon$', '', '1'})
    axis square
    box on
    
    % Export figure
    EXPORT_FIG = true;
    if(EXPORT_FIG)
        f = gcf;
        % name =  sprintf("supra_regimes_%s.pdf",strrep(num2str(supraRegimes{regime}),' ',''));
        name =  sprintf("energy_in_supra_regimes_%s.pdf",strrep(num2str(supraRegimes{regime}),' ',''));
        exportgraphics(f,name,'Resolution',600)
    end
    pause(1)
end


%% Plot the CL scenarios individually
close all
clc

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
    % EXPORT_FIG = true;
    % if(EXPORT_FIG)
    %     exportgraphics(fig,sprintf('cl_case_%i.png',index),'Resolution',600);
    % end
end