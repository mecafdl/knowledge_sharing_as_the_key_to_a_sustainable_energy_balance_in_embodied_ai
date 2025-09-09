function Gamma = fcn_generateGamma(gamma_mean, gamma_std, numberOfRobots, COMM_INTERRUPT)
    if COMM_INTERRUPT == 1
        commInterruptMatrix = randn(numberOfRobots,numberOfRobots)>-0.5;
        commInterruptMatrix = (commInterruptMatrix + transpose(commInterruptMatrix))./2;
    else
        commInterruptMatrix = ones(numberOfRobots,numberOfRobots);
    end
    
    % Gamma   = pearsrnd(gamma_mean, gamma_std, -1.5, 10, numberOfRobots,numberOfRobots);
    Gamma   = gamma_mean + gamma_std.*randn(numberOfRobots,numberOfRobots);
    Gamma   = (Gamma + transpose(Gamma))./2;
    Gamma   = Gamma - diag(Gamma).*eye(numberOfRobots);
    Gamma   = Gamma.*commInterruptMatrix;
    % Gamma   = Gamma.*(full(sprandsym(Gamma))>-0.5);
end