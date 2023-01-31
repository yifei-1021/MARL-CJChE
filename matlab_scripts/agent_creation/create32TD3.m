function agent = create32TD3(obsInfo, actInfo, Ts)
% Distillation, 2nd Feedback + Decoupler
% Save as load('DistillDecagent32.mat','agent32') 
j = 1/3;
initialGain = single([j j j j]);
learnrate = 0.8e-6;
numObservations = obsInfo.Dimension(1);
numActions = actInfo.Dimension(1);
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','state')
    fullyConnectedPILayer(initialGain, 'Action')];

actorNetwork = dlnetwork(actorNetwork);
actorOptions = rlOptimizerOptions('LearnRate',learnrate,'GradientThreshold',1);
actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);

criticNetwork = localCreateCriticNetwork(numObservations,numActions);
criticOpts = rlOptimizerOptions('LearnRate',learnrate,'GradientThreshold',1);

critic1 = rlQValueFunction(dlnetwork(criticNetwork),obsInfo,actInfo,...
    'ObservationInputNames','state','ActionInputNames','action');
critic2 = rlQValueFunction(dlnetwork(criticNetwork),obsInfo,actInfo,...
    'ObservationInputNames','state','ActionInputNames','action');
critic = [critic1 critic2];

agentOpts = rlTD3AgentOptions(...
    'SampleTime',Ts,...
    'MiniBatchSize',256, ...
    'ExperienceBufferLength',1e8,...
    'ActorOptimizerOptions',actorOptions,...
    'CriticOptimizerOptions',criticOpts);
agentOpts.TargetPolicySmoothModel.StandardDeviation = sqrt(0.01);
agentOpts.ExplorationModel.Variance = 0.001;
agentOpts.ExplorationModel.VarianceDecayRate = 2e-4;
agentOpts.ExplorationModel.VarianceMin = 0.0001;
agent = rlTD3Agent(actor,critic,agentOpts);
end

function criticNetwork = localCreateCriticNetwork(numObservations,numActions)
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','state')
    fullyConnectedLayer(32,'Name','fc1')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','action')
    fullyConnectedLayer(32,'Name','fc2')];
commonPath = [
    concatenationLayer(1,2,'Name','concat')
    reluLayer('Name','reluBody1')
    fullyConnectedLayer(32,'Name','fcBody')
    reluLayer('Name','reluBody2')
    fullyConnectedLayer(1,'Name','qvalue')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);

criticNetwork = connectLayers(criticNetwork,'fc1','concat/in1');
criticNetwork = connectLayers(criticNetwork,'fc2','concat/in2');
end