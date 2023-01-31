function [env,obsInfo,actInfo] = localCreateEnv(mdl)

% Define the observation specification obsInfo and action specification actInfo.
obsInfo1 = rlNumericSpec([4 1]);
obsInfo1.Name = 'observations';
obsInfo1.Description = 'error, integrated error, error derivate, decoupling';
actInfo1 = rlNumericSpec([1 1], "UpperLimit", 5, "LowerLimit", -5);

obsInfo2 = rlNumericSpec([4 1]);
obsInfo2.Name = 'observations';
obsInfo2.Description = 'error, integrated error, error derivate, decoupling';
actInfo2 = rlNumericSpec([1 1], "UpperLimit", 5, "LowerLimit", -5);

obsInfo = {obsInfo1, obsInfo2};
actInfo = {actInfo1, actInfo2};
% Build the environment interface object.
env = rlSimulinkEnv(mdl,[(mdl+'/MARL Control System/RL Agent1'), (mdl+'/MARL Control System/RL Agent2')],obsInfo,actInfo);

% Set a cutom reset function that randomizes the reference values for the model.
env.ResetFcn = @(in)localResetFcn(in,mdl);
end

function in = localResetFcn(in,mdl)
% randomize reference signal
blk = (mdl+'/Reference');
hRef = 2;
end