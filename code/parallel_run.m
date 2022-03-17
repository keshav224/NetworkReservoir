function [prediction,truth] = parallel_run(data, A, varargin)

% Reservoir Parameters
if ~isempty(varargin) % If reservoir parameters are set, use set values
    resparams = varargin{1};
else
    resparams = load('data/resparams_default.mat').resparams; % If not, use default
end

%% Finding neighbors and Creating Parallel Structure
core_nodes = (1:resparams.NetworkSize)'; % Main node to be predicted
overlap = zeros(length(core_nodes),20);
for node = 1:resparams.NetworkSize
    temp = find(A(:,node)); % Find nearest neighbors
    overlap(node,1:length(temp)) = temp;
end
reservoirs = build_reservoir_clusters(resparams,A,core_nodes,overlap); % Set up the parallel Reservoir Structure
[reservoirs,truth] = select_data(reservoirs,data,resparams);

%% Reservoir Training
reservoirs = train_reservoirs(reservoirs);

%% Reservoir Prediction
prediction = predict_parallel(reservoirs);

end
