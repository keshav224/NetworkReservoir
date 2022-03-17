addpath('../code');
load('../data/kuramoto_50.mat');
data = reshape(datfile,resparams.NetworkSize,[],resparams.dimension);
[prediction,truth] = parallel_run(data, A, resparams);

plotIndices = [2,41,17];
figure_plot(resparams,plotIndices,prediction,truth);