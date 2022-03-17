addpath('../code');
load('../data/kuramoto_50.mat');
[prediction,truth] = parallel_run(data, A, resparams);

plotIndices = [2,41,17];
figure_plot(resparams,plotIndices,prediction,truth);