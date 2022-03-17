function figure_plot(resparams,indices,truth,prediction)

r_truth = sqrt(sum((truth(:,:,2)),1).^2 + sum((truth(:,:,1)),1).^2)/resparams.NetworkSize;
r_prediction = sqrt(sum((prediction(:,:,2)),1).^2 + sum((prediction(:,:,1)),1).^2)/resparams.NetworkSize;
rmse = (r_truth-r_prediction).^2./sqrt(mean(r_truth.^2));
valid_time_r = resparams.tau*resparams.lyapExp*find(rmse>0.1,1,'first');

valid_time = zeros(resparams.NetworkSize,1);
for k = 1:resparams.NetworkSize
    rmse_NodeLevel = sqrt(sum((truth(k,:,2)-prediction(k,:,2)).^2,3));
    valid_time(k) = resparams.tau*resparams.lyapExp*find(rmse_NodeLevel>0.1,1,'first');
end

figure();
for i = 1: length(indices)
    subplot(2,length(indices),i)
    ind = indices(i);
    disp(ind);
    plot([1:size(truth,2)]*resparams.lyapExp*resparams.tau,truth(ind,:,1),'b','Linewidth',2); hold on
    plot([1:size(prediction,2)]*resparams.lyapExp*resparams.tau,prediction(ind,:,1),'r','Linewidth',2); 
    xline(valid_time(ind),'--k','Linewidth',2);
    ylabel('sin(\theta)');
    xlh = xlabel('t');
    xlh.Position(2) = xlh.Position(2) + 0.07;
    xlim([0,14]);
    set(gca,'FontSize',18);
end

sp_hand1 = subplot(2,3,length(indices)+1:2*length(indices));
plot([1:length(r_truth)]*resparams.lyapExp*resparams.tau,r_truth,'b','Linewidth',2); hold on
plot([1:length(r_prediction)]*resparams.lyapExp*resparams.tau,r_prediction,'r','Linewidth',2); hold on
pos1 = get(sp_hand1, 'Position'); % gives the position of current sub-plot
new_pos1 = pos1 +[0 0 0 0.04];
set(sp_hand1, 'Position',new_pos1); % set new position of current sub - plot
xline(valid_time_r,'--k','Linewidth',2);
ylim([0,0.55]);
xlim([0,10]);
yticks([0,0.2,0.4]);
ylabel('|R|');
xlabel('t');
set(gca,'FontSize',26);
set(gcf,'Position',[100, 100, 1500, 900]);

end