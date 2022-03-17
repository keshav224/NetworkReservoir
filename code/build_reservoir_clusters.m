function reservoirs = build_reservoir_clusters(resparams,A,core_nodes,overlap)

reservoirs = ClusterRes.empty(size(core_nodes,1),0);
for vertex = 1:size(core_nodes,1)
        reservoirs(vertex) = ClusterRes(resparams,nonzeros(core_nodes(vertex,:)),...
                                              nonzeros(overlap(vertex,:)),size(A,1));
end

end
