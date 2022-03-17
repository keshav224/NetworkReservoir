classdef ClusterRes
    properties
        network_size
        nodes
        bias
        overlap
        dimension
        states_data
        predict_data
        res_num0
        res_num
        assign
        radius
        degree
        N
        sigma
        train_length
        num_inputs
        predict_length
        beta
        lambda
        A
        win
        x
        wout
        states
    end
    methods
        function obj = ClusterRes(resparams,core_nodes,overlap,network_size)
            if nargin == 4
                obj.network_size    = network_size;
                obj.bias            = resparams.bias;
                obj.nodes           = core_nodes;
                obj.overlap         = overlap;
                obj.num_inputs      = length(obj.nodes)+length(obj.overlap);             
                obj.radius          = resparams.radius;
                obj.degree          = resparams.degree;
                obj.sigma           = resparams.sigma;
                obj.train_length    = resparams.train_length;
                obj.dimension       = resparams.dimension;
                obj.res_num0        = round(resparams.N_assign./(length(obj.nodes)*obj.dimension));
                obj.assign          = obj.res_num0*length(obj.nodes)*obj.dimension;
                if isempty(obj.overlap)
                    obj.res_num         = 0;
                    obj.N               = obj.res_num0*length(obj.nodes)*obj.dimension;
                else
                    obj.res_num         = round((resparams.N-obj.assign)/(length(obj.overlap)*obj.dimension));
                    obj.N               = obj.assign + (obj.res_num)*length(obj.overlap)*obj.dimension;
                end
                
                obj.predict_length  = resparams.predict_length;
                obj.beta            = resparams.beta;
                obj.lambda          = resparams.lambda;
                obj.win             = zeros(obj.N,obj.num_inputs);
                for i = 1:length(obj.nodes)
                    for j = 1:obj.dimension
                        obj.win((j-1)*obj.res_num0+(i-1)*obj.res_num0*obj.dimension+1:j*obj.res_num0+(i-1)*obj.res_num0*obj.dimension,obj.num_inputs*(j-1)+i) = resparams.sigma.*(2.*rand(obj.res_num0,1)-1);
                    end
                end
                for i = 1:length(obj.overlap)
                    for j = 1:obj.dimension
                        obj.win((j-1)*obj.res_num+(i-1)*obj.res_num*obj.dimension+obj.assign+1:j*obj.res_num+(i-1)*obj.res_num*obj.dimension+obj.assign,obj.num_inputs*(j-1)+i+length(obj.nodes)) = resparams.sigma.*(2.*rand(obj.res_num,1)-1);
                    end
                end
                sparsity = obj.degree/obj.N;
                obj.A = sprand(obj.N, obj.N, sparsity);
                e = max(abs(eigs(obj.A)));
                obj.A = (obj.A./e).*obj.radius;
            end
        end
        
        function [objs,true_data] = select_data(objs,data,resparams)
            % format of data is nodes x time x sin/cos
            true_data = data(:,resparams.train_length:resparams.train_length+resparams.predict_length-1,:);
            train_data = data(:,1:resparams.train_length,:);
            for index = 1:length(objs)
                objs(index).states_data = ...
                    train_data([objs(index).nodes;objs(index).overlap],:,:);
            end
        end
        
        function states = reservoir_layer(A, win, input, resparams)
            states = zeros(resparams.N, resparams.train_length); 
            for i = 1:resparams.train_length-1
                    states(:,i+1) = (resparams.lambda*states(:,i)) + (1-resparams.lambda)*tanh(A*states(:,i) + win*input(:,i) + resparams.bias);
            end
        end
        
        function objs = train_reservoirs(objs)
            % Generate reservoir states
            for index = 1:length(objs)
                % adjust train_data to be a stacked sin/cos 2x2 matrix
                state_data = reshape(permute(objs(index).states_data,[1,3,2]), size(objs(index).states_data,1)*objs(index).dimension,[]);
                objs(index).states = reservoir_layer(objs(index).A, objs(index).win, state_data, objs(index));
                objs(index).x = objs(index).states(:,end); % Last state
                % determine the number of core_nodes in the cluster
                num_core = length(objs(index).nodes);
                adj = reshape(permute(objs(index).states_data(1:num_core,:,:),[1,3,2]), num_core*objs(index).dimension,[]);
                % Linear regression (Tikhonov regularized)
                objs(index).wout = adj*objs(index).states'/(objs(index).states*objs(index).states'+objs(index).beta*speye(objs(index).N));
                disp(num2str(index));
            end
        end
        
        function extracted = extract_data(objs)
            extracted = zeros(objs(1).network_size,1,2);
            for id = 1:length(objs)
                res_output = objs(id).wout*objs(id).x/norm(objs(id).wout*objs(id).x);
                for j = 1:size(objs(id).states_data,3)
                    extracted(objs(id).nodes,1,j) = res_output(j,:);
                end
            end
        end
        
        function objs = assign_next_data(objs,input)
            % dimension of input is network_size x 1 x 2
            for id = 1:length(objs)
                temp = input([objs(id).nodes;objs(id).overlap],1,:);
                objs(id).predict_data = reshape(permute(temp,[1,3,2]),size(temp,1)*objs(id).dimension,[]);
            end
        end
        
        % method for performing prediction
        function output = predict_parallel(objs)
            output = zeros(objs(1).network_size,objs(1).predict_length,objs(1).dimension);
            output(:,1,:) = extract_data(objs);
            
            for t = 2:objs(1).predict_length
                objs = assign_next_data(objs,output(:,t-1,:));
                for id = 1:length(objs)
                    objs(id).x = (objs(id).lambda*objs(id).x) + (1 - objs(id).lambda) * tanh(objs(id).A*objs(id).x + objs(id).win*objs(id).predict_data + objs(id).bias);
                end
                output(:,t,:) = extract_data(objs);
                if mod(t,1000) == 0
                    disp(t);
                end
            end
        end
               

    end
end