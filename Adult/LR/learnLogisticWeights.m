function newW = learnLogisticWeights(w0, x, y, n, re, lambda)
step = 0.001; % Learning step size
iter = 1;  % Iterate index

% Each iteration
while iter <= n
    % Visit each datapoint
    for i = 1:size(x, 1) 
        % Calculate Sigmoid p(y=1|x)
        p = 1 / ( 1 + exp(-dot(x(i,:),w0)) );
        % Visit each feature
        for j = 1:size(x, 2)
            % Update w0 using gradient decent
            % L1
            if re == 1
                b = sign(w0(j))/lambda;
            % L2
            elseif re == 2
                b = w0(j)/lambda;
            % No regularization
            else
                b = 0;
            end
            w0(j) = w0(j) + step * ( x(i, j) * ( y(i) - p) - b);
        end
        % End of visiting one datapoint & all feature
    end
    % End of visiting all datapoint & all feature
    iter = iter + 1;
end
% End of iterations

newW = w0;
    