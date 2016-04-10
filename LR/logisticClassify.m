function labelVector = logisticClassify(x, w)
    labelVector = zeros(size(x, 1), 1);
    for i = 1:size(x, 1)
        labelVector(i) = sigmoidLikehood(x(i,:), 1, w) > sigmoidLikehood(x(i,:), 0, w);
    end