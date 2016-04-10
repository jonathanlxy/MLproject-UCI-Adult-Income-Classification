function L = sigmoidLikehood(x, y, w)
% x: features of a data point
% y: single number 
sig = 1/(1 + exp(-dot(w,x)));
L = ( (1 - sig)^(1 - y) ) * ( sig^y );
    