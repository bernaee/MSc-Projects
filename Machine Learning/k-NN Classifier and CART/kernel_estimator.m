function kernel = kernel_estimator(distance,u,S)
[sample_size,dim] =size(u);
if isequal(distance,'mahalanobis')
    kernel = (u*inv(S)*u.');
elseif isequal(distance,'euclidean')
    kernel =  sum(u.^2); 
end
end

