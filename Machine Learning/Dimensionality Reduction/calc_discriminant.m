function [discriminant] = calc_discriminant(discriminant_func,class, mean,cov, prior)
[sample_size,dim] =size(class);
 if isequal(discriminant_func,'quadratic')
    discriminant = (-0.5) * log(det(cov)) - (0.5*sum((class.' - mean.').'*inv(cov)*(class.' - mean.'),2)) + log(prior);
 elseif isequal(discriminant_func,'linear')
     discriminant = - (0.5*sum((class.' - mean.').'*inv(cov)*(class.' - mean.'),2)) + log(prior);
 elseif isequal(discriminant_func,'naivebayes')
     discriminant = - (0.5*sum(((class - mean)./diag(cov).').*((class - mean)./diag(cov).'),2)) + log(prior);
 elseif isequal(discriminant_func,'euclidean')
     discriminant =- ((0.5/det(diag(diag(cov))))*sum((class - mean).*(class - mean),2)) + log(prior);
end

