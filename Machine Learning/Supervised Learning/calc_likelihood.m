function [likelihood] = calc_likelihood(class,mean,var)
likelihood = 1/sqrt(2*pi*var) * exp(-0.5*(class - mean).*(class - mean)/var);
end

