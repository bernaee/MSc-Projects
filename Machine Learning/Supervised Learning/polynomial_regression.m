function [model, lse, actual_curve_lse] = polynomial_regression(x,y,var,degree)
sample = length(x);
actual_curve = x.^3 - x + 1
for i=1:degree+1
    for j=1:degree+1
        for k=1:sample
            A(i,j,k)=x(k).^(i+j-2);
        end
    end
end

for j=1:degree+1
    for k=1:sample
        D(k,j)=x(k).^(j-1);
    end
end

for j=1:degree+1
    for k=1:sample
        D_T(j,k)=x(k).^(j-1);
    end
end

w = inv(D_T*D)*D_T*y;

for j=1:sample
    sum=0.0;
    for k=1:degree+1
        sum = sum + w(k) * x(j)^k;
    end
    model(j,:) = sum;
end

lse = sample * log(sqrt(2*pi*(var^2))) + (1/( 2*(var^2))) * mean((y-model).^2);
actual_curve_lse = sample * log(sqrt(2*pi*(var^2))) + (1/( 2*(var^2))) * mean((actual_curve-model).^2);

end

