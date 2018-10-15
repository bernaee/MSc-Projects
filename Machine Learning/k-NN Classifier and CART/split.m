function varargout = split(value, training)
left.x=[];
left.r=[];
right.x=[];
right.r=[];
	for i=1:length(training.x) 
		if training.x(i) < value
			left.x = [left.x,training.x(i)];
            left.r = [left.r,training.r(i)];
        else
			right.x = [right.x,training.x(i)];
            right.r = [right.r,training.r(i)];
        end
    end
varargout={left,right};
end

