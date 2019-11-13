function varargout = scale(training,validation)
min_tr = min(training.sample);
max_tr = max(training.sample);
min_val = min(validation.sample);
max_val = max(validation.sample);
training.sample = rescale(training.sample,-1,1);
validation.sample=rescale(rescale(validation.sample,min_tr,max_tr),-1,1);
varargout = {training, validation};
end

