function run_kNN(data)
[training,validation] = split_train_validation_data(data);

kNN('euclidean',training,validation,1,'Euclidean with k=1');
kNN('euclidean',training,validation,3,'Euclidean with k=3');
kNN('euclidean',training,validation,5,'Euclidean with k=5');

kNN('mahalanobis',training,validation,1,'Mahalanobis with k=1');
kNN('mahalanobis',training,validation,3,'Mahalanobis with k=3');
kNN('mahalanobis',training,validation,5,'Mahalanobis with k=5');

end

