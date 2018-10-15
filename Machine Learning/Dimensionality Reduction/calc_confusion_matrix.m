function calc_confusion_matrix(pred_labels_0,pred_labels_1)
tp = length(find(pred_labels_0==0));
fn = length(find(pred_labels_0==1));
fp = length(find(pred_labels_1==0));
tn =  length(find(pred_labels_1==1));
cm=[[tp,fn];[fp,tn]];
end

