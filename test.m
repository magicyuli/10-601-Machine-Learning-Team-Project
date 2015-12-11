bst_np = 0;
bst_acc = 0;
for np = 102:200
    
    [m, acc] = train_svm(train_data(2001:3000, :), train_label(2001:3000, :), ...
                train_data, train_label, np);
            
    if bst_acc < acc
       bst_acc = acc;
       bst_np = np;
    end
end

fprintf('best pca: %d, best acc: %f\n', bst_np, bst_acc);