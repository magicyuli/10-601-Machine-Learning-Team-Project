bst_np = 0;
bst_acc = 0;
for np = 60:200
    
    y = classify2(Model2, data, np);
    accu = sum(y==labels)/10000;
    fprintf('pca: %d, acc: %f\n', np, accu);
    if bst_acc < accu
       bst_acc = accu;
       bst_np = np;
    end
end

fprintf('best pca: %d, best acc: %f\n', bst_np, bst_acc);