function [RFMtrees,Pre,err] = UpSampling2times_RF_train(LSig,HSig,treenum,errT) %LSig（低分辨率信号），HSig（高分辨率信号），treenum（随机森林中树的数量），以及 errT（误差的阈值）
Edata = single(UpDowndataCombine(single(LSig),single(HSig(:,1))));%将低分辨率信号 LSig 与高分辨率信号 HSig 的第一列结合起来，结果存储在变量 Edata 中
% xx = HSig(:,1);
% x = LSig(:,1);
% y = LSig(:,2);
% Edata = pchip(x,y,xx);
% Edata = LocalNbrVec(Edata,2);
Trees = TreeBagger(treenum,Edata,single(HSig(:,2)),'method','regression');%使用 TreeBagger 函数训练一个随机森林回归模型，使用合并后的数据 Edata 作为输入，使用 HSig 的第二列作为响应变量。森林中树的数量由 treenum 指定
RFMtrees{1} = Trees;%训练好的随机森林模型存储在单元数组 RFMtrees 中的第一个元素
Pre = single(predict(Trees,Edata));%使用训练好的模型对与训练使用的相同数据 Edata 进行预测，预测结果存储在变量 Pre
err = sum(abs(Pre - HSig(:,2)))/length(HSig(:,2));%这一行计算预测值 Pre 与实际高分辨率信号值 HSig 第二列之间的平均绝对误差，并将结果存储在变量 err 中。
done = 0;%如果误差低于指定的阈值 errT，表示训练完成。否则，done 保持为 0，训练过程继续。
if err < errT
    done = 1;
end
n=0; %它使用上一次迭代的预测值 Pre 生成新的训练数据 Tdata。循环继续直到误差低于阈值 errT 或达到最大迭代次数（5次）。每次迭代训练的模型存储在单元数组 RFMtrees 中。
while ~done
    n=n+1;
    Tdata = single(LocalNbrVec(Pre,2));
    Trees = TreeBagger(treenum,Tdata,single(HSig(:,2)),'method','regression');
    RFMtrees{1+n} = Trees;
    Pre = single(predict(Trees,Tdata));
    err = sum(abs(Pre - HSig(:,2)))/length(HSig(:,2));
    if err < errT || n > 4
        done = 1;
    end
end


