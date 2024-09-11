function [OutParams] = MainFun(LogName,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Well Logging SuperResolution MainFun using High resolution ImageRes data%
% 2020-05-15 Created by Dr.Zhimin Cao.                                    %
% Requirements:                                                           %
% 1.模型井数据存放到一个文件夹里，0.125米常规曲线放到（每口井）一个csv文件%
%   里，高分辨电阻率曲线放到（每口井）一个csv文件里。                     %
% 2.保证常规曲线深度范围与高分辨曲线深度范围相同(顶底取整,高分辨深度最接近%
%   整数点或正好整数点)                                                   %
% 3.目前测试井文件夹默认只做1口井的测试，可修改对多井某曲线进行超分辨     %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mode = options.mode;
if strcmp(mode,'train')
    %%% 步骤1：读取高分辨数据
    errT = options.errT;
    treenum = options.treenum;
    [~,HDepths,HResLogs] = HResLogsReading;  %手动读取文件
    [~,CDepths,ConvLogs] = ConvLogsReading(LogName);
    lnum = length(HResLogs);
    lind = 0;
    mlen = 0;
    for k=1:lnum
        tlen = length(HResLogs{k});
        if tlen > mlen
            mlen = tlen;
            lind = k;
        end
    end
    rinds = setdiff(1:lnum,lind);
    %% 极值约束
    for k=1:lnum
        tmedian = median(abs(HResLogs{k}));
        tmedian_l = median(abs(ConvLogs{k}));
        T = 3*tmedian;
        T_l = 3*tmedian_l;
        HResLogs{k} = max(HResLogs{k},-T);
        HResLogs{k} = min(HResLogs{k},T);
        ConvLogs{k} = max(ConvLogs{k},-T_l);
        ConvLogs{k} = min(ConvLogs{k},T_l);
        HResLogs{k} = OutlierRemove(HResLogs{k},3);
        ConvLogs{k} = OutlierRemove(ConvLogs{k},3);
    end
    %% 对齐到最多数据数据井
    for k=1:lnum-1
        HResLogs{rinds(k)} = AlignData_GD(HResLogs{lind},HResLogs{rinds(k)});
        ConvLogs{rinds(k)} = AlignData_GD(ConvLogs{lind},ConvLogs{rinds(k)});
    end 
    OutParams.alignParams = [mean(ConvLogs{lind}) std(ConvLogs{lind}) min(ConvLogs{lind}) max(ConvLogs{lind})];
    for k=1:lnum
        tDepths = HDepths{k};
        tLog = HResLogs{k};
        tData(:,1) = tDepths;
        tData(:,2) = tLog;
        rData = Resampling_v(tData,1/512);   
        xx = min(rData(:,1)):1/8:max(rData(:,1));
        ttData(:,1) = CDepths{k};
        ttData(:,2) = ConvLogs{k};
        ConvLogs{k} = Resampling_v1(ttData,xx);
        HResLogs512{k} = rData;
        clear tData rData tLog tDepths ttData
    end                   
    HHResLogs512 = [];
    for k=1:lnum
        HHResLogs512 = [HHResLogs512;HResLogs512{k}];
    end
    clear HResLogs;
    HHResLogs512(:,1) = 1:length(HHResLogs512);
    mn = min(HHResLogs512(:,2));
    mx = max(HHResLogs512(:,2));
    HHResLogs512(:,2) = (HHResLogs512(:,2) - mn)/(mx-mn);
    %设计低分辨曲线虚拟序号坐标    
    for k=1:lnum
        ConvLogs{k}(:,2) = Normalize_MinMax(ConvLogs{k}(:,2));       
    end
    LResLogs = [];
    LResInds = [];
    for k=1:lnum
        LResLogs = [LResLogs;ConvLogs{k}(:,2)];
        if k == 1
            LResInds = 1:64:(length(ConvLogs{k})-1)*64+1;
            EndInds(1) = length(LResLogs);
        else
            LResInds = [LResInds max(LResInds)+1:64:max(LResInds)+(length(ConvLogs{k})-1)*64+1];
            EndInds(k) = length(LResInds)- sum(EndInds(1:k-1));       
        end
    end
    LResData(:,1) = single(LResInds);
    LResData(:,2) = single(LResLogs);
    clear LResLogs LResInds;
    [LHResData512,RFMtrees_L2H] = MultiScaleSuperResolution_v_rf(HHResLogs512,LResData,EndInds,treenum,errT);
    T = single(LHResData512(:,2));
    for k=1:10
        T = medfilt1(T,3);
    end
    LHResData512(:,2) = single(T);
    OutParams.LHSuperModel = RFMtrees_L2H;
    clear HHResLogs512
    [~,LHResData512_H] = SigDecomp_LMD(LHResData512);
    LHResData512_H = single(LHResData512_H);    
    [RFMtrees_H] = UpSampling64times_RFM_train_H(LHResData512_H,treenum,errT);
    [RFMtrees] = UpSampling64times_RFM_train(LHResData512,treenum,errT);
%     OutParams.L2HSuperModel = RFMtrees_L2H;
    OutParams.HSuperModel = RFMtrees_H;
    OutParams.SuperModel = RFMtrees;
    OutParams.mode = 'test';
elseif strcmp(mode,'test')
    RFMtrees = options.SuperModel;
    RFMtrees_H = options.HSuperModel;
    w = options.W;
    alignParams = options.alignParams; % 包括模型井归一化参数：均值，方差，取值范围等
    [~,CDepths,ConvLogs] = ConvLogsReading_Test(LogName);
    CDepths = cell2mat(CDepths);
    ConvLogs = cell2mat(ConvLogs);
    tmedian = median(ConvLogs);
     T = 3*tmedian;    
     ConvLogs = max(ConvLogs,-T);
     ConvLogs = min(ConvLogs,T);
     ConvLogs = OutlierRemove(ConvLogs,3);
    ConvLogs = AlignData_GD_v1(ConvLogs,alignParams(1),alignParams(2),alignParams(3),alignParams(4));
    ConvLogs = Normalize_MinMax(ConvLogs);
    TData(:,1) = CDepths;
    TData(:,2) = ConvLogs;
    [~,TData_H] = SigDecomp_LMD(TData);
    SSig = ConventionalLogSuperResolution_RF(TData,RFMtrees,512);
    SSig_H = ConventionalLogSuperResolution_RF(TData_H,RFMtrees_H,512);
    SSig(:,2) = SSig(:,2) + 0.5*SSig_H(:,2);
    SSig(:,1) = CDepths(1):1/512:max(CDepths);
    T = SSig(:,2);
%     for k=1:10
%         T = medfilt1(T,3);
%     end
    SSig(:,2) = T;
    SSig = Resampling_v(SSig,0.001);
    SSig = FinalAdjust(SSig,TData,0.001);
    OutParams = SSig;    
end