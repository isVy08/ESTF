% real case study
% Store error metric for var model
tic
    mae_var = zeros(1,1); 
    rmse_var = zeros(1,1);
  
% Store error metric for sstc model
    mae_sstc = zeros(1,1);
    rmse_sstc = zeros(1,1);

    state = "nonstationary"; % real data set is non-statioanry
    pre_cell_var = []; % store forecasting results for VAR model
    pre_cell_sstc = [];% ...SSTC model


% baseline metric ; VAR model

    % read data
    file_name = "airdata.csv" 
    realnonstationary = csvread("D:/ADD DATA PATHWAY"+file_name,0,0); % add data pathway
    md1 = arima(1,0,0);
    MD1 = varm(30,1);
    data = realnonstationary';
    if state == "nonstationary"
        data_diff(2:200,:) = data(5:203,:) - data(4:202,:); % make difference; 
        data_diff(1,:) = 0;
    end
    
    estmd1 = estimate(MD1,data_diff(1:200,:));
    pre = forecast(estmd1,166,data_diff(200,:));
    if state == 'nonstationary'
        pre = cumsum(pre);
    end
    for i =1:30
        mae(i) = errperf(data(204:369,i), pre(:,i), 'mae');
        rmse(i) = errperf(data(204:369,i), pre(:,i), 'rmse');
    end

    mae_var(1,1) = mean(mae);
    rmse_var(1,1) = mean(rmse);
    pre_cell_var(:,:,1) = pre;
toc
tic
    % SSTC model 
    index = [7,13,19,29,33,34,39,50,51,52,56,57,71,72,77,80,81,88,90,95,97,99,100,103,104,105,111,112,147,151]; % selected 30 locations
    x_tile = realnonstationary(:,4:203);
    coords = realcase_air(:,2:3); % transformed data in a matrix form;

    g{1} = 1:30;
    
    W = zeros(30,30);
    for i = 2:size(x_tile,1)
         for j = 1:length(g)
           if ismember(i,g{j})==1
               group = g{j}(g{j}~=i);
           end
        end
        dist = sqrt(sum((coords(index(i),:) - coords(index(group),:)).^2,2));
        W(i-1,group) = dist.^-1;
        W(i-1,:) = W(i-1,:)./sum(W(i-1,:));
    end
    
    info.tl = 2; info.stl = 2;info.ted = 1;info.rmin =-1;info.rmax=1;
    y = x_tile.';
    y = y(:);
    results = sar_sstc(y,[],W,info);
    S = (eye(30) - results.theta(5)*W);
    
    c = mean((x_tile - (results.theta(5)-0.0004)*W*x_tile  - (results.theta(1)*x_tile + results.theta(2).*W*x_tile + results.theta(3)*x_tile + results.theta(4).*W*x_tile ))')';
    % Forecasting Y_t = A Y_t-1 + S^-1 (dY_t-1 + c)
    h = 166;
    y_f = zeros(h,30);
    y_f(1,:) = (inv(S))*(c + results.theta(1)*x_tile(:,end) + results.theta(2).*W*x_tile(:,end) + results.theta(3)*x_tile(:,end) + results.theta(4).*W*x_tile(:,end));
    y_f(2,:) = (inv(S))*(c + results.theta(1)*y_f(1,:)' + results.theta(2).*W*y_f(1,:)' + results.theta(3)*x_tile(:,end) + results.theta(4).*W*x_tile(:,end) );
    for i = 3:h
        y_f(i,:) = (inv(S))*(c + results.theta(1)*y_f(i-1,:).' + results.theta(2).*W*y_f(i-1,:).' + results.theta(3)*y_f(i-2,:).' + results.theta(4).*W*y_f(i-2,:).' );
    end
    
    for i =1:30
        mae(i) = errperf(data(204:369,i), y_f(:,i), 'mae');
        rmse(i) = errperf(data(204:369,i), y_f(:,i), 'rmse');
    end
    mae_sstc(1,1) = mean(mae);
    rmse_sstc(1,1) = mean(rmse);
    pre_cell_sstc(:,:,1) = y_f;
  
toc