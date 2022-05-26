% The code below calculate error metric for VAR model and SSTC model

% Store error metric for var model at each times(100 experiments)
    mae_var = zeros(1,100); 
    rmse_var = zeros(1,100);
  
% Store error metric for sstc model at each times(100 experiments) 
    mae_sstc = zeros(1,100);
    rmse_sstc = zeros(1,100);

    state = "nonstationary"; % set stationary or non-stationary case
    pre_cell_var = []; % store forecasting results for VAR model
    pre_cell_sstc = [];% ...SSTC model

    simulation_times = 100;
% baseline metric ; VAR model
for sim_n = 1:simulation_times
    sim_n
    % read data
    file_name = "s" + int2str(sim_n-1)+".csv";
    simdatastationary = csvread("D:/ADD DATA PATHWAY"+file_name,1,1); % add data pathway
    md1 = arima(1,0,0);
    MD1 = varm(30,1);
    data = simdatastationary';
    if state == "nonstationary"
        data_diff(2:500,:) = data(2:500,:) - data(1:499,:); % make difference
        data_diff(1,:) = 0;
    end
    
    if state == 'stationary'
        estmd1 = estimate(MD1,data(1:300,:));
        pre = forecast(estmd1,200,data(300,:));
    end
    estmd1 = estimate(MD1,data_diff(1:300,:));
    pre = forecast(estmd1,200,data_diff(300,:));
    if state == 'nonstationary'
        pre = cumsum(pre);
    end
    for i =1:30
        mae(i) = errperf(data(301:500,i), pre(:,i), 'mae');
        rmse(i) = errperf(data(301:500,i), pre(:,i), 'rmse');
    end

    mae_var(1,sim_n) = mean(mae);
    rmse_var(1,sim_n) = mean(rmse);
    pre_cell_var(:,:,sim_n) = pre;

    % SSTC model 
     x_tile = simdatastationary(:,1:300);
     coords = spatial_domain;

    g{1} = 1:30;
    
    W = zeros(30,30);
    for i = 2:size(x_tile,1)
         for j = 1:length(g)
           if ismember(i,g{j})==1
               group = g{j}(g{j}~=i);
           end
        end
        dist = sqrt(sum((coords(i,:) - coords(group,:)).^2,2));
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
    h = 200;
    y_f = zeros(h,30);
    y_f(1,:) = (inv(S))*(c + results.theta(1)*x_tile(:,end) + results.theta(2).*W*x_tile(:,end) + results.theta(3)*x_tile(:,end) + results.theta(4).*W*x_tile(:,end));
    y_f(2,:) = (inv(S))*(c + results.theta(1)*y_f(1,:)' + results.theta(2).*W*y_f(1,:)' + results.theta(3)*x_tile(:,end) + results.theta(4).*W*x_tile(:,end) );
    for i = 3:h
        y_f(i,:) = (inv(S))*(c + results.theta(1)*y_f(i-1,:).' + results.theta(2).*W*y_f(i-1,:).' + results.theta(3)*y_f(i-2,:).' + results.theta(4).*W*y_f(i-2,:).' );
    end
    
    for i =1:30
        mae(i) = errperf(data(301:500,i), y_f(:,i), 'mae');
        rmse(i) = errperf(data(301:500,i), y_f(:,i), 'rmse');
    end
    mae_sstc(1,sim_n) = mean(mae);
    rmse_sstc(1,sim_n) = mean(rmse);
    pre_cell_sstc(:,:,sim_n) = y_f;
    
end
