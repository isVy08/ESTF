library(R.matlab)
library(vars)
library(forecast)
library(tseries)
library(fda)
library(urca)
library(matlib)

path = ("D:/programming/my project/SARC_final")
file_name = file.path(path,"Mine_data(2).mat")
mine = readMat(file_name)
mine = mine$mine


find_distance = function(x,y){
  return(sqrt((x[1]-y[1])^2+(x[2]-y[2])^2))
}
n.index = c(465,759,769,259,396,1444,90,175,281,507,1662,1038,429,821,1315,63,941,999,929,800,1014,1174,394,1701,825,184,994,480,1555,41)
W = matrix(0,nrow = length(n.index),ncol = length(n.index))
for (i in 1:length(n.index)){
  for (j in 1:length(n.index)) {
    W[i,j] = 10^{-2}*log(find_distance(mine[n.index[i],2:3],mine[n.index[j],2:3])+1)
    
  }
}

x1 = runif(30,-0.01,0.01)
x1 = rnorm(30,0,0.1)
X = matrix(0,nrow = 30,ncol = 4000)
X[,1] = x1
for (t in 2:4000){
  X[,t] = W %*% (matrix(X[,t-1],ncol = 1))
}
plot(X[10,1:100])
#=============================================
# simulation for non-stationary case
library("matrixStats")
# specify spatial weight matrix using log shape function; alpha control magnitude of the function
generate_w = function(n.index,alpha){
  W = matrix(0,nrow = length(n.index),ncol = length(n.index))
  for (i in 1:length(n.index)){
    for (j in 1:length(n.index)) {
      W[i,j] = alpha*(1/log(find_distance(mine[n.index[i],2:3],mine[n.index[j],2:3])+1))
    }
    diag(W) = 0
    #W[i,] = W[i,] / sum(W[i,]) 
  }
  return(W)
}
# generate one-step delta x; initial_value is observation; W_list is spatial weight matrix; c is deterministic constant; e is noise
generate_delta_x = function(p,initial_value,W_list,c,e){
  non_stationary_term = W_list[[1]] %*% initial_value[,-1]
  n = dim(initial_value)[1]
  tem = matrix(0,nrow = n,ncol = 1)
  for(i in 1:p){
    tem = tem + W_list[[i+1]] %*% matrix(rowDiffs(initial_value)[,i],ncol = 1)
  }
  
  return(tem + non_stationary_term + c +e)
}
generate_sim_data = function(T,p,initial_value,alpha,c,e){
  W_list = list()
  for(i in 1:(p+1)){
    W_list[[i]] = generate_w(n.index,alpha[i])
    #W_list[[2]] = generate_w(n.index,0.01)
  }
  print(W_list[[1]])
  x = cbind(initial_value,initial_value[,ncol(initial_value)]+generate_delta_x(1,initial_value,W_list,c,e))
  for(i in 2:T){
    x = cbind(x,x[,ncol(x)]+generate_delta_x(1,x[,(ncol(x)-1):ncol(x)],W_list,c,e))
  }
  return(x)
}

initial_value = matrix(0,nrow = 30,ncol = 2)
initial_value[,1] = runif(30,-0.001,0.001)
initial_value[,2] = runif(30,-0.001,0.001)
alpha = c(0.0008,0.0006)
c = matrix(runif(30,0,0.0002),nrow = 30,1)
e = matrix(rnorm(30,0,0.0002),nrow = 30,1)
#c = matrix(0,nrow = 30,1)
#e = matrix(0,nrow = 30,1)
x=generate_sim_data(2000,1,initial_value,alpha,c,e)
#x = t(apply(delta_x,1,cumsum))
plot(x[20,])

'''
alpha = c(0.001,0.001)
c = matrix(runif(30,0,0.0002),nrow = 30,1)
e = matrix(rnorm(30,0,0.0002),nrow = 30,1)
x1=generate_sim_data(1000,1,x[,(ncol(x)-1):ncol(x)],alpha,c,e)
#x = t(apply(delta_x,1,cumsum))
#plot(x[10,])
#x=cbind(x,t(apply(delta_x,1,cumsum)))
x = cbind(x,x1)
plot(x[10,])

alpha = c(0.06,0.04)
c = matrix(runif(30,0,0.06),nrow = 30,1)
e = matrix(rnorm(30,0,0.06),nrow = 30,1)
x=cbind(x,generate_sim_data(300,1,x[,(ncol(x)-1):ncol(x)],alpha,c,e))
plot(x[2,])
write.csv(x,"sim_data_nonstationary.csv")
'''
#===================================================================
# Simulation for non-stationary case; time-varing spatial weight matrix

generate_sim = function(initial_value,alpha,mu_t,epsilon_t,T){
  #initial_value is observations starting from time0; n*p matrix and p is time lag
  #alpha is p*1 vector(control p shape functions) that forms p spatial weight matrix
  #mu_t is n*(p+1) matrix
  #epsilon_t is noise matrix
  p = dim(initial_value)[2]
  n = dim(initial_value)[1]
  W_list = list()
  for(i in 1:p){
    W_list[[i]] = generate_w(n.index,alpha[i])
  }
  x = cbind(initial_value)
  for(t in (p+1):T){
    x = cbind(x,generate_one_step(matrix(x[,((ncol(x)+1-p)):ncol(x)],ncol = p), W_list, mu_t[,(t-p):t],epsilon_t[,t]))
  }
  return(x)
  
}
generate_one_step = function(initial_value,W_list,mu_t,epsilon_t){
  #initial_value is a n*p matrix and p is time lag, n is the number of locations
  #W_list is a p*1 list that contains spatial weight matrix
  #mu_t is n*(p+1) matrix
  #epsilon_t is n*1 noise matrix
  #The function generate one step value of X given previous observations
  n = dim(initial_value)[1]
  p = dim(initial_value)[2]
  tem = matrix(0,nrow = n,ncol = 1)
  for (i in 1:p){
    tem = tem + W_list[[i]] %*% (initial_value[,p+1-i]-mu_t[,p+1-i])
  }
  tem = tem + mu_t[,p+1] + epsilon_t
  return(tem)
}
initial_value = matrix(runif(30,-1,1),nrow = 30,ncol = 1)

# create mu_t

mu_t = matrix(0,nrow = 30,ncol = 4000)
mu_coeff = runif(30,0.0001,0.001)
for(t in 1:4000){
  mu_t[,t] = mu_coeff * t^2
}
# create epsilon_t
epsilon_t = matrix(rnorm(30*4000,0,5),nrow = 30)
x = generate_sim(initial_value,alpha=0.001,mu_t,epsilon_t,1000)
plot(x[20,])
x1 = generate_sim(matrix(x[,ncol(x)],nrow = 30),alpha = 0.002,mu_t[,1001:ncol(mu_t)],epsilon_t[,1001:ncol(epsilon_t)],1000)
x = cbind(x,x1)
plot(x[20,])
x2 = generate_sim(matrix(x[,ncol(x)],nrow = 30),alpha = 0.004,mu_t[,2001:ncol(mu_t)],epsilon_t[,2001:ncol(epsilon_t)],1000)
x = cbind(x,x2)
plot(x[20,])
x3 = generate_sim(matrix(x[,ncol(x)],nrow = 30),alpha = 0.008,mu_t[,3001:ncol(mu_t)],epsilon_t[,3001:ncol(epsilon_t)],1000)
x = cbind(x,x3)
plot(x[2,])
write.csv(x,"time_varing_non_statiaonry_simdata.csv")