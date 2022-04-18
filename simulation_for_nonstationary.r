library(R.matlab)
library(vars)
library("matrixStats")
set.seed(18)
options(warn=-1) 
# calculate distance for given coordinates
find_distance = function(x,y){
  return(sqrt((x[1]-y[1])^2+(x[2]-y[2])^2))
}
n.index = c(465,759,769,259,396,1444,90,175,281,507,1662,1038,429,821,1315,63,941,999,929,800,1014,1174,394,1701,825,184,994,480,1555,41)

# read mine data
path = ("D:/programming/my project/SARC_final")
file_name = file.path(path,"Mine_data(2).mat")
mine = readMat(file_name)
mine = mine$mine

# function to creat spatial weight matrix;
generate_w = function(n.index,alpha){
  W = matrix(0,nrow = length(n.index),ncol = length(n.index))
  for (i in 1:length(n.index)){
    for (j in 1:length(n.index)) {
      W[i,j] = exp(-(alpha*find_distance(mine[n.index[i],2:3],mine[n.index[j],2:3])^2-400*alpha*find_distance(mine[n.index[i],2:3],mine[n.index[j],2:3])))
    }
    diag(W) = 0
    W[i,] = W[i,] / sum(W[i,]) # here I use normalization to avoid explosive larger value of WX.
  }
  return(W)
}

# function on generate simulated data at one step given previous value, spatial weight matrix, mean function and noise
generate_one_step = function(x.t_1, W.t_1,epsilon.t_t){
  n = dim(W.t_1)[1]
  x.t = matrix(0,nrow = n,ncol = 1)
  x.t = W.t_1 %*% x.t_1  + epsilon.t_t  # this is the form of our non-stationary model
  return(x.t)
}

# The function aims to simulate all data. It iteratively apply generate_one_step function
generate_sim = function(x_1,alpha.bound,T,epsilon){
  x = cbind(x_1)
  n = dim(x_1)[1]
  for(t in 2:T){
    alpha.t_1 = (1 - (t-1)/T)*alpha.bound[1]+(t-1)/T*alpha.bound[2]
    W.t_1 = generate_w(n.index,alpha.t_1)
    tem = matrix(0,nrow = n,ncol = 1)
    tem = generate_one_step(matrix(x[,ncol(x)],nrow = n),W.t_1,epsilon[,t])
    x = cbind(x,tem)
  }
  return(x)
}


path = "D:/OneDrive - The University of Melbourne/my_project/deeplearning for ST/csv/nonstationary/"

for (i in 0:99) {
  file_ = paste0('s',as.character(i))
  file_ = paste0(file_,'.csv',seq = "")
  file_name = file.path(path,file_)
  
  # create epsilon_t
  epsilon_t = matrix(rnorm(30*4000,0,1),nrow = 30)
  # initial x_1 at time=1
  initial_value = matrix(runif(30,-1,1),nrow = 30,ncol = 1)
  alpha.bound = -runif(1,1,9)*(10^-5)
  x = generate_sim(initial_value,c(alpha.bound,10*alpha.bound),500,epsilon_t)
  
  write.csv(x,file_name)
}
plot(x[30,],type = 'l')

