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
generate_w = function(n.index,alpha,d_cutoff){
  W = matrix(0,nrow = length(n.index),ncol = length(n.index))
  for (i in 1:length(n.index)){
    for (j in 1:length(n.index)) {
      W[i,j] = alpha*(-log(find_distance(mine[i,2:3],mine[j,2:3])+1)+log(d_cutoff))
    }
    diag(W) = 0
  }
  return(W)
}


path = "D:/OneDrive - The University of Melbourne/my_project/deeplearning for ST/csv/stationary/"

for (i in 0:99) {
  file_ = paste0('s',as.character(i))
  file_ = paste0(file_,'.csv',seq = "")
  file_name = file.path(path,file_)
  
  
  x1 = runif(30,-0.01,0.01) # 
  X = matrix(0,nrow = 30,ncol = 500)
  X[,1] = x1
  
  alpha = runif(1,0.005,0.009)
  d_cutoff =200
  
  W = generate_w(n.index,alpha,d_cutoff)
  epsilon_t = matrix(rnorm(30*500,0,1),nrow = 30)
  
  for (t in 2:500){
    X[,t] = W %*% (matrix(X[,t-1],ncol = 1))+epsilon_t[,t]
  }
  
  write.csv(X,file_name)
}

'''
x1 = runif(30,-0.01,0.01) # 
X = matrix(0,nrow = 30,ncol = 500)
X[,1] = x1

alpha = runif(1,0.005,0.009)
d_cutoff =200

W = generate_w(n.index,alpha,d_cutoff)
epsilon_t = matrix(rnorm(30*500,0,1),nrow = 30)

for (t in 2:500){
  X[,t] = W %*% (matrix(X[,t-1],ncol = 1))+epsilon_t[,t]
}
plot(X[30,],type = 'l')