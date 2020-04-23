library(data.table)
library(lubridate)

hw_param<-fread("/home/k2uxam/tda_sample/HullWhite.csv",sep=",")
hw_param$Dates<-as.Date(hw_param$Field,format = "%m/%d/%Y")
#
#select target variables
hw_param_df<-hw_param[,c("Dates","HW_a","HW_sigma")]

##########################################################################
# installing required packages
##########################################################################
if (!require(package = "FNN")) {
  install.packages(pkgs = "FNN")
}
if (!require(package = "igraph")) {
  install.packages(pkgs = "igraph")
}
if (!require(package = "scales")) {
  install.packages(pkgs = "scales")
}
##########################################################################
# installing R package TDA
##########################################################################
if (!require(package = "TDA")) {
  install.packages(pkgs = "TDA")
}
##########################################################################
# loading R package TDA
##########################################################################
library(package = "TDA")

#########################################################################
# distance function #Exp & functions
########################################################################
hw_param_mx<-as.matrix(hw_param_df[,2:3])
hw_param_scale_mx<-scale(hw_param_mx)

library(doParallel)
no_cores<-10
registerDoParallel(cores=no_cores)  
cl <- makeCluster(no_cores, type="FORK")  
hw_param_diag_list<-foreach(i = 1:(nrow(hw_param_scale_mx)-59)) %dopar% {
  hw_param_scale_mx_temp<-hw_param_scale_mx[i:(i+59),]
  hw_param_tmp<-ripsDiag(X=hw_param_scale_mx_temp,maxdimension = 2,maxscale = 5, library = c("GUDHI", "Dionysus"), location = TRUE, printProgress = FALSE)
  hw_param_tmp
}
saveRDS(hw_param_diag_list,"/home/k2uxam/tda_sample/hw_param_ripsdiag_list.rds")
stopCluster(cl)  
#2：16

# #step1: comparion between each diag distance
# #step2: check various bith values

dist_diags<-c()
for(i in 1:(length(hw_param_diag_list)-1)){
  dist_diags<-c(dist_diags,bottleneck(hw_param_diag_list[[i]]$diagram,hw_param_diag_list[[i+1]]$diagram))
}

dist_df<-hw_param_df[1:length(dist_diags),]
dist_df$dist_diags<-dist_diags

mst_result<-read.csv("/home/k2uxam/tda_sample/mst_result.csv",sep="|")
dist_df$Date_v<-as.numeric(paste0(substring(dist_df$Dates,1,4),substring(dist_df$Dates,6,7),substring(dist_df$Dates,9,10)))

data_clean<-merge(dist_df,mst_result,by="Date_v",all=FALSE)
write.csv(data_clean,"/home/k2uxam/tda_sample/data_clean.csv")
