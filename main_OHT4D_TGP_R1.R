library(tgp)
library(tictoc)
library(R.matlab)

S <- 50;
inputFile = paste('result_OHT_4d.mat',sep="")
dat <- readMat(inputFile)

# seed data size
start <- 20*4
d  <- 4

XX <- dat$xt
yt <- dat$yt
yt <- array(yt, dim=c(length(yt)))
X  <- dat$x.AL[[1]]
X  <- array(unlist(X),dim=c(length(unlist(X))/d,d))
X  <- X[1:start, ]
y  <- dat$y.AL[[1]]
y  <- array(unlist(y),dim=c(length(unlist(y)),1))
y  <- y[1:start, ]
xc <- dat$xc
yc <- dat$yc

Nc <- 500
TNC <- nrow(xc)

Nt <- nrow(XX)

end <- start+S

tic()

MSE <- matrix(0,end-start+1,ncol=nrow(XX))
RMSE <- matrix(0,end-start+1,ncol=nrow(XX))
NLPD <- matrix(0,end-start+1,ncol=nrow(XX))
CRPS <- matrix(0,end-start+1,ncol=nrow(XX))

for(t in start:end){
  
  xc_ind = order(runif(TNC));
  xc_s = xc[xc_ind[1:Nc], ];
  yc_s = yc[xc_ind[1:Nc], ];
  
  obj <- btgp(X=X, Z=y, XX=rbind(xc_s, XX), corr="expsep", Ds2x=TRUE)
  ypred <- obj$ZZ.km[(Nc+1):(Nc+Nt)]
  ys2   <- obj$ZZ.ks2[(Nc+1):(Nc+Nt)]
  
  #MSE[t-start+1] <- mean((ypred-yt)^2)
  #RMSE[t-start+1] <- mean(abs(ypred-yt))
  #NLPD[t-start+1] <- mean( (ypred-yt)^2/(2*ys2) + log(sqrt(2*pi*ys2)))
  
  MSE[t-start+1,] <- ((ypred-yt)^2)
  RMSE[t-start+1,] <- (abs(ypred-yt))
  NLPD[t-start+1,] <- ((ypred-yt)^2/(2*ys2) + log(sqrt(2*pi*ys2)))
  sig = sqrt(ys2)
  scores = (yt-ypred)/sig;
  CRPS[t-start+1,] <- sig * (1/sqrt(pi) - 2*dnorm(scores) - scores*(2*pnorm(scores)-1))
  
  ## extract via ALM, ALC, EI-prec
  al <- obj$Ds2x[1:Nc]
  m <- which.max(al)
  
  xstar <- xc[m,]
  ystar <- yc[m]
  
  ## update the fit for the next round
  X <- rbind(X, xstar)
  y <- c(y, ystar)
  
  outputFile_MSE = paste('tgp_OHT_result_MSE.csv',sep="")
  outputFile_RMSE = paste('tgp_OHT_result_RMSE.csv',sep="")
  outputFile_NLPD = paste('tgp_OHT_result_NLPD.csv',sep="")
  outputFile_CRPS = paste('tgp_OHT_result_CRPS.csv',sep="")
  outputFile_X = paste('tgp_OHT_result_X.csv',sep="")
  outputFile_y = paste('tgp_OHT_result_y.csv',sep="")
  
  write.table(MSE, outputFile_MSE, sep=",",col.names=FALSE, row.names=FALSE)
  write.table(RMSE, outputFile_RMSE, sep=",",col.names=FALSE, row.names=FALSE)
  write.table(NLPD, outputFile_NLPD, sep=",",col.names=FALSE, row.names=FALSE)
  write.table(CRPS, outputFile_CRPS, sep=",",col.names=FALSE, row.names=FALSE)
  write.table(X, outputFile_X, sep=",",col.names=FALSE, row.names=FALSE)
  write.table(y, outputFile_y, sep=",",col.names=FALSE, row.names=FALSE)
  
  save.image("tgp_OHT_result_RData.RData")
  toc()
}

outputFile_MSE = paste('tgp_OHT_result_MSE.csv',sep="")
outputFile_RMSE = paste('tgp_OHT_result_RMSE.csv',sep="")
outputFile_NLPD = paste('tgp_OHT_result_NLPD.csv',sep="")
outputFile_CRPS = paste('tgp_OHT_result_CRPS.csv',sep="")
outputFile_X = paste('tgp_OHT_result_X.csv',sep="")
outputFile_y = paste('tgp_OHT_result_y.csv',sep="")

write.table(MSE, outputFile_MSE, sep=",",col.names=FALSE, row.names=FALSE)
write.table(RMSE, outputFile_RMSE, sep=",",col.names=FALSE, row.names=FALSE)
write.table(NLPD, outputFile_NLPD, sep=",",col.names=FALSE, row.names=FALSE)
write.table(CRPS, outputFile_CRPS, sep=",",col.names=FALSE, row.names=FALSE)
write.table(X, outputFile_X, sep=",",col.names=FALSE, row.names=FALSE)
write.table(y, outputFile_y, sep=",",col.names=FALSE, row.names=FALSE)
save.image("tgp_OHT_result_RData.RData")
