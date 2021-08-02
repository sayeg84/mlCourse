require(gtools) 
TABUMLR <- function(D,m) {
 

i<-c("Weight","Fglucose","GlucoseInt","InsulinResp","InsulineResist")
r1<-unlist(sapply(1:length(i),FUN = function(y) sapply(y:length(i), 
         FUN = function(x) paste("I(",i[y],"*",i[x],")",sep = "") )))
r<-c(i,r1)
#r<-i
CN<-colnames(D)
GF<-c()
modelo<-c()
l <- vector("list", 10)
n<-10
k<-5
t1<-split(1:33, sort(1:33%%k))
t2<-split(1:36, sort(1:36%%k))
t3<-split(1:76, sort(1:76%%k))
d<-apply(as.matrix(1:k),1,FUN=function(x) c(length(t1[[x]]),length(t2[[x]]),length(t3[[x]])))
t<-apply(as.matrix(1:k),1,FUN=function(x) c(33-length(t1[[x]]),36-length(t2[[x]]),76-length(t3[[x]])))
pl<-lapply(1:n,FUN=function(y) lapply(list(1:33,34:69,70:145),FUN=function(x) sample(x)))
rf<-apply(as.matrix(1:n), 1, FUN=function(y) apply(as.matrix(1:k), 1,
                   FUN=function(x) c(pl[[y]][[1]][t1[[x]]],pl[[y]][[2]][t2[[x]]],pl[[y]][[3]][t3[[x]]])))

for (j in 1:3){
  u<-combinations(n = length(r), j)
  
  evaluate <- function(binario){ 
    rows<-sum( sapply(1:length(binario),FUN = function(x) binario[x]*(2^( x-1 )) ) )
    if (1<=rows && rows<dim(u)[1]){
      Q<-as.formula(paste(c(CN[length(CN)]," ~",paste(r[u[rows,]],collapse = "+")),collapse=""))
      fit <- vglm(Q, multinomial(refLevel = 2), data = D)
      tr<-sum(Anova(fit, type="II",test.statistic="F")$`Pr(>F)`[1:j]<0.05)
      if (tr<j ){g<-0}
        else {
          if(m==1){
            g<-1/AIC(fit)
            }
          else if (m==2){
            g<-1/BIC(fit)
          } 
          else if (m==3){
            g<-deviance(fit)  
          }else if (m==4){
            
            g<-1/max(PredictionLR(D,Q,t1,t2,t3,d,t,pl,rf,n,k,"na","FALSE"))  
          }else{  
            g<-1/sum(PredictionLR(D,Q,t1,t2,t3,d,t,pl,rf,n,k,"na","FALSE"))
          }
        }
    }else {g<-0 }
    return(g)
  }  
  
  z<-length(binary(dim(u)[1]))
  res <- tabuSearch(size = z, iters = 50, objFunc = evaluate,
                    listSize = 3, nRestarts = 1,verbose=TRUE ) # matrix(1,1,z) 
  if(sum(res$eUtilityKeep)==0){
    print("No solution found")
    break}
    f<-which.max(res$eUtilityKeep)
    binario<-res$configKeep[f,]
    rows<-sum( sapply(1:length(binario),FUN = function(x) binario[x]*(2^( x-1 )) ) )
  
  Qf<-as.formula(paste(c(CN[length(CN)]," ~",paste(r[u[rows,]],collapse = "+")),collapse=""))

  fit <- vglm(Qf, multinomial(refLevel = 2), data = D)

  if(m==1){
    l[[k]] <-Qf
    GF<-c(GF,AIC(fit))
  }
  else if (m==2){
    l[[k]] <-Qf
    GF<-c(GF,BIC(fit))
  } 
  else if (m==3){
    l[[k]] <-Qf
    GF<-c(GF,deviance(fit))
  }else{  
    l[[k]] <-Qf
    GF<-c(GF,sum(PredictionLR(D,Qf,t1,t2,t3,d,t,pl,rf,n,k,"na","FALSE")))
  }
  
}

return(list(l,GF))

} 


binary<-function(p_number) {
  bsum<-c()
  bexp<-1
  while (p_number > 0) {
    digit<-p_number %% 2
    p_number<-floor(p_number / 2)
    bsum<-c(bsum, digit)
    bexp<-bexp * 10
  }
  return(bsum)
}