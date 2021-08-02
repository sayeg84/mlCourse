#Tarea-Examen 5-6: Selección de modelos y regularización
# Ejercicio 2
# Curso Avanzado de Estadística. Profa. Guillermina Eslava Gómez.
# Aldo Sayeg Pasos Trejo. Cesar Cossio Guerrero.
# Posgrado en Ciencias Matemáticas. Universidad Nacional Autónoma de México.

setwd("G:/Mi unidad/Classroom/SML, Maestría, 2020-2/Tarea 4 y 5")
rm(list = ls())

#Este código se divide en 4 secciones.

#1.... Carga y evaluación de los datos Riboflavin
#2.... Selección de modelos
#3.... Cálculo de errores
#4.... Funciones especiales qeu se utilizaron (correr primero)

# 1.... Carga y evaluación de los datos Riboflavin----------------------------------------

#LibrerÃs necesarias
library(hdi)
library(glmnet)

# Llamamos a la base de datos y la exploramos
data(riboflavin)
str(riboflavin)
x<-riboflavin$x
y<- riboflavin$y
# summary(riboflavin)
# head(x)
# dim(x)
# length(y)

#2.... Selección de modelos--------------------------------------------------------


#Modelo base o nulo:regresion lineal.

M[[1]]<-lm(cbind(y,x)$y~.^2,data=as.data.frame(cbind(y,x)))

# Selección del modelo 4,5, y 6 lasso, elasticnet y ridge
ErrorB<- EREGU(x,y,500) #Genera una tabla con todos los datos sobre los errores de los modelos que da glmnet
                        # 500 son el número de repeticiones.  

#Con las siguientes lineas se pueden conocer las variables de cada modelo. Pero no se agregan en el 
# trabajo por ser muchas y no tenemos clara su interpretación
# Mcv<- cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=1)
# rownames(predict(Mcv,type="coefficient",s=.6))[which(predict(Mcv,type="coefficient",s=.6)!=0)]



#3.... Cálculo de errores------------------------------------------------------------------------

#Con la función CVREgu se calculan los errores aparentes para el modelo seleccionado
#Por los cálculos de los errores contenidos en la varaible ErrorB.

ErrorA1<-CVRegu(x,y,500,5,ErrorB,'no aparente') #23 minutos con B=500 #0.2953210 0.2596251 0.2483960
ErrorA2<-CVRegu(x,y,500,5,ErrorB,'aparente') # 0.2953210 0.2596251 0.2483960
ErrorMN1<-CVRegu(x,y,500,5,ErrorB,'Modelo Nulo') # Este es el modelo nulo, se presentan ambos errores


#4....Funciones especiales que se utilizaron (correr primero)--------------------------------------------

#Con esta función se calculan errores por cross validation con B repeticiones
#Y se extraen los valores de lambda mínimo y 1se para 1 repetición y
#todos los valores de lambda 1se para B repeticiones. 
#Se realiza una gráfica de los errores para una repetición y para todas las B's.

EREGU <- function(x,y,B){
  
  
  #En esta parte se llena la matriz de errores para la selección de modelos, así como las graficas
  #de los errores por crossvalidation para la selección de un valor de lmabda.
  
  errB <-matrix(0,3,9)
  rownames(errB) <- c('Ridge','Elasticnet','Lasso')
  colnames(errB) <- c('mse.min','mse.1se','mse RCV lambda','lambda.min','lambda.1se','RCV lambda','df.min','df.1se','df.RCV lambda')
  
  alpha <- c(0,0.5,1)
  for (Nmod in 1:3){
    M <-  glmnet(x,y, alpha=alpha[Nmod])
    Mcv<- cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=alpha[Nmod])
    plot(Mcv)
    errB[Nmod,1]<- Mcv$cvm[which(Mcv$lambda==Mcv$lambda.min)]
    errB[Nmod,4]<- Mcv$lambda.min
    errB[Nmod,7]<- Mcv$nzero[which(Mcv$lambda==Mcv$lambda.min)]+1
    errB[Nmod,2]<- Mcv$cvm[which(Mcv$lambda==Mcv$lambda.1se)]
    errB[Nmod,5]<- Mcv$lambda.1se
    errB[Nmod,8]<- Mcv$nzero[which(Mcv$lambda==Mcv$lambda.1se)]+1
    s<- apply(as.matrix(1:B),1, FUN=function(z){  #Cross validation,B repeticiones, mse.
      r <-c(); cv <- cv.glmnet(x, y,type.measure="mse",nfolds=5,alpha=alpha[Nmod])
      r<-cbind(r,c( cv$cvm[which(cv$lambda==cv$lambda.1se)] , cv$lambda.1se,cv$nzero[which(cv$lambda==cv$lambda.1se)]+1)) })
    plot(log(s[2,]),s[1,],xlab = 'log(lambda)',ylab = 'MSE', main = paste0('Cross Validation con alpha= ',alpha[Nmod] ) )
    abline(h = s[1,][which.min(s[1,])],lty = 2)
    abline(v = log(s[2,])[which.min(s[1,])],lty = 2)
    errB[Nmod,3]<- s[1,][which.min(s[1,])]
    errB[Nmod,6]<- s[2,][which.min(s[1,])]
    errB[Nmod,9]<- s[3,][which.min(s[1,])]
  }
  
  
  return(errB)
  
} 

CVRegu <- function(x,y,B,K,Er,ANA){ # Funcion para realizar k-fold-cross-validation
  
  
  #En esta parte se llena la matriz de errores para el cálculo de crossvalidation
  #Tanto aparente como no aparente y para Lasso, Elasticnet y Ridge
  
  t<-split(1:dim(x)[1], sort(1:dim(x)[1]%%K))
  d<-apply(as.matrix(1:K),1,FUN=function(r) t[[r]])
  MSE<-c()
  alpha <- c(0,0.5,1)
  for (Nmod in 1:3){
    
    if(ANA=='no aparente'){ 
      MSE<-c(MSE,mean(apply(as.matrix(1:B),1,FUN=function(i){ 
        pl<-sample(1:dim(x)[1]) 
        mean(apply(as.matrix(1:K),1, 
                   FUN=function(z) {dgx<-x[-pl[d[[z]]],]; dhx<-x[pl[d[[z]]],]
                   dgy<-y[-pl[d[[z]]]]; dhy<-y[pl[d[[z]]]]
                   mean((predict(cv.glmnet(dgx, dgy,type.measure="mse",nfolds=5,alpha=alpha[Nmod],
                                           lambda = seq(Er[Nmod,4],Er[Nmod,5],length=20) ),
                                 s=Er[Nmod,6],newx = dhx)-dhy)^2)
                   }))
      })))
    } else if(ANA=='aparente'){
      MSE<-c(MSE,mean(apply(as.matrix(1:B),1,FUN=function(i){ 
        pl<-sample(1:dim(x)[1]) 
        mean(apply(as.matrix(1:K),1, 
                   FUN=function(z) {dgx<-x[-pl[d[[z]]],]; dgy<-y[-pl[d[[z]]]]
                   mean((predict(cv.glmnet(dgx, dgy,type.measure="mse",nfolds=5,alpha=alpha[Nmod],
                                           lambda = seq(Er[Nmod,4],Er[Nmod,5],length=20) ),
                                 s=Er[Nmod,6],newx = dgx)-dgy)^2)
                   }))
      })))
    } else {#Modelo Nulo
      
      MSE<-c(MSE,mean(apply(as.matrix(1:B),1,FUN=function(i){ #Error no aparente
        pl<-sample(1:dim(x)[1]) 
        mean(apply(as.matrix(1:K),1, 
                   FUN=function(z) mean( ( mean(y[-pl[d[[z]]]]) - y[pl[d[[z]]]] )^2  ) ))
      })))
      
      MSE<-c(MSE,mean(apply(as.matrix(1:B),1,FUN=function(i){ #Error aparente
        pl<-sample(1:dim(x)[1]) 
        mean(apply(as.matrix(1:K),1, 
                   FUN=function(z) mean( ( mean(y[-pl[d[[z]]]]) - y[-pl[d[[z]]]] )^2  ) ))
      })))
      return(MSE) 
    }
    
    
  }
  
  return(MSE) 
} 
