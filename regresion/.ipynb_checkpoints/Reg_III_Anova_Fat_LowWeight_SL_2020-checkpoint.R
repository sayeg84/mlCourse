 # One- and Twoway Analysis of Variance

getwd()
options(digits=4, width=120)
#Example 1.  Experimental data on fat absorbsion

grams<- c(64,72,68,77,56,95, 78, 91, 97, 82, 85, 77,75,93,78,71,63,76,55,66,49,64,70,68)  
fattype<- c("A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C", "C", "D", "D", "D", "D", "D", "D")

Fatdata<- data.frame(factor(fattype), grams)
Fatdata
str(Fatdata)
attach(Fatdata)

by(grams, fattype, summary)


par(mfrow=c(1,2)); par(mar= c(4,4,2,1.5))
boxplot(grams ~ fattype, xlab='Fat type', ylab='Abserbed fat', las=1, col=2:5)
points(1:4, means, pch = 23, cex = 0.95, bg = "red")
stripchart(grams ~ fattype, xlab='Fat type', ylab='Abserbed fat', vertical=TRUE,cex=1.2,pch=16, las=1, col=2:5)

model1 <- lm(grams ~ fattype)
anova(model1)
summary(model1)
#Alternatively
#summary(aov(grams ~ fattype))

#change of reference group
newfattype<- relevel(factor(fattype), ref="D")

#Fatdata$newfattype <- newfattype
model1D<-lm(grams~newfattype)
anova(model1D)
summary(model1D)

par(mfrow=c(2,2))
plot(model1, which=1:4)

model1B <- lm(grams ~ fattype - 1)
summary(model1B)

#Example 2. Birth weight

lbw <- read.delim("lbw.txt")

#Attach data set and see first and last rows in the dataset.
attach(lbw);
ls()
head(lbw); tail(lbw)

str(lbw)

lbw$smoke<-factor(smoke)
lbw$bwt<- as.numeric(lbw$bwt)

str(lbw)

#summary(lbw)

by(lbw$bwt, lbw$race, summary)

par(mfrow=c(1,2)); par(mar= c(4,4,2,1.5))
boxplot(bwt ~ race, data =lbw, xlab='Race', ylab='Birth weight', las=1, col=2:4)

stripchart(bwt~race, data=lbw, vertical=TRUE, method="jitter", xlab="Race", 
           ylab="Birth weight",cex=1.2,pch=16, las=1, col=2:4)

lbw$race<-as.factor(lbw$race)

model1<-lm(bwt ~ race, data = lbw)

anova(model1)
drop1(model1, test="F") #alternative extraction of the F-test
summary(model1)

model1B<-lm(bwt ~ race - 1, data = lbw)

summary(model1B)

confint(model1, level=.95)
confint(model1B, level=.95)

par(mfrow=c(2,2))

plot(model1, which=1:4)

#Two-way Analysis of Variance

lbw$smoke<-as.factor(lbw$smoke)

par(mfrow=c(1,2)); par(mar= c(4,4,2,1.5))
boxplot(bwt ~ smoke*race, data = lbw, xlab='Race and Smoke', ylab='Birth weight', las=1, col=2:4)
stripchart(bwt ~ smoke*race, data = lbw, vertical=TRUE, method="jitter", xlab='Race and Smoke', ylab='Birth weight', las=1, col=2:4)

interaction.plot(lbw$race,lbw$smoke,lbw$bwt, fun=mean, type=c("b"), ylab="Birth weight", las=1,lwd=2, trace.label="Smoke",xlab="Race")

#Model with interaction

model2<-lm(bwt ~ race + smoke + race:smoke, data = lbw)
#model2<-lm(bwt ~ race*smoke, data = lbw) #idem
anova(model2)
drop1(model2, test="F")
summary(model2)

#Same model different parametrization

#model2B<-lm(bwt ~ race + smoke + race:smoke -1, data = lbw)
#summary(model2B)
#anova(model2)

#Model without interaction

model3 <- lm(bwt ~ race + smoke, data = lbw)
anova(model3)
drop1(model3, test="F")
summary(model3)

model3B <- lm(bwt ~ race + smoke -1, data = lbw)
anova(model3B)
drop1(model3B, test="F")
summary(model3B)
confint(model3B)

#Residuals an fitted values the same for any paramatrization of the same model, e.g. model3 and model3B.
par(mfrow=c(1,2)); par(mar= c(4,4,2,1.5))
plot(model3)


plot(model3, which=1:4)

#End

