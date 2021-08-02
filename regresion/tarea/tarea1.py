
# ## Librerias y definiciones necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api  as smf
import os
if not(os.path.isdir("tarea")):
    os.mkdir("tarea")
    
def cleanVarName(var):
    if len(var) > 1:
        if var[0]=="C" and var[1]=="(":
            if len(var.split("[")) > 1:
                return var.split(",")[0][2:] + " [" + var.split("[")[1]
            else:
                return var.split(",")[0][2:] 
        elif len(var.split(":")) > 1:
            fin = ""
            for v in var.split(":"):
                fin += " : " + v 
            return fin
        else:
            return var
    else:
        return var
def significanceTable(fit):
    df = pd.DataFrame(columns = ["R-squared",
                                 "AIC",
                                 "BIC",
                                 "Log-Likelihood",
                                "F-statistic",
                                "Prob (F-statistic)"])
    df.loc[0]=[fit.rsquared,
    fit.aic,
    fit.bic,
    fit.llf,
    fit.fvalue,
    fit.f_pvalue]
    return df
def coefficientTable(fit):
    df = fit.summary2().tables[1]
    df = df.rename(columns={"P>|t|":"$P (> |t|)$"})
    
    df.index = [cleanVarName(var) for var in df.index]
    return df
def anovaTable(fit):
    df = sm.stats.anova_lm(fit)
    df = df.rename(columns={"PR(>F)":"$P (> F$)"})
    df.index = [cleanVarName(var) for var in df.index]
    df.index.name = "Variable"
    return df 
def comparisonTable(fits):
    dfs = [significanceTable(fit) for fit in fits]
    df = dfs[0]
    for i in range(1,len(dfs)):
        df.loc[i] = dfs[i].iloc[0]
    formulas = [fit.model.formula for fit in fits]
    df.insert(loc=0,column="Model",value=formulas)
    return df


# ## Ejercicio 1. Problema 10
df = pd.read_csv("carseats.csv",header=0)
df.head().to_latex("tarea/1-data.tex")

fit = smf.ols(formula="Sales ~ Price + C(Urban) + C(US)",
              data=df).fit()
print(fit.summary())

coefficientTable(fit)

coefficientTable(fit).to_latex(buf="tarea/1-mod1.tex")

fit2 = smf.ols(
    formula="Sales ~ Price + C(US)",
    data=df).fit()
print(fit2.summary())

coefficientTable(fit2).to_latex(buf="tarea/1-mod2.tex")
comparisonTable([fit,fit2]).to_latex(buf="tarea/1-modCompar.tex")

abs(fit.rsquared - fit2.rsquared)

print(fit2.conf_int(alpha = 0.05))

a=0.5
fig = plt.figure(figsize=(6,4))
plt.scatter(fit2.predict(),fit2.resid.tolist(),alpha = a)
plt.ylabel("Residuals")
plt.xlabel("Fitted Sales")
plt.title("Residual Plot")
plt.grid()
plt.tight_layout()
plt.savefig("tarea/1-resplot.pdf")
plt.show()

fig = plt.figure(figsize=(10,4))
n = len(fit2.resid.tolist())
plt.scatter(range(n),fit2.resid.tolist(), alpha = a)
plt.plot([0,n],[0,0],c="k",lw=3,ls="--", alpha = a)
plt.ylabel("Residuals")
plt.xlabel("Index")
plt.title("Residual Series")
plt.grid()
plt.tight_layout()
plt.savefig("tarea/1-restimeplot.pdf")
plt.show()

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_ccpr_grid(fit2,fig=fig)
for ax in fig.axes:
    ax.grid()
fig.suptitle("Component-Component Residuals",x=0.5,y=1.05,fontsize=20,ha="center")
plt.tight_layout()
fig.savefig("tarea/1-compcomp.pdf",bbox_inches="tight")

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(fit2,"Price",fig=fig)
for ax in fig.axes:
    ax.grid()
plt.tight_layout()
plt.savefig("tarea/1-regdiag.pdf")

fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(fit2, ax=ax)
plt.grid()
plt.tight_layout()
plt.savefig("tarea/1-influence.pdf")
plt.show()

sm.graphics.qqplot(fit2.resid)
plt.title("Q-Q plot of predicted values")
plt.grid()
plt.tight_layout()
plt.savefig("tarea/1-qqplot.pdf")
plt.show()

df.iloc[[42,174,165],:].to_latex(buf="tarea/1-outliers.tex")


# ## Ejercicio 1. Problema 14
np.random.seed(1)
x1 = np.random.uniform(size=100)
x2 = 0.5*x1 + np.random.randn(100)/10
y = 2 + 2*x1 +0.3*x2 + np.random.randn(100)
df = pd.DataFrame()
df["x1"] = x1
df["x2"] = x2
df["y"] = y
df.head().to_latex(buf="tarea/2-data.tex")
print(df.head())

df.plot.scatter(x="x1",y="x2")
plt.grid()
plt.title(u"Correlación entre x1 y x2")
plt.savefig("tarea/2-corr.pdf")
cor = np.corrcoef(x1,x2)[0][1]
plt.text(0.8,0.0,"R^2={0}".format(round(cor,5)),
         va = "top",
         ha = "center",
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig("tarea/2-corr.pdf")
plt.show()

formulas = ["y ~ x1 +x2", "y ~ x1", "y ~x2"]
fits = [ smf.ols(formula = form , data=df).fit() for form in formulas] 
for fit in fits:
    print(fit.summary())
    print()
    print()
    print(fit.pvalues)
    print()
    print()
    coefficientTable(fit).to_latex(buf = "tarea/2-mod{0}.tex".format(fits.index(fit)+1))

comparisonTable(fits).to_latex(buf="tarea/2-modComp.tex",index=False)

df.loc[100] = [0.1,0.8,0.6]
df.tail()

df.iloc[1:100].plot.scatter(x="x1",y="x2",label="data")
plt.scatter([0.1],[0.8],c="C1",label="New point")
plt.legend()
plt.grid()
plt.title("Plano x-y")
plt.tight_layout()
plt.savefig("tarea/2-newData.pdf")
plt.show()

fit.resid

fits = [ smf.ols(formula = form , data=df).fit() for form in formulas] 
for fit in fits:
    print(fit.summary())
    print()
    print()
    print(fit.pvalues)
    print()
    print()
    print( fit.resid[100] / ( fit.resid.std() / np.sqrt(101) ) )
    coefficientTable(fit).to_latex(buf = "tarea/2-newMod{0}.tex".format(fits.index(fit)+1))


# ## Problema 2
df = pd.read_csv("June_13_data.csv",header=0)
df=df.astype({'year': 'object',"Month":"object","Time_of_Day":"object"})
df

par = ["p{2cm}" for i in range(10)]
colFormat = "ccccc"
for p in par:
    colFormat += p
df.head().to_latex(buf="tarea/3-data.tex",column_format=colFormat)

cols = df.columns.tolist()
for col in cols:
    if col != "Crash_Score":
        nb=30
        fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(12,6))
        df.boxplot(column="Crash_Score",
                           by=col, 
                           fontsize=16,
                           figsize=(10,10),
                           rot = 45,
                           ax = ax[0]
        )
        for value in set(df[col]):
            df[df[col]==value]["Crash_Score"].hist(alpha=0.5,
                                                   ax=ax[1],
                                                   label=str(value),
                                                   bins = nb,
                                                   density=True
            )
        ax[1].legend()
        ax[1].set_title("Histogram")
        plt.tight_layout()
        plt.savefig("tarea/3-{0}Hist.pdf".format(col))
        plt.show()
        plt.close()
        """
        hist = df.hist(column="Crash_Score",
                        by=col, 
                        figsize=(8,8),
                        bins=nb,
                        sharex=True,
                        sharey=True,
                        density = True,
        )
        for a in hist.flat:
            a.grid()
        plt.tight_layout()
        plt.show()
        plt.close() 
        """

df1 = df[["Crash_Score","year","Month","Time_of_Day"]]
df1["time"]=(df1["year"]-df1["year"].min())*12*6 + (df1["Month"]-1)*6 + df["Time_of_Day"]-1
df1.plot.scatter(y="Crash_Score",x="time",figsize=(14,6),alpha=0.3)
times = list(set(df1["time"]))
cs_avg = []
for t in times:
    avg = df1[df1["time"]==t]["Crash_Score"].mean()
    cs_avg.append(avg)
plt.plot(times,cs_avg,color = "C1",alpha=0.8)
plt.grid()
plt.title("Serie de tiempo",fontsize=24)
plt.xlabel("Tiempo",fontsize=20)
plt.ylabel("Crash_Score",fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig("tarea/3-timeplot.pdf")
plt.show()

refers = {
    "year":"2014",
    "Month":"10",
    "Time_of_Day":"4",
    "Rd_Feature":"NONE",
    "Rd_Character":"STRAIGHT-LEVEL",
    "Rd_Class":"OTHER",
    "Rd_Configuration":"TWO-WAY-UNPROTECTED-MEDIAN",
    "Rd_Surface":"SMOOTH ASPHALT",
    "Rd_Conditions":"OTHER",
    "Light":"DAYLIGHT",
    "Weather":"CLEAR",
    "Traffic_Control":"NONE",
    "Work_Area":"NO"
}

cols = df.columns.tolist()
model = "Crash_Score ~ "
for col in cols:
    if col != "Crash_Score":
        if col == "year" or col =="Month" or col=="Time_of_Day":
            model+=(" + C({0},Treatment({1}))".format(col,refers[col]))
        else:
            model+=(" + C({0},Treatment('{1}'))".format(col,refers[col]))
        

print("Model to try:")
print()
print(model)

models = []
mod = smf.ols(formula=model,data=df)
models.append(mod.fit())

print(models[-1].summary().tables[0])
print(anovaTable(models[-1]))

import scipy.stats as st

st.boxcox_normplot(df["Crash_Score"],-3.0,4.0,plot=plt)
plt.grid()
plt.tight_layout()
plt.savefig("tarea/3-boxcox.pdf")
plt.show()

st.boxcox_normmax(df["Crash_Score"],brack=(-3.0,4.0))

lmax = st.boxcox(df["Crash_Score"])[1]
df1 = df.copy()
df1["Crash_Score"] = st.boxcox(df1["Crash_Score"],lmax)
df1.rename(columns={'Crash_Score':'Crash_Score_boxcox_0_27'}, inplace=True)

cols = df1.columns.tolist()
model = "Crash_Score_boxcox_0_27 ~ "
for col in cols:
    if col != "Crash_Score_boxcox_0_27":
        if col == "year" or col =="Month" or col=="Time_of_Day":
            model+=(" + C({0},Treatment({1}))".format(col,refers[col]))
        else:
            model+=(" + C({0},Treatment('{1}'))".format(col,refers[col]))

print("Model to try:")
print()
print(model)

mod = smf.ols(formula=model,data=df1)
models.append(mod.fit())
print(models[-1].summary().tables[0])
print(anovaTable(models[-1]))

necesary = ["Time_of_Day",
"Rd_Feature",
"Rd_Character",
"Rd_Class",
"Rd_Surface",
"Light",
"Traffic_Control"]
model = "Crash_Score_boxcox_0_27 ~ "
model1= ""
for col in necesary:
    if col == "year" or col =="Month" or col=="Time_of_Day":
        model+=(" + C({0},Treatment({1}))".format(col,refers[col]))  
    else:
        model+=(" + C({0},Treatment('{1}'))".format(col,refers[col]))
model1 = model1[1:]

print("Model to try:")
print()
print(model)

mod = smf.ols(formula=model,data=df1)
models.append(mod.fit())
models[-1].summary().tables[0]

necesary = [#"Time_of_Day",
#"Rd_Feature",
#"Rd_Character",
"Rd_Class",
#"Rd_Surface",
#"Light",
"Traffic_Control"]
inter1 = "Time_of_Day*Light"
inter2 = "Rd_Feature*Rd_Character*Rd_Surface"
model = "Crash_Score_boxcox_0_27 ~ "
model1= ""
for col in necesary:
    if col == "year" or col =="Month" or col=="Time_of_Day":
        model+=(" + C({0},Treatment({1}))".format(col,refers[col]))  
    else:
        model+=(" + C({0},Treatment('{1}'))".format(col,refers[col]))
        
model1 = inter1 + " + " + inter2
model += " + " + model1

print("Model to try:")
print()
print(model)

mod = smf.ols(formula=model,data=df1)
models.append(mod.fit())

models[-1].summary().tables[0]

sm.stats.anova_lm(models[-1])

necesary = [#"Time_of_Day",
#"Rd_Feature",
#"Rd_Character",
"Rd_Class",
#"Rd_Surface",
#"Light",
"Traffic_Control"]
inter1 = "Rd_Feature*Rd_Character*Rd_Surface*Time_of_Day"
inter2 = "Rd_Feature*Rd_Character*Rd_Surface*Light"
model = "Crash_Score_boxcox_0_27 ~ "
model1= ""
for col in necesary:
    if col == "year" or col =="Month" or col=="Time_of_Day":
        model+=(" + C({0},Treatment({1}))".format(col,refers[col]))  
    else:
        model+=(" + C({0},Treatment('{1}'))".format(col,refers[col]))
        
model1 = inter1 + " + " + inter2
model += " + " + model1
print("Model to try:")
print()
print(model)

mod = smf.ols(formula=model,data=df1)
models.append(mod.fit())

models[-1].summary().tables[0]

tab = sm.stats.anova_lm(models[-1])
print(tab)

sm.graphics.qqplot(models[-1].resid)
plt.title("Q-Q plot of predicted values")
plt.grid()
plt.tight_layout()
plt.savefig("tarea/3-qqplot4.pdf")
plt.show()

model = ""
for var in tab.index.tolist():
    if tab.loc[var,"PR(>F)"]<= 0.5:
        model += "+ " + var
model = "Crash_Score_boxcox_0_27 ~ " + model

print("Model to try:")
print()
print(model)

mod = smf.ols(formula=model,data=df1)
models.append(mod.fit())

models[-1].summary().tables[0]

model = ""
for var in tab.index.tolist():
    if tab.loc[var,"PR(>F)"]<= 0.1:
        model += "+ " + var
model = "Crash_Score_boxcox_0_27 ~ " + model

print("Model to try:")
print()
print(model)

mod = smf.ols(formula=model,data=df1)
models.append(mod.fit())

models[-1].summary().tables[0]

for model in models:
    i = models.index(model)
    coefficientTable(model).to_latex(buf="tarea/3-mod{0}Cof.tex".format(i),column_format="p{4cm}cccccc",longtable=True,float_format = "%.4f")
    anovaTable(model).to_latex(buf="tarea/3-mod{0}Anova.tex".format(i),column_format="p{6cm}lccccc",longtable=True,float_format = "%.4f")

tab = comparisonTable(models).drop("Model",axis="columns")
tab.index.name="model"
tab.to_latex(buf="tarea/3-comparison.tex",column_format="lrrrrrr")



