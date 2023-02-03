## for data
import numpy as np
import pandas as pd
features = []
## for plotting
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns


## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

## for machine learning
from sklearn import preprocessing, impute, utils, model_selection, metrics, ensemble
import imblearn

## for deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow import math
from keras import models, layers, backend as K
import minisom

# Scikit-learn : machine learning library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## for explainer
import shap

###############################################################################
#                       DATA ANALYSIS                                         #
###############################################################################
def utils_recognize_type(dtf, col, max_cat=20):
    '''
    Recognize whether a column is numerical or categorical.
    :parameter
        :param dtf: dataframe - input data
        :param col: str - name of the column to analyze
        :param max_cat: num - max number of unique values to recognize a column as categorical
    :return
        "cat" if the column is categorical, "dt" if datetime, "num" otherwise
    '''
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    elif dtf[col].dtype in ['datetime64[ns]','<M8[ns]']:
        return "dt"
    else:
        return "num"

'''
Get a general overview of a dataframe.
:parameter
    :param dtf: dataframe - input data
    :param max_cat: num - mininum number of recognize column type
'''
def dtf_overview(dtf, max_cat=20, figsize=(10,5)):
    ## recognize column type
    dic_cols = {col:utils_recognize_type(dtf, col, max_cat=max_cat) for col in dtf.columns}
        
    ## print info
    len_dtf = len(dtf)
    print("Shape:", dtf.shape)
    print("-----------------")
    for col in dtf.columns:
        info = col+" --> Type:"+dic_cols[col]
        info = info+" | Nas: "+str(dtf[col].isna().sum())+"("+str(int(dtf[col].isna().mean()*100))+"%)"
        if dic_cols[col] == "cat":
            info = info+" | Categories: "+str(dtf[col].nunique())
        elif dic_cols[col] == "dt":
            info = info+" | Range: "+"({x})-({y})".format(x=str(dtf[col].min()), y=str(dtf[col].max()))
        else:
            info = info+" | Min-Max: "+"({x})-({y})".format(x=str(int(dtf[col].min())), y=str(int(dtf[col].max())))
        if dtf[col].nunique() == len_dtf:
            info = info+" | Possible PK"
        print(info)
                
    ## plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = dtf.isnull()
    for k,v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Dataset Overview')
    #plt.setp(plt.xticks()[1], rotation=0)
    plt.show()
    
    ## add legend
    print("\033[1;37;40m Categerocial \033[m", "\033[1;30;41m Numerical/DateTime \033[m", "\033[1;30;47m NaN \033[m")


'''
Moves columns into a dtf.
:parameter
    :param dtf: dataframe - input data
    :param lst_cols: list - names of the columns that must be moved
    :param where: str - "front" or "end"
:return
    dtf with moved columns
'''
def pop_columns(dtf, lst_cols, where="front"):
    current_cols = dtf.columns.tolist()
    for col in lst_cols:    
        current_cols.pop( current_cols.index(col) )
    if where == "front":
        dtf = dtf[lst_cols + current_cols]
    elif where == "end":
        dtf = dtf[current_cols + lst_cols]
    return dtf



'''
Plots the frequency distribution of a dtf column.
:parameter
    :param dtf: dataframe - input data
    :param x: str - column name
    :param max_cat: num - max number of uniques to consider a numerical variable as categorical
    :param top: num - plot setting
    :param show_perc: logic - plot setting
    :param bins: num - plot setting
    :param quantile_breaks: tuple - plot distribution between these quantiles (to exclude outilers)
    :param box_logscale: logic
    :param figsize: tuple - plot settings
'''
def freqdist_plot(dtf, x, max_cat=20, top=None, show_perc=True, bins=100, quantile_breaks=(0,10), box_logscale=False, figsize=(10,5)):
    try:
        ## cat --> freq
        if utils_recognize_type(dtf, x, max_cat) == "cat":   
            ax = dtf[x].value_counts().head(top).sort_values().plot(kind="barh", figsize=figsize)
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=10, color='black')
            else:
                total = sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=10, color='black')
            ax.grid(axis="x")
            plt.suptitle(x, fontsize=20)
            plt.show()
            
        ## num --> density
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x, fontsize=20)
            ### distribution
            ax[0].title.set_text('distribution')
            variable = dtf[x].fillna(dtf[x].mean())
            breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
            variable = variable[ (variable > breaks[quantile_breaks[0]]) & (variable < breaks[quantile_breaks[1]]) ]
            sns.distplot(variable, hist=True, kde=True, kde_kws={"shade":True}, ax=ax[0])
            des = dtf[x].describe()
            ax[0].axvline(des["25%"], ls='--')
            ax[0].axvline(des["mean"], ls='--')
            ax[0].axvline(des["75%"], ls='--')
            ax[0].grid(True)
            des = round(des, 2).apply(lambda x: str(x))
            box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
            ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=1))
            ### boxplot 
            if box_logscale == True:
                ax[1].title.set_text('outliers (log scale)')
                tmp_dtf = pd.DataFrame(dtf[x])
                tmp_dtf[x] = np.log(tmp_dtf[x])
                tmp_dtf.boxplot(column=x, ax=ax[1])
            else:
                ax[1].title.set_text('outliers')
                dtf.boxplot(column=x, ax=ax[1])
            plt.show()   
        
    except Exception as e:
        print("--- got error ---")
        print(e)

'''
Plots a bivariate analysis.
:parameter
    :param dtf: dataframe - input data
    :param x: str - column
    :param y: str - column
    :param max_cat: num - max number of uniques to consider a numerical variable as categorical
'''
def bivariate_plot(dtf, x, y, max_cat=20, figsize=(10,5)):
    try:
        ## num vs num --> stacked + scatter with density
        if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
            ### stacked
            dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
            breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
            groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, duplicates='drop')])[y].agg(['mean','median','size'])
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            groups[["mean", "median"]].plot(kind="line", ax=ax)
            groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True, color="grey", alpha=0.3, grid=True)
            ax.set(ylabel=y)
            ax.right_ax.set_ylabel("Observazions in each bin")
            plt.show()
        
        ## cat vs cat --> hist count + hist %
        elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):  
            fig, ax = plt.subplots(1, 2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### count
            ax[0].title.set_text('count')
            order = dtf.groupby([x,y])[y].count()
            order = order.reset_index(name = 'count')
            sns.barplot(data=order, x=x, y='count', hue=y, ax=ax[0])
            ax[0].grid(True)
            ### percentage
            ax[1].title.set_text('percentage')
            a = dtf.groupby(x)[y].count().reset_index()
            a = a.rename(columns={y:"tot"})
            b = dtf.groupby([x,y])[y].count()
            b = b.reset_index(name = 'count')
            b = b.rename(columns={'count':0}).reset_index()
            b = b.merge(a, how="left")
            b["%"] = b[0] / b["tot"] *100
            sns.barplot(x=x, y="%", hue=y, data=b, ax=ax[1]).get_legend().remove()
            ax[1].grid(True)
            ### fix figure
            plt.show()
        
        ## num vs cat --> density + stacked + boxplot 
        else:
            if (utils_recognize_type(dtf, x, max_cat) == "cat"):
                cat,num = x,y
            else:
                cat,num = y,x
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### distribution
            ax[0].title.set_text('density')
            for i in sorted(dtf[cat].unique()):
                sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
            ax[0].grid(True)
            ### stacked
            dtf_noNan = dtf[dtf[num].notnull()]  #can't have nan
            ax[1].title.set_text('bins')
            breaks = np.quantile(dtf_noNan[num], q=np.linspace(0,1,11))
            tmp = dtf_noNan.groupby([cat, pd.cut(dtf_noNan[num], breaks, duplicates='drop')]).size().unstack().T
            tmp = tmp[dtf_noNan[cat].unique()]
            tmp["tot"] = tmp.sum(axis=1)
            for col in tmp.drop("tot", axis=1).columns:
                tmp[col] = tmp[col] / tmp["tot"]
            tmp.drop("tot", axis=1)[sorted(dtf[cat].unique())].plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
            ### fix figure
            plt.show()
        
    except Exception as e:
        print("--- got error ---")
        print(e)

###############################################################################
#                         CORRELATION                                         #
###############################################################################        

    
'''
Computes correlation/dependancy and p-value (prob of happening something different than what observed in the sample)
'''
def test_corr(dtf, x, y, max_cat=20):
    ## num vs num --> pearson
    if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
        dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
        coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## cat vs cat --> cramer (chiquadro)
    elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):
        cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
        chi2_test = scipy.stats.chi2_contingency(cont_table)
        chi2, p = chi2_test[0], chi2_test[1]
        n = cont_table.sum().sum()
        phi2 = chi2/n
        r,k = cont_table.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Cramer Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## num vs cat --> 1way anova (f: the means of the groups are different)
    else:
        if (utils_recognize_type(dtf, x, max_cat) == "cat"):
            cat,num = x,y
        else:
            cat,num = y,x
        model = smf.ols(num+' ~ '+cat, data=dtf).fit()
        table = sm.stats.anova_lm(model)
        p = table["PR(>F)"][0]
        coeff, p = None, round(p, 3)
        conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
        print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")
        
    return coeff, p

###############################################################################
#                       PREPROCESSING                                         #
###############################################################################
'''
Split the dataframe into train / test
'''
def dtf_partitioning(dtf, y, test_size=0.3, shuffle=False):
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, shuffle=shuffle) 
    print("X_train shape:", dtf_train.drop(y, axis=1).shape, "| X_test shape:", dtf_test.drop(y, axis=1).shape)
    print("y_train mean:", round(np.mean(dtf_train[y]),2), "| y_test mean:", round(np.mean(dtf_test[y]),2))
    print(dtf_train.shape[1]-1, "features:", dtf_train.drop(y, axis=1).columns.to_list())
    return dtf_train, dtf_test

'''
Rebalances a dataset with up-sampling and down-sampling.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - column to use as target 
    :param balance: str - "up", "down", if None just prints some stats
    :param method: str - "random" for sklearn or "knn" for imblearn
    :param size: num - 1 for same size of the other class, 0.5 for half of the other class
:return
    rebalanced dtf
'''
def rebalance(dtf, y, balance=None,  method="random", replace=True, size=1):
    ## check
    print("--- situation ---")
    check = dtf[y].value_counts().to_frame()
    check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
    print(check)
    print("tot:", check[y].sum())

    ## sklearn
    if balance is not None and method == "random":
        ### set the major and minor class
        major = check.index[0]
        minor = check.index[1]
        dtf_major = dtf[dtf[y]==major]
        dtf_minor = dtf[dtf[y]==minor]

        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   randomly replicate observations from the minority class (Overfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   randomly remove observations of the majority class (Underfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

    ## imblearn
    if balance is not None and method == "knn":
        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   create synthetic observations from the minority class (Distortion risk)")
            smote = imblearn.over_sampling.SMOTE(random_state=123)
            dtf_balanced, y_values = smote.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
       
        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   select observations that don't affect performance (Underfitting risk)")
            nn = imblearn.under_sampling.CondensedNearestNeighbour(random_state=123)
            dtf_balanced, y_values = nn.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
        
    ## check rebalance
    if balance is not None:
        print("--- new situation ---")
        check = dtf_balanced[y].value_counts().to_frame()
        check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
        print(check)
        print("tot:", check[y].sum())
        return dtf_balanced
    

'''
Scales features.
'''
def scaling(dtf, y, scalerX=None, scalerY=None, fitted=False, task="classification"):
    scalerX = preprocessing.MinMaxScaler(feature_range=(0,1)) if scalerX is None else scalerX
    if fitted is False:
        scalerX.fit(dtf.drop(y, axis=1))
    X = scalerX.transform(dtf.drop(y, axis=1))
    dtf_scaled = pd.DataFrame(X, columns=dtf.drop(y, axis=1).columns, index=dtf.index)
    if task == "regression":
        scalerY = preprocessing.MinMaxScaler(feature_range=(0,1)) if scalerY is None else scalerY
        dtf_scaled[y] = scalerY.fit_transform(dtf[y].values.reshape(-1,1)) if fitted is False else dtf[y]
        return dtf_scaled, scalerX, scalerY
    else:
        dtf_scaled[y] = dtf[y]
        return dtf_scaled, scalerX

'''
Computes all the required data preprocessing.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - name of the dependent variable 
    :param processNas: str or None - "mean", "median", "most_frequent"
    :param processCategorical: str or None - "dummies"
    :param split: num or None - test_size (example 0.2)
    :param scale: str or None - "standard", "minmax"
    :param task: str - "classification" or "regression"
:return
    dictionary with dtf, X_names lsit, (X_train, X_test), (Y_train, Y_test), scaler
'''
def data_preprocessing(dtf, y, processNas=None, processCategorical=None, split=None, scale=None, task="classification"):
    try:
        dtf = pop_columns(dtf, [y], "front")
        
        ## missing
        ### check
        print("--- check missing ---")
        if dtf.isna().sum().sum() != 0:
            cols_with_missings = []
            for col in dtf.columns.to_list():
                if dtf[col].isna().sum() != 0:
                    print("WARNING:", col, "-->", dtf[col].isna().sum(), "Nas")
                    cols_with_missings.append(col)
            ### treat
            if processNas is not None:
                print("...treating Nas...")
                cols_with_missings_numeric = []
                for col in cols_with_missings:
                    if dtf[col].dtype == "O":
                        print(col, "categorical --> replacing Nas with label 'missing'")
                        dtf[col] = dtf[col].fillna('missing')
                    else:
                        cols_with_missings_numeric.append(col)
                if len(cols_with_missings_numeric) != 0:
                    print("replacing Nas in the numerical variables:", cols_with_missings_numeric)
                imputer = impute.SimpleImputer(strategy=processNas)
                imputer = imputer.fit(dtf[cols_with_missings_numeric])
                dtf[cols_with_missings_numeric] = imputer.transform(dtf[cols_with_missings_numeric])
        else:
            print("   OK: No missing")
                
        ## categorical data
        ### check
        print("--- check categorical data ---")
        cols_with_categorical = []
        for col in dtf.columns.to_list():
            if dtf[col].dtype == "O":
                print("WARNING:", col, "-->", dtf[col].nunique(), "categories")
                cols_with_categorical.append(col)
        ### treat
        if len(cols_with_categorical) != 0:
            if processCategorical is not None:
                print("...trating categorical...")
                for col in cols_with_categorical:
                    print(col)
                    dtf = pd.concat([dtf, pd.get_dummies(dtf[col], prefix=col)], axis=1).drop([col], axis=1)
        else:
            print("   OK: No categorical")
        
        ## 3.split train/test
        print("--- split train/test ---")
        X = dtf.drop(y, axis=1).values
        Y = dtf[y].values
        if split is not None:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=split, shuffle=False)
            print("X_train shape:", X_train.shape, " | X_test shape:", X_test.shape)
            print("y_train mean:", round(np.mean(y_train),2), " | y_test mean:", round(np.mean(y_test),2))
            print(X_train.shape[1], "features:", dtf.drop(y, axis=1).columns.to_list())
        else:
            print("   OK: step skipped")
            X_train, y_train, X_test, y_test = X, Y, None, None
        
        ## 4.scaling
        print("--- scaling ---")
        if scale is not None:
            scalerX = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
            X_train = scalerX.fit_transform(X_train)
            scalerY = 0
            if X_test is not None:
                X_test = scalerX.transform(X_test)
            if task == "regression":
                scalerY = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
                y_train = scalerY.fit_transform(y_train.reshape(-1,1))
            print("   OK: scaled all features")
        else:
            print("   OK: step skipped")
            scalerX, scalerY = 0, 0
        
        return {"dtf":dtf, "X_names":dtf.drop(y, axis=1).columns.to_list(), 
                "X":(X_train, X_test), "y":(y_train, y_test), "scaler":(scalerX, scalerY)}
    
    except Exception as e:
        print("--- got error ---")
        print(e)


###############################################################################
#                     VISUALIZE MODELS AND DATA                               #
###############################################################################
'''
Plots a 2d classification model result.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param y_test: array
    :param model: model istance (before fitting)
'''
def plot2d_classif_model(X_train, y_train, X_test, y_test, model=None, annotate=False, figsize=(10,5)):
    ## n features > 2d
    if X_train.shape[1] > 2:
        print("--- reducing dimensions to 2 ---")
        X_train, X_test, pca = utils_dimensionality_reduction(X_train, X_test, n_features=2)
    
    ## fit 2d model
    print("--- fitting 2d model ---")
    model_2d = ensemble.GradientBoostingClassifier() if model is None else model
    model_2d.fit(X_train, y_train)
    
    ## plot predictions
    print("--- plotting test set ---")
    from matplotlib.colors import ListedColormap
    colors = {np.unique(y_test)[0]:"black", np.unique(y_test)[1]:"green"}
    X1, X2 = np.meshgrid(np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=0.01),
                         np.arange(start=X_test[:,1].min()-1, stop=X_test[:,1].max()+1, step=0.01))
    fig, ax = plt.subplots(figsize=figsize)
    Y = model_2d.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    ax.contourf(X1, X2, Y, alpha=0.5, cmap=ListedColormap(list(colors.values())))
    ax.set(xlim=[X1.min(),X1.max()], ylim=[X2.min(),X2.max()], title="Classification regions")
    for i in np.unique(y_test):
        ax.scatter(X_test[y_test==i, 0], X_test[y_test==i, 1], c=colors[i], label="true "+str(i))  
    if annotate is True:
        for n,i in enumerate(y_test):
            ax.annotate(n, xy=(X_test[n,0], X_test[n,1]), textcoords='offset points', ha='left', va='bottom')
    plt.legend()
    plt.show()
    

'''
Extract info for each layer in a keras model.
'''
def utils_nn_config(model):
    lst_layers = []
    if "Sequential" in str(model): #-> Sequential doesn't show the input layer
        layer = model.layers[0]
        lst_layers.append({"name":"input", "in":int(layer.input.shape[-1]), "neurons":0, 
                           "out":int(layer.input.shape[-1]), "activation":None,
                           "params":0, "bias":0})
    for layer in model.layers:
        try:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":layer.units, 
                         "out":int(layer.output.shape[-1]), "activation":layer.get_config()["activation"],
                         "params":layer.get_weights()[0], "bias":layer.get_weights()[1]}
        except:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":0, 
                         "out":int(layer.output.shape[-1]), "activation":None,
                         "params":0, "bias":0}
        lst_layers.append(dic_layer)
    return lst_layers

'''
Plot the structure of a keras neural network.
'''
def visualize_nn(model, description=False, figsize=(10,8)):
    ## get layers info
    lst_layers = utils_nn_config(model)
    layer_sizes = [layer["out"] for layer in lst_layers]
    
    ## fig setup
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.set(title=model.name)
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right-left) / float(len(layer_sizes)-1)
    y_space = (top-bottom) / float(max(layer_sizes))
    p = 0.025
    
    ## nodes
    for i,n in enumerate(layer_sizes):
        top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
        layer = lst_layers[i]
        color = "green" if i in [0, len(layer_sizes)-1] else "blue"
        color = "red" if (layer['neurons'] == 0) and (i > 0) else color
        
        ### add description
        if (description is True):
            d = i if i == 0 else i-0.5
            if layer['activation'] is None:
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
            else:
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
                plt.text(x=left+d*x_space, y=top-p, fontsize=10, color=color, s=layer['activation']+" (")
                plt.text(x=left+d*x_space, y=top-2*p, fontsize=10, color=color, s="Σ"+str(layer['in'])+"[X*w]+b")
                out = " Y"  if i == len(layer_sizes)-1 else " out"
                plt.text(x=left+d*x_space, y=top-3*p, fontsize=10, color=color, s=") = "+str(layer['neurons'])+out)
        
        ### circles
        for m in range(n):
            color = "limegreen" if color == "green" else color
            circle = plt.Circle(xy=(left+i*x_space, top_on_layer-m*y_space-4*p), radius=y_space/4.0, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            ### add text
            if i == 0:
                plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
            elif i == len(layer_sizes)-1:
                plt.text(x=right+4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$y_{'+str(m+1)+'}$')
            else:
                plt.text(x=left+i*x_space+p, y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, fontsize=10, s=r'$H_{'+str(m+1)+'}$')
    
    ## links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i+1]
        color = "green" if i == len(layer_sizes)-2 else "blue"
        color = "red" if layer['neurons'] == 0 else color
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i*x_space+left, (i+1)*x_space+left], 
                                  [layer_top_a-m*y_space, layer_top_b-o*y_space], 
                                  c=color, alpha=0.5)
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    plt.show()

###############################################################################
#                   MODEL DESIGN & TESTING - CLASSIFICATION                   #
###############################################################################
'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()

'''
Fits a keras artificial/deep neural network.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param batch_size: num - keras batch
    :param epochs: num - keras epochs
    :param threshold: num - predictions > threshold are 1, otherwise 0
:return
    model fitted and predictions
'''
def fit_dl_classif(X_train, y_train, X_test, model=None, batch_size=32, epochs=100, verbose=0, threshold=0.5):
    ## model
    if model is None:
        ### define F1 metrics for Keras
        def Recall(y, y_hat):
            true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
    
        def Precision(y, y_hat):
            true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_hat, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
    
        def F1(y, y_hat):
            precision = Precision(y, y_hat)
            recall = Recall(y, y_hat)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
        ### build ann
        n_features = X_train.shape[1]
        #### layer input
        inputs = layers.Input(name="input", shape=(n_features,))
        #### hidden layer 1
        h1 = layers.Dense(name="h1", units=int(round((n_features+1)/2)), activation='relu')(inputs)
        h1 = layers.Dropout(name="drop1", rate=0.2)(h1)
        #### hidden layer 2
        h2 = layers.Dense(name="h2", units=int(round((n_features+1)/4)), activation='relu')(h1)
        h2 = layers.Dropout(name="drop2", rate=0.2)(h2)
        #### layer output
        outputs = layers.Dense(name="output", units=1, activation='sigmoid')(h2)
        model = models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',F1])
        print(model.summary())
    
    ## train
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3)
    if epochs > 1:
        utils_plot_keras_training(training)
    
    ## test
    predicted_prob = training.model.predict(X_test)
    predicted = (predicted_prob > threshold)
    return training.model, predicted_prob, predicted

'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
    :param predicted_prob: array
    :param show_thresholds: bool - if True annotates thresholds on the curves
'''
def evaluate_classif_model(y_test, predicted, predicted_prob, show_thresholds=True, figsize=(25,5)):
    classes = np.unique(y_test)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    
    ## Accuray e AUC
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob)
    print("Accuracy (overall correct predictions):",  round(accuracy,2))
    print("Auc:", round(auc,2))
    
    ## Precision e Recall
    recall = metrics.recall_score(y_test, predicted)  #= true 1s / all 1s in test (got and missed)
    precision = metrics.precision_score(y_test, predicted)  #= true 1s / all 1s predicted (true 1s + false 1s)
    f1 = metrics.f1_score(y_test, predicted)  #=2 * (precision * recall) / (precision + recall)
    print("Recall (all 1s predicted right):", round(recall,2))  #true positive rate, how many 1s I got
    print("Precision (confidence when predicting a 1):", round(precision,2))  #how many 1s were really 1s
    print("F1 score:", round(f1,2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in classes]))
    
    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
    sns.heatmap(cm, annot=True, annot_kws={"size":15}, fmt='d', ax=ax[0], cmap=plt.cm.Blues, cbar=False)
    ax[0].set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax[0].set_yticklabels(labels=classes, rotation=0)
    
    ## Plot roc
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_prob)    
    ax[1].plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % metrics.auc(fpr, tpr))
    ax[1].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[1].hlines(y=recall, xmin=-0.05, xmax=1-cm[0,0]/(cm[0,0]+cm[0,1]), color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[1].vlines(x=1-cm[0,0]/(cm[0,0]+cm[0,1]), ymin=0, ymax=recall, color='red', linestyle='--', alpha=0.7)
    ax[1].set(xlim=[-0.05,1], ylim=[0.0,1.05], xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")     
    ax[1].legend(loc="lower right")
    ax[1].grid(True)
    if show_thresholds is True:
        thres_in_plot = []
        for i,t in enumerate(thresholds):
            t = np.round(t,1)
            if t not in thres_in_plot:
                ax[1].annotate(t, xy=(fpr[i],tpr[i]), xytext=(fpr[i],tpr[i]), textcoords='offset points', ha='left', va='bottom')
                thres_in_plot.append(t)
    
    ## Plot precision-recall curve
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, predicted_prob)
    ax[2].plot(recalls, precisions, color='darkorange', lw=3, label='area = %0.2f' % metrics.auc(recalls, precisions))
    ax[2].plot([0,1], [(cm[1,0]+cm[1,0])/len(y_test), (cm[1,0]+cm[1,0])/len(y_test)], linestyle='--', color='navy', lw=3)
    ax[2].hlines(y=precision, xmin=0, xmax=recall, color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[2].vlines(x=recall, ymin=0, ymax=precision, color='red', linestyle='--', alpha=0.7)
    ax[2].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[2].legend(loc="lower left")
    ax[2].grid(True)
    if show_thresholds is True:
        thres_in_plot = []
        for i,t in enumerate(thresholds):
            t = np.round(t,1)
            if t not in thres_in_plot:
                ax[2].annotate(np.round(t,1), xy=(recalls[i],precisions[i]), xytext=(recalls[i],precisions[i]), textcoords='offset points', ha='right', va='bottom')
                thres_in_plot.append(t)
    
    plt.show()

'''
Use shap to build an a explainer.
:parameter
    :param model: model instance (after fitting)
    :param X_names: list
    :param X_instance: array of size n x 1 (n,)
    :param X_train: array - if None the model is simple machine learning, if not None then it's a deep learning model
    :param task: string - "classification", "regression"
    :param top: num - top features to display
:return
    dtf with explanations
'''
def explainer_shap(model, X_names, X_instance, X_train=None, task="classification", top=10):
    ## create explainer
    ### machine learning
    if X_train is None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
    ### deep learning
    else:
        explainer = shap.DeepExplainer(model, data=X_train[:100])
        shap_values = explainer.shap_values(X_instance.reshape(1,-1))[0].reshape(-1)

    ## plot
    ### classification
    if task == "classification":
        shap.decision_plot(explainer.expected_value, shap_values, link='logit', feature_order='importance',
                           features=X_instance, feature_names=X_names, feature_display_range=slice(-1,-top-1,-1))
    ### regression
    else:
        shap.waterfall_plot(explainer.expected_value[0], shap_values, 
                            features=X_instance, feature_names=X_names, max_display=top)
        
        
        
def colorGenerator(numberOfColors:int):
  """_summary_

  Args:
      numberOfColors (int): minimum 8 colors
  """
  if (numberOfColors<8):
    numberOfColors = 8
  
  posibleCombinations:int = 0
  posibleCombinations+=255#red up
  posibleCombinations+=255#green up
  posibleCombinations+=255#red down
  posibleCombinations+=255#blue up
  posibleCombinations+=255#green down
  posibleCombinations+=255#red up
  posibleCombinations+=255#blue down
  
  colors = ["#ff0000"]
  rgbd:list[int] = [255,0,0,2]
  for i in range(0,numberOfColors-1):
    rgbd = split(rgbd[0],rgbd[1],rgbd[2],rgbd[3],posibleCombinations//(numberOfColors-1))
    try:
      r:str=hex(rgbd[0]).replace("0x","")
      g:str=hex(rgbd[1]).replace("0x","")
      b:str=hex(rgbd[2]).replace("0x","")
      
      if(len(r) < 2):
        r="0"+r
      if(len(g) < 2):
        g="0"+g
      if(len(b) < 2):
        b="0"+b
      
      colors.append(("#" + r + g + b))
    except:
      print(rgbd[0],rgbd[1],rgbd[2])
  return colors


def split(r:int,g:int,b:int,direction:int,difference:int):
  if(direction == 0):
    r+=difference
    if(r>255):
      g+=r%255
      r=255
      direction+=1
    return [r,g,b,direction]
  if(direction == 1):
    b-=difference
    if(b<0):
      g+=abs(b)
      b=0
      direction+=1
    return [r,g,b,direction]
  if(direction == 2):
    g+=difference
    if(g>255):
      r-=g%255
      g=255
      direction+=1
    return [r,g,b,direction]
  if(direction == 3):
    r-=difference
    if(r<0):
      b+=abs(r)
      r=0
      direction+=1
    return [r,g,b,direction]
  if(direction == 4):
    b+=difference
    if(b>255):
      g-=b%255
      b=255
      direction+=1
    return [r,g,b,direction]
  if(direction == 5):
    g-=difference
    if(g<0):
      r+=abs(g)
      g=0
      direction=0
    return [r,g,b,direction]
  return [r,g,b,direction]

print(colorGenerator(10))