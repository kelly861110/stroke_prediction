import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import io
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score

# the code to get the dataset with categorical bmi
def categorical_bmi_and_smoke():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    # drop the gender == other
    df = df.drop(df[df['gender']=="Other"].index)
    # get all the data points with bmi == nan
    newdf = df[df["bmi"].isna()]
    df.dropna(inplace = True)
    # get the data points with no nan (use this to train our model)
    bmi = df["bmi"]
    check_bmi = []
    # get the label for the traing data set
    for i in bmi:
        if i < 18.5:
            check_bmi.append("underweight")
        elif i >= 18.5 and i < 25:
            check_bmi.append("healthy")
        elif i >= 25.0 and i < 30:
            check_bmi.append("overweight")
        else:
            check_bmi.append("obese")
    df = df.drop(columns = ['bmi'])
    df['bmi'] = check_bmi
    # one hot encoded the predicting data points
    X_newdf = pd.get_dummies(newdf, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
    X_newdf['X0'] = 1

    coef = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke', 'gender_Male', 'smoking_status_formerly smoked',
            'smoking_status_never smoked', 'smoking_status_smokes',
            'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'ever_married_Yes', 'Residence_type_Urban', 'X0']

    new_X = X_newdf[coef]
    # one hot encoded the training set
    X_df = pd.get_dummies(df, columns= ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
    X_df['X0'] = 1
    X = X_df[coef]
    Y = check_bmi
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # git the training set
    Logistic = LogisticRegression(solver = "newton-cg", multi_class='multinomial').fit(X_train, Y_train)
    print("the accuracy of predicting the bmi: " + str(Logistic.score(X_test,Y_test)))

    Logistic = LogisticRegression(solver = "newton-cg", multi_class='multinomial').fit(X, Y)
    newbmi = Logistic.predict(new_X)
    newdf = newdf.drop(columns = ['bmi'])
    newdf['bmi'] = newbmi

    # this is the new dataframe after predicting the bmi and change the bmi to categrical
    df = df.append(newdf)
    # get the data points with smoking_status == unkown
    newdf = df[df["smoking_status"] == 'Unknown']
    print(len(newdf))
    # get all the other data points
    df = df[df["smoking_status"] != 'Unknown']
    print(len(df))
    # one hot encoded the predicting data points
    X_newdf = pd.get_dummies(newdf, columns=['gender', 'ever_married', 'work_type', 'Residence_type'], drop_first=True)
    X_newdf['X0'] = 1
    # one hot encoded the training set
    X_df = pd.get_dummies(df, columns= ['gender', 'ever_married', 'work_type', 'Residence_type'])
    X_df['X0'] = 1

    coef = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
              'gender_Male', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'ever_married_Yes', 'Residence_type_Urban', 'X0']
    # get the X and Y for the model
    new_X = X_newdf[coef]
    #new_Y = X_newdf['smoking_status']
    X = X_df[coef]
    Y = X_df['smoking_status']
    # get training and testing set from the
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # get the predicting score
    Logistic = LogisticRegression(solver = "newton-cg", multi_class='multinomial').fit(X_train, Y_train)
    print(Logistic.score(X_test,Y_test))

    # predict the unknown smoking status
    Logistic = LogisticRegression(solver = "newton-cg", multi_class='multinomial').fit(X, Y)
    newstatus = Logistic.predict(new_X)
    newdf['smoking_status'] = newstatus

    # this is the new dataframe after predicitng the Unknown smokeing status
    df = df.append(newdf)

    return df



def cont_bmi_dataframe(meanbmi=False):
    '''
    returns a one-hot encoded dataframe where smoking status is calculated by a knn classifier and
    bmi is calculated by knn regression if meanbmi is false, otherwise nans in the bmi column are
    replaced by the mean column bmi.

    returns Pandas DataFrame, Target Variable
    '''

    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    #mix up the data since it came in ordered
    df = df.sample(frac=1)
    #drop the id column since it holds no predictive value as its not discrete
    df = df.drop('id', axis=1)

    #We now check to verify there are sufficient values in each category for the columns to run a prediction
    categories = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for c in categories:
        continue
        print(df[c].value_counts())

    #Since there is only one person with gender Other we drop the row because there are not enough for a prediction
    df = df.drop(df[df['gender']=='Other'].index).reset_index(drop=True)

    '''
    We leave the rows where people never worked since 22 is enough to gather some predictive information and
    never working is very distinct from having a job. However, it may be linked to age, as there are some babies
    in the dataset.

    The smoking status column has the categories former smoker, never smoked, smokes, and unknown. The unknown values are
    essencially Nan values. We are going to use a knn classifier to impute these values.

    Before we do this, we need to one-hot encode the other categorical variables in order for the classifier to work, and
    we need to drop the stroke column, as we don't want it to be used in this imputer because that would bias our final
    predictions. We also need to encode the labels of the smoking_status column in order to be able to use the KNN classifier.
    '''

    le1 = LabelEncoder()
    newcol = le1.fit_transform(df['smoking_status'])
    df['smoking_status'] = pd.Series(newcol)
    keys = le1.classes_
    values = le1.transform(keys)
    dictionary = dict(zip(keys, values))
    inversedictionary = dict(zip(values, keys))

    #one-hot encode the dataframe
    df = pd.get_dummies(df, columns= ['gender', 'ever_married', 'work_type', 'Residence_type'])
    y = df['stroke']
    df = df.drop('stroke', axis=1)

    #Find the best number of neighbors to use for the prediction
    dfclass = df.drop('bmi', axis=1).dropna().copy()
    dfclass['smoking_status'] = pd.Series(newcol)
    dfclass = dfclass[dfclass['smoking_status'] != dictionary['Unknown']].copy()
    yclass = dfclass['smoking_status'].copy()
    dfclass = dfclass.drop('smoking_status', axis=1)

    #Encode the labels in the prediction column
    le = LabelEncoder().fit(yclass)
    knnclass = KNeighborsClassifier()
    parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,29]}
    bestparamsclass = GridSearchCV(knnclass, parameters).fit(dfclass, yclass).best_params_


    #Use the best value to fit a knnclassifier
    df['smoking_status'] = df['smoking_status'].replace(to_replace=dictionary['Unknown'], value=np.nan)
    yclassifier = df['smoking_status'].copy()
    dfclassifier = df.drop(['bmi','smoking_status'], axis=1).copy()
    knnclassifier = KNeighborsClassifier(n_neighbors=bestparamsclass['n_neighbors']).fit(dfclass, yclass)

    #predict nan values only
    for i in range(len(df)):
        if np.isnan(yclassifier[i]):
            yclassifier[i] = int(knnclassifier.predict(dfclassifier.loc[[i]]))

    #replace all values with original or new predicted values
    for i in range(len(yclassifier)):
        yclassifier[i] = inversedictionary[yclassifier[i]]

    #We next explore options to account for the Nan values in the BMI column
    if meanbmi:

        #We replace Nan values with the mean bmi
        imputermean = SimpleImputer(missing_values=np.nan, strategy='mean')
        dfmean = pd.DataFrame(imputermean.fit_transform(df), columns=df.columns)
        dfmean['smoking_status'] = yclassifier
        dfmean = pd.get_dummies(dfmean, columns=['smoking_status'])
        return dfmean, y

    else:
        #We use knn to fill in the missing values
        #We begin by finding the optimal number of neighbors to use for the prediction
        dfreg = df.dropna()
        yreg = dfreg['bmi']
        dfreg = dfreg.drop(['bmi','smoking_status'], axis=1)
        knnreg = KNeighborsRegressor()
        parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,23,26,29]}
        bestparams = GridSearchCV(knnreg, parameters).fit(dfreg, yreg).best_params_

        #Use the best value to impute the values to use for bmi
        imputerKNN = KNNImputer(n_neighbors=bestparams['n_neighbors'])
        dfKNN = pd.DataFrame(imputerKNN.fit_transform(df), columns = df.columns)
        dfKNN['smoking_status'] = yclassifier
        dfKNN = pd.get_dummies(dfKNN, columns=['smoking_status'])
        return dfKNN, y
