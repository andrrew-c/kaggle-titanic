
""" Kaggle competition:   https://www.kaggle.com/c/titanic
    Date: February 2021
"""
import pandas as pd
import numpy as np

import re


from datetime import datetime

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer as knni
from sklearn.impute import SimpleImputer 


# Transformers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder


# Modelling
from sklearn.linear_model import RidgeClassifier

# Model selection
from sklearn.model_selection import GridSearchCV

# Testing
from sklearn.dummy import DummyClassifier 


import matplotlib.pyplot as plt
import seaborn as sns

train_all = 'train.csv'

# Testing for submission
test_sub = 'test.csv'
seed = 20210220


# Today 
today = datetime.now().strftime('%Y-%m-%d')

#map titles
tmap = {'Ms.':'Miss.'
            , 'Mme.':'Mrs.'
        , 'Mlle.':'Miss'
        }
    

def deriveTitle(name):


    """ Take a string and return a title
        Assumes title is the first 'word',
    """

    tokens = re.split(', ', name)
    
    title = tokens[1].split()[0]
    if title in tmap:
        title = tmap[title]
    return title


def encodeNonMiss(BaseEstimator, TransformerMixin):
    """ Purpose:
            Encode non-missing data only and then impute later

        enctype - encoder type 'nom' for nominal 'ord' for ordinal

        Source:
                Idea for the approach: https://towardsdatascience.com/preprocessing-encode-and-knn-impute-all-categorical-features-fast-b05f50b4dfaa

                Ordinal/nominal encoding: https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/

    """

    def __init__(self):
        self.encoders = dict()
    
    def fit(self):
        df = data.copy()
    
    # Non-missing data only
    dnm = df[df.notnull()].copy()
    

    # Turn data into array
    dnm = np.array(dnm).reshape(-1,1)

    # Use ordinal encoder for labels
    labels = oe.fit_transform(dnm.copy())
    print(labels.shape)
    
    # Update 
    df.loc[df.notnull()] = np.squeeze(labels)
    

    return df

class NameTransformer(BaseEstimator, TransformerMixin):

    """ Derive title from name """

    def __init__(self):
        
        # map titles
        self.tmap = {'Ms.':'Miss.'
            , 'Mme.':'Mrs.'
        , 'Mlle.':'Miss'
        }

    def getTitle(self, name):

        """ Take a string and return a title
            Assumes title is the first 'word',
        """


        # Split name into tokens
        tokens = re.split(', ', name)
        
        # Get title = first token
        title = tokens[1].split()[0]

        # If token in dictionary (e.g. Ms -> Miss)
        if title in self.tmap:
            title = self.tmap[title]

        return title

    def fit(self, X, y=None):


        # Get titles
        X['titles'] = X.Name.apply(lambda x: self.getTitle(x))

        # Create dummy variables
        title_dummies = pd.get_dummies(X['titles'], prefix='title')

        # Make note of columns
        self.title_names = title_dummies.columns

        return self

    def transform(self, X, y=None):

      
        
        # Get titles
        X['titles'] = X.Name.apply(lambda x: self.getTitle(x))

        # Create dummy variables
        title_dummies = pd.get_dummies(X[titles], prefix='title')

        title_dummies = title_dummies.reindex(columns=self.columns)

        return X


    
class CustomEncoder(BaseEstimator, TransformerMixin):

    """ On init creates dictionary of encoders
        fit:
            loop through columns
            create label encoder
            get classes/vlaues

            update encoders

        transform:
            loop through columns
            get dictionary from object
            
        Source: https://stackoverflow.com/questions/64900801/implementing-knn-imputation-on-categorical-variables-in-an-sklearn-pipeline"""
    def __init__(self):
        self.encoders = dict()

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X.loc[X[col].notna(), col])
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))

            # Set unknown to new value so transform on test set handles unknown values
            max_value = max(le_dict.values())
            le_dict['_unk'] = max_value + 1

            self.encoders[col] = le_dict
        return self

    def transform(self, X, y=None):
        Xc = X.copy()
        for col in X.columns:
            le_dict = self.encoders[col]
            Xc.loc[Xc[col].notna(), col] = Xc.loc[Xc[col].notna(), col].apply(
                lambda x: le_dict.get(x, le_dict['_unk'])).values
        return Xc

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)    

def derivedVars(df):

    """ Derive title and section"""

    # copy dataframe
    dff = df.copy()

    
    #dff['Sex'] = df.Sex.map(sexMap)
    
    # Derive title of passenger
    dff['title'] = df.Name.apply(deriveTitle)

    # Derive section
    dff['section'] =df.Cabin.str[0]

    
    return dff


if __name__=='__main__':

    
    # Ordinal encoder
    oe = OrdinalEncoder()

    # Load in all training data
    df = pd.read_csv(train_all)
    df = df.set_index('PassengerId')

    # Load in testing data
    df_test = pd.read_csv(test_sub)

    # Check shape of data
    print("df.shape:", df.shape)
    print(df.info(verbose=True))

    # Categorical variables
    catVars = "Sex,Cabin,Embarked,title,section".split(',')

    # Numerical variables
    numVars = "Age,SibSp,Parch,Fare".split(',')

    # Ordinal variables - 
    ordVars = "Pclass".split(',')

    #sexMap = dict(male=0, female=1)

 

    X = derivedVars(df)
    print(X.columns, "_---------_")
    
    # Drop any columns which we won't be modelling
    X = X.drop(columns='Survived,Name,Ticket'.split(','))
    
    # Outcome variable
    y = df.Survived
    
    print(X.shape, y.shape)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Categorical variables - transformers
    cat_pipeline = Pipeline(steps=[
                            ('custenc', CustomEncoder())
                        
                            ])

    # Pipeline for processing numeric variables
    num_pipeline = Pipeline(steps=[('scaler', StandardScaler())
                            ])

    

    # Process categorical and numeric
    union = ColumnTransformer(transformers=[('cat', cat_pipeline, catVars)
                                                    , ('num', num_pipeline, numVars)
                                                   
                                                    ])

    # Pre-processing, 
    full_pipe = Pipeline(steps=[('union', union)
                                , ('impute', knni())
                                , ('classifier', RidgeClassifier())
                                ])

    # Search grid
    param_grid = dict(classifier__alpha=np.linspace(0, 10, 4)
                        , classifier__fit_intercept = [True,False]
                      , classifier__normalize=[True, False])
    # Grid search    
    search = GridSearchCV(full_pipe, param_grid)

    # Fit model
    search.fit(X_train, y_train)

##    modelname = f'model_{today}.pkl'
##    ck = input(f"Save model as {modelname}?\nY or N:")
##    if ck.upper() == 'Y':
##        with open(modelname, 'wb') as f: pickle.dump(search.best_estimator_)
    score_train = search.score(X_train, y_train)
    score_test = search.score(X_test, y_test)
    print("Training score: {:.2%}".format(score_train))
    print("Test score: {:.2%}".format(score_test))

    def makeSubmission(df, model):

        print("Derived variables")
        print("Shape before:", df.shape)
        df_ = derivedVars(df)
        print("Shape after:", df_.shape)

        # Change index
        # df_test = df_test.set_index('PassengerId')

        # Re-order columns
        df_ = df_[X_train.columns]

        # Predictions
        preds = pd.DataFrame(data=model.predict(df_), index=df.PassengerId, columns=['Survived']).reset_index()

        # Set up today as a variable and define filename
        today = datetime.today().strftime('%Y-%m-%d')
        fname = f'submission_{today}.csv'

        # Output information for user
        print("Output predictions to file:", fname) 

        
        # Output csv
        preds.to_csv(f'submission_{today}.csv', index=False)
        print("File saved")
        
    

makeSubmission(df_test, search.best_estimator_)
    
""" 2021-04-17 - you were writing the nametransformer - fit works, need to fix transform
    2021-06-19 - created make submission - haven't looked into the above issue"""
