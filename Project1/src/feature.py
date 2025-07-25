from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# numeric_cols = list of numeric column names
# categorical_cols = list of categorical column names

def build_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
                            ('imputer', SimpleImputer(strategy='mean')),            # ('step_name', transformer)
                            ('scaler', StandardScaler())                            # ('step_name', transformer)
                        ])
    cat_pipe = Pipeline([
                            ('imputer', SimpleImputer(strategy='most_frequent')),   # ('step_name', transformer)
                            ('OHE', OneHotEncoder())                                # ('step_name', transformer)
                        ])
    preprocessor = ColumnTransformer([
                                       ('num', 'num_pipe', 'numeric_cols'),         # ('step_name', transformer, columns)
                                       ('col', 'cat_pipe', 'categorical_cols')      # ('step_name', transformer, columns)
                                     ])
    
'''
1. Input:
- numeric_cols: list of numeric column names
- categorical_cols: list of categorical column names

2. Numeric pipeline (num_pipe):
- SimpleImputer(strategy='mean'):     fills missing values with the mean
- StandardScaler():                   scales features to have mean=0 and std=1 (normalisation)

3. Categorical pipeline (cat_pipe):
- SimpleImputer(strategy='most_frequent'): fills missing values with the most frequent category
- OneHotEncoder(handle_unknown='ignore'):  converts categories to binary vectors, ignoring unseen categories at test time

4. Combine both with ColumnTransformer:
- Applies num_pipe to numeric columns
- Applies cat_pipe to categorical columns

5. Returns:
- A combined transformer that can be used in fit() and transform() steps for ML models

'''