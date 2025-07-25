# Project 1: Full ML Workflow

## 📁 Project Folder Structure
    my_classification_project/
    ├── data/
    │   ├── raw/        ← download here
    │   └── processed/
    ├── notebooks/      ← EDA & prototyping
    ├── src/
    │   ├── data.py     ← loading + cleaning
    │   ├── features.py ← preprocessing pipeline
    │   ├── train.py    ← model training & evaluation
    │   └── infer.py    ← inference with ONNX
    ├── models/
    │   ├── model.pkl
    │   └── model.onnx
    ├── requirements.txt
    ├── setup_venv.sh
    └── README.md

* data/raw – for raw input data
* data/processed – for cleaned/transformed data
* notebooks – for Jupyter notebooks
* src – for source code/scripts
* models – for saved machine learning models

<hr>

## 1. ✅ Setup & Git workflow
### 1.1 Initialise Git + venv
    mkdir my_classification_project && cd $_
    git init
    mkdir -p data/raw data/processed notebooks src models
    touch requirements.txt setup_venv.sh .gitignore README.md

<!-- Use touch setup_venv.sh only if you want to create an empty script file to edit later.
        Use bash setup_venv.sh  when you have a completed script ready and want to run it to perform the setup tasks 
        (e.g., create venv, install packages). -->

<!-- You need to create the setup_venv.sh script manually 
    (or via a command like your cat << EOF), 
    it is not created automatically by python -m venv. 
    python -m venv only creates the virtual environment folder (venv/), not any setup scripts.-->

### 1.2: Create .gitignore
    echo -e "venv/\n__pycache__/\n*.pyc\nmodels/\n*.onnx\n*.pkl\n.ipynb_checkpoints/\ndata/processed/" > .gitignore

### 1.3 Create setup_venv.sh:
<!--    touch setup_venv.sh 
or 
        cat << EOF > setup_venv.sh 
        > python -m venv venv .....
        ....
        > EOF    → this acts as end of text - save and exit CLI

-->
    python -m venv venv
    source venv/Scripts/activate
    pip install --upgrade pip setuptools
    pip install scikit-learn pandas numpy onnx onnxmltools onnxruntime matplotlib seaborn
    pip freeze > requirements.txt

### 1.3 Git Branching
    git checkout -b feature/data_loading
    <!-- add data.py, features.py -->
    git add .
    git commit -m "Add data loading & cleaning"
    git checkout main
    git merge feature/data_loading
<!-- deletes the local branch only if it has been merged. -->
    git branch -d feature/data_loading          


## 📥 2. Data import & cleaning (src/data.py)
<!-- src/data.py -->
    import pandas as pd
    from sklearn.model_selection import train_test_split

    def load_raw(path):
        df = pd.read_csv(path)
        return df

    def clean_data(df):
        df = df.dropna()  # drop missing
        # or df.fillna(...)
        return df

    def split(df, target='label', test_size=0.2, random_state=42):
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


## ⚙️ 3. Preprocessing & feature pipeline (src/features.py)
<!-- src/features.py -->
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    def build_preprocessor(numeric_cols, categorical_cols):
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipe, numeric_cols),
            ('cat', cat_pipe, categorical_cols)
        ])
        return preprocessor


## 🏋️‍♂️ 4. Training & evaluation (src/train.py)
<!-- src/train.py -->
    import joblib
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    from onnxmltools import convert_sklearn
    import onnx

    from data import load_raw, clean_data, split
    from features import build_preprocessor

    def train_and_evaluate(raw_csv):
        df = load_raw(raw_csv)
        df = clean_data(df)

        X_train, X_test, y_train, y_test = split(df, target='target')

        numeric = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
        categorical = X_train.select_dtypes(include=['object','category']).columns.tolist()
        pre = build_preprocessor(numeric, categorical)

        pipe = Pipeline([
            ('pre', pre),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1]
        print(classification_report(y_test, y_pred))
        print('ROC AUC:', roc_auc_score(y_test, y_proba))

        joblib.dump(pipe, '../models/model.pkl')
        return pipe, numeric, categorical

    def convert_to_onnx(pipe, numeric, categorical, output_path='../models/model.onnx'):
        initial_types = [('num', FloatTensorType([None, len(numeric)]))]
        onnx_model = convert_sklearn(pipe, initial_types=initial_types)
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

    if __name__ == '__main__':
        pipe, num, cat = train_and_evaluate('../data/raw/dataset.csv')
        convert_to_onnx(pipe, num, cat)


## 🔍 5. Hyperparameter tuning
<!-- Use GridSearchCV inside pipeline before .fit(): -->
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'clf__n_estimators': [100,200],
        'clf__max_depth': [None, 10, 20]
    }
    cv = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    cv.fit(X_train, y_train)
    print(cv.best_params_, cv.best_score_)
    pipe = cv.best_estimator_


## ▶️ 6. Inference using ONNX runtime (src/infer.py)
<!-- src/infer.py -->
    import numpy as np
    import pandas as pd
    import onnxruntime as rt

    def load_onnx(path):
        sess = rt.InferenceSession(path)
        return sess

    def predict(sess, df: pd.DataFrame):
        input_name = sess.get_inputs()[0].name
        X = df.values.astype(np.float32)
        return sess.run(None, {input_name: X})

    if __name__=='__main__':
        sess = load_onnx('../models/model.onnx')
        df = pd.read_csv('../data/processed/sample_input.csv')
        preds = predict(sess, df)
        print(preds)


## 🗂️ 7. EDA & notebooks
<!-- In notebooks/, include: -->

    * eda.ipynb: missing values, distributions, correlations, class balance.
    * features.ipynb: test preprocessing pipeline.
    * metrics.ipynb: ROC curve, confusion matrix.


## 🧠 8. ONNX optimisation
<!-- You can use onnxoptimizer to strip unused nodes: -->
    model = onnx.load('models/model.onnx')
    passes = ['eliminate_nop_transpose', 'eliminate_unused_initializer']
    opt_model = optimize(model, passes)
    onnx.save(opt_model, 'models/model_opt.onnx')


# End‑to‑End workflow
1. Data: place CSV in data/raw/
2. EDA: run notebooks → inspect.
3. Train: python src/train.py
4. Convert: produces .onnx
5. Infer: python src/infer.py
6. Hyper‑tune: integrate above.
7. Optimise ONNX.
8. Versioning: tag your commit after each major step.