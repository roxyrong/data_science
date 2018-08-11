import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

pd.set_option("display.max_columns", 20)

#load data
loan_data = pd.read_csv('Decision Tree/lending-club-data.csv')
loan_data['safe_loans'] = loan_data['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loan_data['bad_loans']

features = ['grade',                     # grade of the loan
           #'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            ]
target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

loan_data = loan_data[features + [target]]

# deal with categorical data
obj_df = loan_data.select_dtypes(include=['object']).copy()
grade = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
obj_df['grade'] = obj_df['grade'].replace(grade)
obj_df['home_ownership'] = obj_df['home_ownership'].astype('category').cat.codes
obj_df['purpose'] = obj_df['purpose'].astype('category').cat.codes
obj_df['term'] = obj_df['term'].astype('category').cat.codes

categorical_col = obj_df.columns
loan_data[categorical_col] = obj_df


with open('Decision Tree/module-5-assignment-1-train-idx.json') as json_file:
    train_idx = json.load(json_file)
with open('Decision Tree/module-5-assignment-1-validation-idx.json') as json_file:
    validation_idx = json.load(json_file)

train_data = loan_data.iloc[train_idx]
validation_data = loan_data.iloc[validation_idx]

# train model
X = train_data[features].values
Y = train_data[target].values
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, Y)

# calc accuracy on validation set
X_valid = validation_data[features].values
Y_valid = validation_data[target].values
Y_hat = clf.predict(X_valid)
accuracy = sum(1 for a, b in zip(Y_valid, Y_hat) if a == b) / len(Y_valid)

# plot decision tree
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=features, class_names=target,
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('Decision Tree/simple_tree', view=True)
