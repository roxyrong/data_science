import pandas as pd
import json

pd.set_option("display.max_columns", 20)

#load data
loan_data = pd.read_csv('Decision Tree/lending-club-data.csv')
loan_data['safe_loans'] = loan_data['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
del loan_data['bad_loans']

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
            ]
target = 'safe_loans'

loan_data = loan_data[features + [target]]
loan_data = pd.get_dummies(loan_data)

with open('Decision Tree/module-5-assignment-2-train-idx.json') as json_file:
    train_idx = json.load(json_file)
with open('Decision Tree/module-5-assignment-2-test-idx.json') as json_file:
    test_idx = json.load(json_file)

train_data = loan_data.iloc[train_idx]
test_data = loan_data.iloc[test_idx]


def intermediate_node_num_mistakes(labels_in_node):
    if len(labels_in_node) == 0:
        return 0
    num_safe = sum(labels_in_node == 1)
    num_risky = sum(labels_in_node == -1)
    return min(num_safe, num_risky)


def best_splitting_feature(data, features, target):
    best_feature = None
    best_error = 10
    num_data_points = float(len(data))

    for feature in features:
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        left_mistakes = intermediate_node_num_mistakes(left_split[target])
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
        error = (left_mistakes + right_mistakes) / num_data_points
        if error < best_error:
            best_error = error
            best_feature = feature
    return best_feature


def create_leaf(target_values):
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}

    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1
    return leaf


def reached_minimum_node_size(data, min_node_size):
    if len(data) <= min_node_size:
        return True
    else:
        return False


def error_reduction(error_before_split, error_after_split):
    return max((error_before_split - error_after_split), 0)


def decision_tree_create(data, features, target, current_depth=0, max_depth=10, min_node_size=1, min_error_reduction=0.0):
    remaining_features = features[:]

    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

    # stopping condition
    if intermediate_node_num_mistakes(target_values) == 0:
        print("Stopping condition 1 reached. Node has no mistakes.")
        return create_leaf(target_values)
    if not remaining_features:
        print("Stopping condition 2 reached. No remaining features.")
        return create_leaf(target_values)

    # early stopping condition
    if reached_minimum_node_size(target_values, min_node_size):
        print("Early stopping condition 1 reached. Reached minimum node size.")
        return create_leaf(target_values)
    if current_depth >= max_depth:
        print("Early stopping condition 2 reached. Reached maximum depth.")
        return create_leaf(target_values)

    splitting_feature = best_splitting_feature(data, features, target)
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print("Early stopping condition 3 reached. Minimum error reduction.")
        return create_leaf(target_values)

    remaining_features.remove(splitting_feature)

    print("Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split)))

    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth,
                                     min_node_size, min_error_reduction)
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth,
                                      min_node_size, min_error_reduction)

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree}


# build a decision tree
X_train = train_data.drop('safe_loans', 1)
features_new = [col for col in X_train.columns]
my_decision_tree = decision_tree_create(train_data, features_new, target, current_depth=0, max_depth=6,
                                        min_node_size=100, min_error_reduction=0.0)


def classify(tree, x, annotate=False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


# test data classification
X_test_dict = test_data.drop('safe_loans', 1).to_dict(orient='records')
Y_test = list(test_data['safe_loans'])
Y_test_hat = []

for item in X_test_dict:
    Y_test_hat.append(classify(my_decision_tree, item, False))

accuracy = sum(1 for a, b in zip(Y_test, Y_test_hat) if a == b) / len(Y_test)
