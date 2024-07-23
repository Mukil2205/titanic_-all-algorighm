# titanic_-all-algorighm
{

 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bf2ab7c5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-07-22T14:29:41.338397Z",
     "iopub.status.busy": "2024-07-22T14:29:41.338007Z",
     "iopub.status.idle": "2024-07-22T14:29:44.528236Z",
     "shell.execute_reply": "2024-07-22T14:29:44.526927Z"
    },
    "papermill": {
     "duration": 3.196403,
     "end_time": "2024-07-22T14:29:44.530566",
     "exception": false,
     "start_time": "2024-07-22T14:29:41.334163",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5\n",
      "  warnings.warn(f\"A NumPy version >={np_minversion} and <{np_maxversion}\"\n",
      "/opt/conda/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/conda/lib/python3.10/site-packages/sklearn/ensemble/_base.py:166: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "/opt/conda/lib/python3.10/site-packages/sklearn/ensemble/_base.py:166: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 0.7988826815642458\n",
      "K-Nearest Neighbors Accuracy: 0.7094972067039106\n",
      "Bagged Decision Trees Accuracy: 0.8100558659217877\n",
      "Random Forest Accuracy: 0.8044692737430168\n",
      "AdaBoost Accuracy: 0.776536312849162\n",
      "XGBoost Accuracy: 0.7988826815642458\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Load the dataset\n",
    "train_data = pd.read_csv('/kaggle/input/titanic/train.csv')\n",
    "test_data = pd.read_csv('/kaggle/input/titanic/test.csv')\n",
    "\n",
    "# Feature engineering and selection\n",
    "features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']\n",
    "\n",
    "# Define preprocessing steps\n",
    "numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='mean'))\n",
    "])\n",
    "\n",
    "categorical_features = ['Sex', 'Embarked']\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "    ('encoder', OneHotEncoder(handle_unknown='ignore'))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numeric_features),\n",
    "        ('cat', categorical_transformer, categorical_features)\n",
    "    ])\n",
    "\n",
    "# Preprocess the data\n",
    "X = preprocessor.fit_transform(train_data[features])\n",
    "y = train_data['Survived']\n",
    "\n",
    "# Split the data into train and validation sets\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Initialize models\n",
    "logistic_model = LogisticRegression()\n",
    "knn_model = KNeighborsClassifier()\n",
    "bagged_tree_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)\n",
    "random_forest_model = RandomForestClassifier(n_estimators=100)\n",
    "adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)\n",
    "xgboost_model = XGBClassifier()\n",
    "\n",
    "# Train models\n",
    "logistic_model.fit(X_train, y_train)\n",
    "knn_model.fit(X_train, y_train)\n",
    "bagged_tree_model.fit(X_train, y_train)\n",
    "random_forest_model.fit(X_train, y_train)\n",
    "adaboost_model.fit(X_train, y_train)\n",
    "xgboost_model.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions\n",
    "logistic_preds = logistic_model.predict(X_val)\n",
    "knn_preds = knn_model.predict(X_val)\n",
    "bagged_tree_preds = bagged_tree_model.predict(X_val)\n",
    "random_forest_preds = random_forest_model.predict(X_val)\n",
    "adaboost_preds = adaboost_model.predict(X_val)\n",
    "xgboost_preds = xgboost_model.predict(X_val)\n",
    "\n",
    "# Evaluate models\n",
    "print(\"Logistic Regression Accuracy:\", accuracy_score(y_val, logistic_preds))\n",
    "print(\"K-Nearest Neighbors Accuracy:\", accuracy_score(y_val, knn_preds))\n",
    "print(\"Bagged Decision Trees Accuracy:\", accuracy_score(y_val, bagged_tree_preds))\n",
    "print(\"Random Forest Accuracy:\", accuracy_score(y_val, random_forest_preds))\n",
    "print(\"AdaBoost Accuracy:\", accuracy_score(y_val, adaboost_preds))\n",
    "print(\"XGBoost Accuracy:\", accuracy_score(y_val, xgboost_preds))\n",
    "\n",
    "# Train the selected model on the entire training data and make predictions on the test set\n",
    "selected_model = xgboost_model  # Change this to the best-performing model\n",
    "selected_model.fit(X, y)\n",
    "test_features = preprocessor.transform(test_data[features])\n",
    "test_preds = selected_model.predict(test_features)\n",
    "\n",
    "# Prepare submission file\n",
    "submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_preds})\n",
    "submission.to_csv('submission.csv', index=False)\n"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "databundleVersionId": 26502,
     "sourceId": 3136,
     "sourceType": "competition"
    }
   ],
   "dockerImageVersionId": 30527,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 18.21074,
   "end_time": "2024-07-22T14:29:48.137499",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-07-22T14:29:29.926759",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
