
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

from sklearn.tree import export_text

col_names = ['Refactor','Restructure','Rewrite','Introduce','Simplify','Extend','Optimize','Split','Cleanup','Replace','Improve','Extract','Reduce','Move','Change','Import','Enhance','Modify','Remove','Better','Avoid','Add','Fix','Smell','Outcome']

#reimplementation note for phase 2: crucial data unavailable for training decision tree classifiers
df = pd.read_csv("dt_rules2.csv", engine='python')



feature_cols = ['Refactor','Restructure','Rewrite','Introduce','Simplify','Extend','Optimize','Split','Cleanup','Replace','Improve','Extract','Reduce','Move','Change','Import','Enhance','Modify','Remove','Better','Avoid','Add','Fix','Smell']

X = df[feature_cols] # Features
y = df.Outcome # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

######################################################################Display Tree
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DecisionTree.png')
Image(graph.create_png())

############################################################################# print rules
r = export_text(clf, feature_cols, decimals=0,spacing=1)
print(r)
