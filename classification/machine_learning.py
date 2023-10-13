import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight



def svm_classification(train_x, train_y, test_x, test_y, scaler=None, model=None, do_shuffle=True):
    # Shuffling data
    #if do_shuffle is True:
    #    train_x, train_y = \
    #        shuffle(train_x, train_y, random_state=10)

    #print("number of 1 in train_y is ", np.count_nonzero(train_y == 1))
    #print("number of 0 in train_y is ", np.count_nonzero(train_y == 0))

    # scaling
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(train_x)
    else:
        scaler.partial_fit(train_x)
    # Fit on training set only.
    # Apply transform to both the training set and the test set.
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    if model is None:
        # SGD with this parameters is linear SVM
        class_weight = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_y)
        class_weight_dict = {0: class_weight[0],
                             1: class_weight[1]}
        model = SGDClassifier(loss="hinge",
                              penalty='l2',
                              class_weight=class_weight_dict,
                              max_iter=1000,
                              tol=1e-3,
                              shuffle=True,
                              warm_start=True,
                              random_state=10)
        #clf = svm.SVC(C=150, kernel="poly", degree=2, gamma="auto", probability=True)
        # clf = KNeighborsClassifier(n_neighbors=3)
        model = RandomForestClassifier(n_estimators=200, warm_start=True, class_weight=class_weight_dict, random_state=10)
        # clf = AdaBoostClassifier(n_estimators=100, learning_rate=1)
        #clf = GaussianNB()
        # clf = QuadraticDiscriminantAnalysis()
        # clf = LinearDiscriminantAnalysis()
        # clf = MLPClassifier()
        # clf = LinearDiscriminantAnalysis()
        model.fit(train_x, train_y)
    else:
        model.partial_fit(train_x, train_y)
    #print("prediction", clf.predict(test_x))
    try:
        pred_values = model.predict_proba(test_x)
        pred_values= np.argmax(pred_values, axis=1)
    except:
        pred_values= model.predict(test_x)
    
    acc = accuracy_score(pred_values, test_y)
    #print(classification_report(test_y, pred_values))
    return model, scaler, acc

def rf_classification(train_x, train_y, test_x, test_y, scaler=None, model=None, do_shuffle=True, just_detection=False):
    # Shuffling data
    #if do_shuffle is True:
    #    train_x, train_y = \
    #        shuffle(train_x, train_y, random_state=10)

    #print("number of 1 in train_y is ", np.count_nonzero(train_y == 1))
    #print("number of 0 in train_y is ", np.count_nonzero(train_y == 0))

    # scaling
    if just_detection is True:
        try:
            pred_values = model.predict_proba(test_x)
            pred_values= np.argmax(pred_values, axis=1)
        except:
            pred_values= model.predict(test_x)
    else:
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(train_x)
        else:
            scaler.partial_fit(train_x)
        # Fit on training set only.
        # Apply transform to both the training set and the test set.
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        if model is None:
            class_weight = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_y)
            class_weight_dict = {0: class_weight[0],
                                1: class_weight[1]}
            model = RandomForestClassifier(n_estimators=200, warm_start=True, class_weight=class_weight_dict, random_state=10)
        else:
            model.n_estimators += 200

        model.fit(train_x, train_y)
        #print("prediction", clf.predict(test_x))
        try:
            pred_values = model.predict_proba(test_x)
            pred_values= np.argmax(pred_values, axis=1)
        except:
            pred_values= model.predict(test_x)
    
    acc = accuracy_score(pred_values, test_y)
    #print(classification_report(test_y, pred_values))
    return model, scaler, acc