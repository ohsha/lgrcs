import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import config_lgrcs as config
from my_tools import dev_tools as dvt
from my_tools.decision_tree import DecisionTree


class ModelHandler():

    def __init__(self, user):
        self.user = user

    def data_handling(self):
        """
        Loading the data from the csv and set the landmarks as X and the label as y.
        :return: X: landmarks , y: label
        """
        data = pd.read_csv(self.user.csv_path)
        X = pd.iloc[:, 0:-2]
        y = pd.iloc[:, -1]
        return X, y

    def _extract_features(self, export_dataset=True):
        """
        calculates new features from the raw data
        :param X: landmarks
        :param y:  labels
        :return: X - a new dataset  that contains the new features
                 y - labels
        """
        y = self.user.df['label']

        inner_idx = config.LANDMARKS_INDEXES['Inner_lips']
        outer_idx = config.LANDMARKS_INDEXES['Outer_lips']

        inner_col = range(inner_idx[0], inner_idx[1])
        outer_col = range(outer_idx[0], outer_idx[1])

        # fixing the dtypes of the arrays for the cv2 functions use
        inner = np.array(self.user.df[inner_col].to_numpy().tolist())
        outer = np.array(self.user.df[outer_col].to_numpy().tolist())

        inner_area = [dvt.calculate_area(i) for i in inner] # contains 8 landmarks
        outer_area = [dvt.calculate_area(i) for i in outer] # contains 12 landmarks
        in2out_ratio = dvt.get_ratio(inner_area, outer_area)

        # landmark #48 gets the index 0
        width = dvt.get_distance(outer[:,0], outer[:,6], axis=1) # landmarks: #48 & #54
        height = dvt.get_distance(outer[:,3], outer[:,9], axis=1) # landmarks: #51 & #57
        h2w_ratio = dvt.get_ratio(height, width)

        vector_u = inner[:, 4] - inner[:, 3]
        vector_v = inner[:, 4] - inner[:, 5]
        angle = [dvt.get_angle(vector_u[i], vector_v[i]) for i in range(inner.shape[0])]

        columns = ['height', 'width', 'h2w','innArea','outArea', 'in2out','angle' ,'label']
        to_df = np.array([height, width, h2w_ratio, inner_area, outer_area, in2out_ratio, angle, y]).T
        dataset = pd.DataFrame(data=to_df, columns=columns)

        pickle_path = os.path.join(self.user.parent_dir, 'processing_dataset.pkl')
        dataset.to_pickle(pickle_path)

        csv_path = os.path.join(self.user.parent_dir, 'processing_dataset.csv')
        dataset.to_csv(csv_path)

        print('[INFO] feature extracted was written to csv and pickle files')
        return dataset

    def fit_predict(self, export_summary=True):
        dataset = self._extract_features(export_dataset=True)

        count_outliers = {}
        # drop outlier samples
        for gesture in config.GESTURES:
            df_ = dataset[dataset['label'] == gesture]
            outliers_to_drop = dvt.detect_outliers(df_, n_appears=1, features=['height', 'width', 'innArea', 'outArea'])
            dataset = dataset.drop(outliers_to_drop, axis=0)
            count_outliers['gesture'] = len(outliers_to_drop)
            print(outliers_to_drop)

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=config.TEST_SIZE ,random_state=42)

        n_nodes = X.shape[1] + 1
        # classifier = DecisionTree(max_depth=self.n_nodes)
        # self.classifier = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=n_nodes)
        self.classifier = DecisionTree(max_depth=n_nodes)
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=config.GESTURES)

        print(cr)
        print(f'accuracy: {acc}')

        if export_summary:
            self.export_summary(cr, acc, count_outliers)

    def export_model(self):
        with open(self.user.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def export_summary(self, cr, acc, outliers):

        path = os.path.join(self.user.parent_dir ,'summary.txt')
        hash_ = '#'*30

        with open(path, 'w') as f:
            f.write('\n')
            f.write(f'  {hash_}\tFINAL REPORT\t{hash_}\n\n\n')
            f.write(f'User: {self.user.name}\n\n')
            f.write(f'Collected: {config.NUM_ITER * len(config.GESTURES)} gestures.\n\n')
            f.write(f'Number of outliers:')
            [f.write(f'\t{gst}: {cnt}') for gst, cnt in outliers.items()]

            f.write(f'\n')
            f.write(f'Model Accuracy: {acc}\n\n')
            f.write('\n' + cr + '\n')
            f.write('\n')
