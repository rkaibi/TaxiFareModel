# imports
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = self.set_pipeline()
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        self.distance_col = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
        self.time_col = ['pickup_datetime']

        dist_pipe = Pipeline([
            ('dist', DistanceTransformer()),
            ('scaler', StandardScaler())
        ])

        time_pipe = Pipeline([
            ('convert_time', TimeFeaturesEncoder('pickup_datetime')),
            ('encode', OneHotEncoder())
        ])

        preproc_pipe = ColumnTransformer([
            ('dist_transformer', dist_pipe, self.distance_col),
            ('time_transformer', time_pipe, self.time_col)
        ])

        model_pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('model', LinearRegression())
        ])
        return model_pipe

    def run(self):
        """set and train the pipeline"""
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        fited_model = self.run()
        y_pred = fited_model.predict(X_test)

        return(compute_rmse(y_pred, y_test))


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns=['key', 'fare_amount'])
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    t = Trainer(X_train, y_train)
    t.run()
    # evaluate
    t.evaluate(X_test, y_test)
    print(t.evaluate(X_test, y_test))
