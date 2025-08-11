
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
import statsmodels.api as sm
class Regression:

    def __init__(self):
        pass
    
    def commonRegress(self,X,y):
        X_train , X_test, y_train,y_test = train_test_split(X,y, test_size=.3,random_state=1234)
        self.lr = LinearRegression()
        self.lr.fit(X_train,y_train)
        # lr.coef_
        # lr.intercept_

        y_pred = self.lr.predict(X_test)
        y_true = y_test
        return mean_squared_error(y_true,y_pred) 
    
    def trainTestSplit(self,df,parm):
        df_train , df_test = train_test_split(df, test_size=.3,random_state=1234)
   
        model = sm.OLS.from_formula(parm,data=df_train)
        #상호작용 
        # model = sm.OLS.from_formula("species ~ sepal_length*sepal_width*petal_length*petal_width",data=df_train)
        # #2개끼리 상호작용 
        # model = sm.OLS.from_formula("species ~ sepal_length*sepal_width+petal_length*petal_width",data=df_train)
        
        result = model.fit()
        result.summary()
        y_pred = result.predict(df_test)

        return result ,mean_squared_error(df_test['MEDV'], y_pred)
   
    def predict(self,X_test):
        return self.lr.predict(X_test)
    
    def LRInfo(self):
        return self.lr 