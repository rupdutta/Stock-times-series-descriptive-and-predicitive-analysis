

# Loading Required Libraries 

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model,preprocessing
from scipy.stats import iqr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statistics import mean
from datetime import datetime as dt
from sklearn.preprocessing import PolynomialFeatures
from datetime import date
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import math



#Functions declarations
    
    #Function defined to validate date format taken from user throughout the program
def validate(date_text):
    try:
        if date_text != dt.strptime(date_text, "%Y-%m-%d").strftime('%Y-%m-%d'):
            raise ValueError
        return True
    except ValueError:
        return False

#Function defined to plot descriptive time series plot 
def plot_df_time_series(df, x, y, title="", xlabel='Date', ylabel='Closeing Price'):
    plt.figure(figsize=(8,6))
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

#Function defined to plot descriptive line trend plot
def linear_trend_lines():
    xs = stock_df['date_numeric']
    ys = stock_df['close']
        
    def slope_interc(xs,ys):
        m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
             ((mean(xs)*mean(xs)) - mean(xs*xs)))
        c = mean(ys) - m*mean(xs)
        return m, c

    m, c = slope_interc(xs,ys)
    regression_line = [(m*x)+c for x in xs]
    
    plt.figure(figsize=(8,6))
    plt.scatter(xs.apply(dt.fromordinal),ys,color='yellow')
    plt.plot(xs.apply(dt.fromordinal), regression_line)
    plt.gca().set(title='Linear Trend Line', xlabel='Date', ylabel='Closing Price')
    plt.show()

#Function defined to plot simple moving average 
def sma():
    rolling_mean = stock_df.close.rolling(window=20).mean()
    rolling_mean2 = stock_df.close.rolling(window=50).mean()
    plt.figure(figsize = (8,6))
    plt.plot(stock_df.date, stock_df.close, label=cmpny_name)
    plt.plot(stock_df.date, rolling_mean, label=cmpny_name+' 20 Day SMA', color='orange')
    plt.plot(stock_df.date, rolling_mean2, label=cmpny_name+' 50 Day SMA', color='magenta')
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend(loc='upper right')
    plt.show()

#Function defined to plot exponential moving average 
def ema():
    exp1 = stock_df.close.ewm(span=29, adjust=False).mean()
    exp2 = stock_df.close.ewm(span=9, adjust=False).mean()
    plt.figure(figsize = (8,6))
    plt.plot(stock_df.date, stock_df.close, label=cmpny_name)
    plt.plot(stock_df.date, exp1, label= cmpny_name+' 29 Day EMA')
    plt.plot(stock_df.date, exp2, label= cmpny_name+' 9 Day EMA')
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend(loc='upper left')
    plt.show()

#Function defined to plot simple,exponential and weighted moving average together 
def wma_sma_ema():
    weights = np.arange(1,11)
    wma10 = stock_df['close'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    sma10 = stock_df['close'].rolling(10).mean()
    ema10 = stock_df['close'].ewm(span=10).mean()
    plt.figure(figsize = (10,8))
    plt.plot(stock_df.date,stock_df['close'], label="close")
    plt.plot(stock_df.date,wma10, label="10-Day WMA")
    plt.plot(stock_df.date,sma10, label="10-Day SMA")
    plt.plot(stock_df.date,ema10, label="10 Day EMA")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.show() 

#Function defined to plot moving average convergence/divergence     
def macd():
    exp1 = stock_df.close.ewm(span=12, adjust=False).mean()
    exp2 = stock_df.close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    exp3 = macd.ewm(span = 9, adjust = False).mean()
    plt.figure(figsize = (8,6))
    plt.plot(stock_df.date, macd, label= 'MACD', color = 'Blue')
    plt.plot(stock_df.date, exp3, label='Signal Line', color='Red')
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend(loc='upper left')
    plt.show()        

#Function defined to navigate descriptive menu 
def descriptive():
    print("1. Normalized Closing Prices \n2. MEAN of the closing price \n3. RANGE of the closing price \n4. MEDIAN of the closing price \n5. MODE of the closing price \n6. KURTOSIS of the closing price \n7. SKEW of the closing price \n8. VARIANCE of the closing price \n9. STANDARD DEVIATION of the closing price \n10. Coefficient of Variation of the closing price \n11. Interquartile Range of the closing price \n12. 1st Quartile of the closing price \n13. 2nd Quartile of the closing price \n14. 3rd Quartile of the closing price \n\n DESCRIPTIVE VISUALISATION \n \n15. Time Series \n16. Linear Trend Lines \n17. SMA \n18. EMA \n19. WMA EMA SMA \n20. MACD \n21. Quit")
    desc_choice = input("Please choose option: ")
    while desc_choice != "21":
        if desc_choice == "1": # Normalized Values
            plt.plot(normalized_X[0])
            plt.ylabel('Normalized Values')
            plt.xlabel('Time')
            plt.title('Normalized Values Plot')
            plt.xticks(())
            plt.show()            
        elif desc_choice == "2": # Mean value
            print("The MEAN of the closing price is: " + str(Mean))
        elif desc_choice == "3": # Range
            print("The RANGE of the closing price is: " + str(Range))
        elif desc_choice == "4": # Median
            print("The MEDIAN of the closing price is: " + str(Med))
        elif desc_choice == "5": # Mode
            print("The MODE of the closing price: \n" + str(Mode)) 
        elif desc_choice == "6": # KURTOSIS
            print("The KURTOSIS of the closing price is: " + str(Kurt))
        elif desc_choice == "7": # Skew
            print("The SKEW of the closing price is: " + str(Skew))
        elif desc_choice == "8": # Variance
            print("The VARIANCE of the closing price is: " + str(Var))
        elif desc_choice == "9": # Std. Deviation
            print("The STANDARD DEVIATION of the closing price is: " + str(SD))
        elif desc_choice == "10": # Coeff Variation
            print("The Coefficient of Variation of the closing price is: " + str(Coeff_Var))
        elif desc_choice == "11": # Interquartile Range
            print("The Interquartile Range of the closing price is: " + str(IQR))
        elif desc_choice == "12": # 1st Quartile 
            print("The 1st Quartile of the closing price is " + str(*Quartiles_1))
        elif desc_choice == "13": # 2nd Quartile
            print("The 2nd Quartile of the closing price is " + str(*Quartiles_2))
        elif desc_choice == "14": # 3rd Quartile
            print("The 3rd Quartile of the closing price is " + str(*Quartiles_3))
        elif desc_choice == "15": # Time Series
            plot_df_time_series(stock_df, x=stock_df.date, y=stock_df.close, title='Stock Raw Time Series Analysis')   
        elif desc_choice == "16": # Linear Trend Lines
            linear_trend_lines() 
        elif desc_choice == "17": # Simple Moving Average
            sma()
        elif desc_choice == "18": # Exponential Moving Average
            ema()
        elif desc_choice == "19": # Weighted Moving Average
            wma_sma_ema()
        elif desc_choice == "20": # Weighted Moving Average
            macd()
        else: # Wrong choice
            print("Wrong choice, please try again.")
        desc_choice = input("\n1. Normalized Closing Prices \n2. MEAN of the closing price \n3. RANGE of the closing price \n4. MEDIAN of the closing price \n5. MODE of the closing price \n6. KURTOSIS of the closing price \n7. SKEW of the closing price \n8. VARIANCE of the closing price \n9. STANDARD DEVIATION of the closing price \n10. Coefficient of Variation of the closing price \n11. Interquartile Range of the closing price \n12. 1st Quartile of the closing price \n13. 2nd Quartile of the closing price \n14. 3rd Quartile of the closing price \n\n DESCRIPTIVE VISUALISATION \n \n15. Time Series \n16. Linear Trend Lines \n17. SMA \n18. EMA \n19. WMA EMA SMA \n20. MACD \n21. Quit \n\nPlease choose option: ")

#Function defined to plot linear regression 
def linear_reg():
    df_X=stock_df.iloc[:,13:14]
    df_Y=np.array(stock_df['close'])

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=0)
    
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    print("\nModel is Trained for given period. \n1. Press 1 for trained data statistics (e.g. Root Mean Squared Error, R-Squared Value) \n2. Press 2 for predicting closing price \n3. Quit")
    choice = input("Please choose option: ")
    
    while choice != "3":
        if choice == "1": # RMSE and R-square value
            print("Root Mean Squared Error: %.2f"
                  % np.sqrt(mean_squared_error(y_test, y_pred)))
            print('R-sqaure: %.2f' % r2_score(y_test, y_pred))
            
            # Plot outputs
            plt.figure(figsize = (8,6))
            plt.scatter(X_test['date_numeric'].apply(dt.fromordinal), y_test,  color='black')
            plt.plot(X_test['date_numeric'].apply(dt.fromordinal), y_pred, color='blue', linewidth=3)
            plt.gca().set(title='Linear Regression with Signle Independent Variable')
            plt.xlabel("Date")
            plt.ylabel("Closing Price")
            plt.show()
        
        elif choice == "2": # Predicting Value for a given date
            d=input('please input a date in yyyy-mm-dd format for prediction : ')
            if validate(d) == True:
                d=dt.strptime(d, '%Y-%m-%d').date()
                d= d.toordinal()
                print(*regr.predict([[d]]))
            else:
                print("\nPlease enter a valid date and try again")

        else: # Wrong choice
            print("\nWrong choice, please try again.")
        choice = input("Model is Trained for given period. \n1. Press 1 for trained data statistics (e.g. coefficient, mean squared error, variance score, intercept value) \n2. Press 2 for predicting closing price \n3. Quit \n\nPlease provide option ")

def ploynomial_regression():
    X =stock_df.iloc[:,13:14]
    Y =np.array(stock_df['close'])
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    # Training the model
    
    poly_reg = PolynomialFeatures(degree=2)
    X_train_poly = poly_reg.fit_transform(X_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_train_poly, y_train)
    
    y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))
   
    #RMSE and R-sqaured value
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_test = r2_score(y_test, y_pred)
    
    print('Root Mean Squared Error for given training period is {}'.format(rmse_test))
    print('R-Squared value for given traing period is {}'.format(r2_test))

    #PLotting the graph
    plt.figure(figsize = (8,6))
    plt.plot(X_test['date_numeric'].apply(dt.fromordinal).sort_values(ascending=True), y_test, linewidth=3, label = 'Actual') 
    plt.plot(X_test['date_numeric'].apply(dt.fromordinal).sort_values(ascending=True), y_pred,color = 'blue', linewidth=3, label = 'Predicted') 
    plt.gca().set(title='Non-Linear Polynomial (Quadratic) Regression')
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend(['Actual', 'Predicted'])
    plt.show()
    
    d=input('please input a date in yyyy-mm-dd format for prediction : ')
    if validate(d) == True:
        d=dt.strptime(d, '%Y-%m-%d').date()
        d= d.toordinal()
        y_pred = lin_reg_2.predict(poly_reg.fit_transform([[d]]))
        print(y_pred)
    else:
        print("\nPlease enter a valid date and try again")

def arima():
    arima_date=np.array(stock_df['date'])
    arima_close=np.array(stock_df['close'])
            
    x = list(arima_close)
    xdiff = [x[n+1]-x[n] for n in range(0,len(x)-1)]
    xdiff.append(-1*arima_close[-1,])
    
    arima_df_before= pd.DataFrame(
        {'Date': list(arima_date),
         'ClosePrice': stock_df['close']
        })
    
    arima_df_after = pd.DataFrame(
        {'Date': list(arima_date),
         'CloseDiff': xdiff
        })
    
    # Plot ACF (Auto Correlation Function) Before Stationarizing the Time Series
    plot_acf(arima_df_before['ClosePrice'])
    plt.ylabel('Closing Price')
    plt.title('ACF before Stationarizing the Time Series')
    pyplot.show()
    
    #Plot ACF after Stationarize the Time Series 
    plot_acf(arima_df_after['CloseDiff'])
    plt.ylabel('Closing Price Difference')
    plt.title('ACF after Stationarizing the Time Series')
    pyplot.show()
    
    #Plot PACF (Partial Auto Correlation Function) Before Stationarizing the Time Series
    plot_pacf(arima_df_before['ClosePrice'], lags=5)
    plt.ylabel('Closing Price')
    plt.title('PACF before Stationarizing the Time Series')
    pyplot.show()
    
    #Plot PACF after Stationarize the Time Series 
    plot_pacf(arima_df_after['CloseDiff'], lags=5)
    plt.ylabel('Closing Price Difference')
    plt.title('PACF after Stationarizing the Time Series')
    pyplot.show()
    
    print("Costrution of ARIMA model with p and q values as 1. These optimal values obtained as per the ACF and PACF plot")
    print("Costrution of ARIMA model with d value as 1. This  value obtained as we used 1 level of differencing to get the optimal values of p and q")
    
    # fit ARIMA model
    model = ARIMA(arima_df_after['CloseDiff'], order=(1,1,1))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot(kind='kde')
    pyplot.show()
    plt.title('Density Plot')
    print(residuals.describe())
   
    #Forcasting using ARIMA Model 

    def parser(x):
    	return datetime.strptime(x, '%d/%m/%Y')

    mydata= arima_df_after['CloseDiff']
    X = mydata.values
    size = int(len(X) * 0.85)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
    	model = ARIMA(history, order=(1,1,1))
    	model_fit = model.fit(disp=0)
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	history.append(obs)
    
    MSE = mean_squared_error(test, predictions)
    RMSE = math.sqrt(MSE) 
    print('\nRMSE for the Test Data is: %.2f' % RMSE)
    
    
    # make prediction for a future date range
    date_new = date(2020, 11, 29)
    date_today = date.today()
    delta = date_new - date_today
    pred2 = model_fit.forecast(steps=delta.days) [0]
    pyplot.plot(pred2)
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.xticks(())
    plt.title('Prediction Plot for the next one year prediction')
    pyplot.show()

#Defining function to navigate predictive menu         
def predictive():
    print("1. Linear Regression with single independent variable \n2. Arima Time Series Model \n3. Polynomial Regression \n4. Quit")
    pre_choice = input("Please choose option: ")
    while pre_choice != "4":
        if pre_choice == "1": # Linear Regression with single variable
            linear_reg()
        elif pre_choice == "2": # Arima Time Series Model
            arima()
        elif pre_choice == "3": # Non-Linear Regression ploynomial
            ploynomial_regression()
        else: # Wrong choice
            print("Wrong choice, please try again.")
        pre_choice = input("1. Linear Regression with single independent variable \n2. Arima Time Series Model \n3. Polynomial Regression \n4. Quit \n\nPlease choose option: ")

def analytics_menu():
    print("1. Descriptive Analytics\n2. Predictive Analytics \n3. Quit")
    choice = input("Please choose option: ")
    
    while choice != "3":
        if choice == "1": # Descriptive Analytics
            descriptive()
        elif choice == "2": # Predictive Analytics
            predictive()
        else: # Wrong choice
            print("Wrong choice, please try again.")
        choice = input("\n1. Descriptive Analytics\n2. Predictive Analytics \n3. Quit \n\nPlease choose option: ")


#Main Program


if __name__ == '__main__':
    """
    ==================================================================
    Project Name:  Stock Price Analytics 
    Author: Rupam Dutta
    ==================================================================
    
    """
    print(__doc__)
    row = "| {:10} | {:10} |"
    print("{:-^70}".format("Programming for Analytics"))
    print("{:-^70}".format("Assignment: Python Project"))
    print("{:^70}".format(""))
    print("{:^70}".format(""))
    print("{:^70}".format(""))
    print("{:-^70}".format("Welcome to the Stock Prediction Model"))
    print("{:^70}".format(""))

    
    print("\n1. Company Symbol Name & Date Window \n2. Company List \n3. Quit ")
    choice1 = input("Please choose option: ")
    
    while choice1 != "3":
        if choice1 == "1": # Validating Company Name and Date Window                        
            cmpny_listdata=pd.read_csv("https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download",usecols=[0,1,2,3,4,5])
            cmpny_name = input('Please input the Company Symbol name : ').upper()
            start_date= input('Please provide start date for analytics in yyyy-mm-dd format : ').upper()
            end_date= input('Please provide end date for analytics in yyyy-mm-dd format : ').upper()
            
            if validate(start_date) == True and validate(end_date) == True and dt.strptime(end_date, "%Y-%m-%d") > dt.strptime(start_date, "%Y-%m-%d") and dt.strptime(start_date, "%Y-%m-%d") > dt.strptime('1988-01-01', "%Y-%m-%d") and dt.strptime(end_date, "%Y-%m-%d") < dt.strptime(str(date.today()), "%Y-%m-%d") :
                #Date - user interface
                api_token = 'Token f08e73c61f222d51c450e3847a80877658d63fdf'
                api_website = 'https://api.tiingo.com/tiingo/daily/'+cmpny_name+'/prices?startDate='+start_date+'%20&endDate='+end_date+''
    
                # Gather data using API Toekn
                headers = {
                        'Content-Type': 'application/json',
                        'Authorization' : api_token
                        }
                
                response_req = requests.get(api_website , headers=headers)
                json_data = response_req.json()
                
                # Making Dataframe
                if len(cmpny_listdata[cmpny_listdata.Symbol==cmpny_name]) == 1 and len(json_data) > 0:
                    print("\n Company is registered, ready to proceed.\n")
                    x=[]
                    y=[]
                    count=0
                    for i in json_data:
                        count+=1
                        if count!=0:
                            for a,b in i.items():
                                if count == 1:
                                    x.append(a)
                                    y.append(b)
                                else:
                                    y.append(b)
                               
                    d=np.array(y)
                    d.shape=(int(len(d)/len(x)),len(x))
                               
                    stock_df = pd.DataFrame(d,columns=x)
                    stock_df['date'] = pd.to_datetime(stock_df['date'])
                    stock_df = stock_df.apply(pd.to_numeric)
                    stock_df['date'] = pd.to_datetime(stock_df['date'])
                    stock_df['date_numeric'] = stock_df['date'].apply(lambda x: x.toordinal())
                    
                    # Descriptive Analytics
                    x_array = np.array(stock_df['close'])
                    normalized_X = preprocessing.normalize([x_array])
                    Mean = stock_df["close"].mean()         
                    Range = stock_df["close"].max() - stock_df["close"].min()
                    Med = stock_df["close"].median()
                    Mode = stock_df["close"].mode()
                    Kurt = stock_df["close"].kurtosis()
                    Skew = stock_df["close"].skew()
                    Var = stock_df["close"].var()
                    SD = np.std(stock_df["close"])
                    Coeff_Var = (SD / Mean) * 100
                    IQR = iqr(stock_df["close"])
                    Closing = stock_df["close"]
                    Quartiles_1 = np.percentile(Closing, [25]) 
                    Quartiles_2= np.percentile(Closing, [50]) 
                    Quartiles_3= np.percentile(Closing, [75])
                    print(stock_df.dtypes)
                    analytics_menu()
                else:
                    print("\n Wrong choice, Company name is not registered, please try again.\n")
            else:
                print("\nSorry can't proceed further, due to following reasons: \n 1.Date Format is not valid \n 2.Training Start Date is either past 30 years or greater than Training End Date \n 3.Training End Date is greater than today's date\n\nPlease try again..")
        elif choice1 == "2": # Company List
            pd.set_option('display.max_rows', None)
            print(pd.read_csv("https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download",usecols=[0,1],index_col=0))
        else: 
            print("Wrong choice, please try again.")
        choice1 = input("\n1. Company Symbol Name and Date Window \n2. Company List  \n3. Quit \n\nPlease choose option: ")
    
    
    














