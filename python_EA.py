import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API
import pandas as pd
import tkinter
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import openai
import chronological


api = API(access_token="<YOUR ACCESS TOKEN>")


def get_data(instrument):

        params = {"granularity": "D", "count": 200}

        r = instruments.InstrumentsCandles(
            instrument=instrument, params=params)

        api.request(r)

        df = pd.DataFrame(r.response['candles'])[['time', 'volume', 'mid']]

        df['time'] = pd.to_datetime(df['time'])

        df['mid'] = df['mid'].apply(lambda x: (
            x['o'] + x['h'] + x['l'] + x['c']) / 4)

        return df


def preprocess_data(df):

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(df)

        return scaled_data


def build_model():

            model = Sequential()

            model.add(LSTM(50, return_sequences=True))

            model.add(Dropout(0.2))

            model.add(LSTM(50))

            model.add(Dropout(0.2))

            model.add(Dense(1))

            model.compile(loss='mean_squared_error', optimizer='adam')

            return model


def sentiment_analysis(text):

        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(text)
        return sentiment['compound']


def calculateEntryPoints(data):
    """Calculates entry points for trades using the NNFX system and deep learning.

    Parameters:
    data (pandas DataFrame): Historical data from OANDA API

    Returns:
    entryPoints (list): List of entry points for trades.
    """

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(data)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(scaledData.shape[1], 1)))

    model.add(LSTM(units=50))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(scaledData, batch_size=1, epochs=1)

    predictions = model.predict(scaledData)

    entryPoints = []

    for i in range(len(predictions)):

        if predictions[i] > 0:

            entryPoints.append(i)

    return entryPoints


def calculateExitPoints(data):
    """Calculates exit points for trades using the NNFX system and deep learning.

    Parameters:
    data (pandas DataFrame): Historical data from OANDA API

    Returns:
    exitPoints (list): List of exit points for trades.
    """

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(data)

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(scaledData.shape[1], 1)))

    model.add(LSTM(units=50))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(scaledData, batch_size=1, epochs=1)

    predictions = model.predict(scaledData)

    exitPoints = []

    for i in range(len(predictions)):

        if predictions[i] < 0:

            exitPoints.append(i)
    return exitPoints


def riskManagementRules():
    """Implements risk management rules developed over 30 years of trading experience."""

    maxRiskPerTrade = 0.01 * accountBalance

    maxPositionSize = 0.05 * accountBalance

    stopLoss = 10 * pipValue

    return maxRiskPerTrade, maxPositionSize, stopLoss


def moneyManagementRules():
    """Implements money management rules developed over 30 years of trading experience."""

    positionSize = (maxRiskPerTrade / stopLoss) * pipValue

    if positionSize > maxPositionSize:

       positionSize = maxPositionSize

    return positionSize


def economicCalendarData():
    """Retrieves economic calendar data from Trading Economics."""

    r = requests.get('http://docs.tradingeconomics.com/')

    data=r.json()

    return data
def newsEventsData():
    """Retrieves news events data from CNN, Fox, and Sky News."""

    cnn_url='https://www.cnn.com/'
    fox_url='https://www.foxnews.com/'
    sky_url='https://news.sky.com/'

    cnn_data=requests.get(cnn_url).text
    fox_data=requests.get(fox_url).text
    sky_data=requests.get(sky_url).text

    return cnn_data, fox_data, sky_data

def start_trading():

    instruments=instrumentsEntry.get()
    orderTypes=orderTypesEntry.get()


    df=get_data(instruments)


    scaledData=preprocess_data(df)


    model=build_model()


    entryPoints=calculateEntryPoints(scaledData)


    exitPoints=calculateExitPoints(scaledData)


    riskManagementRules()


    moneyManagementRules()              


    sentimentScore=sentimentAnalysis()


    economicCalendarData=economicCalendarData()


    newsEventsData=newsEventsData()

    print("Trading process started...")


def stop_trading():
    """
    This function stops trading when the stop_button is 
    pressed.
    """
    print("Stopping trading...")
    global is_trading 
    is_trading = False

root=tkinter.Tk()
root.title("Exert Advisor")
root.geometry("500x500")

tkinter.Label(root, text="Trading Instruments:").grid(row=0)
tkinter.Label(root, text="Order Types:").grid(row=1)
tkinter.Label(root, text="Risk Management Rules:").grid(row=2)
tkinter.Label(root, text="Money Management Rules:").grid(row=3)
tkinter.Label(root, text="Sentiment Analysis:").grid(row=4)
tkinter.Label(root, text="Economic Calendar Data:").grid(row=5)
tkinter.Label(root, text="News Events Data:").grid(row=6)
 

# Add buttons to start and stop the trading process  
start_button = tkinter.Button(text='Start', width=25, command=start_trading)   # add start button with command to execute start_trading function when clicked   							    	    	    	    	    	    	        
stop_button = tkinter.Button(text='Stop', width=25, command=stop_trading)    # add stop button with command to execute stop_trading function when clicked     

# Add label to display results of trading logic implemented in functions  
result_label = tkinter.Label(text="Result: ").grid(row=3)

instrumentsEntry=tkinter.Entry(root)
instrumentsEntry.grid(row=0, column=1)
orderTypesEntry=tkinter.Entry(root)
orderTypesEntry.grid(row=1, column=1)
riskManagementRulesEntry=tkinter.Entry(root)
riskManagementRulesEntry.grid(row=2, column=1)
moneyManagementRulesEntry=tkinter.Entry(root)
moneyManagementRulesEntry.grid(row=3, column=1)
sentimentAnalysisEntry=tkinter.Entry(root)
sentimentAnalysisEntry.grid(row=4, column=1)
economicCalendarDataEntry=tkinter.Entry(root)
economicCalendarDataEntry.grid(row=5, column=1)
newsEventsDataEntry=tkinter.Entry(root)
newsEventsDataEntry.grid(row=6, column=1)

def createTable(root):
    """Creates a table where users can edit the sources for the neural network with categories depending on whether is it for chart data or NLP data."""


    tableFrame=tkinter.Frame(root)
    tableFrame.grid(row=7, columnspan=2)


    tkinter.Label(tableFrame, text="Source").grid(row=0, column=0)
    tkinter.Label(tableFrame, text="Chart Data").grid(row=0, column=1)
    tkinter.Label(tableFrame, text="NLP Data").grid(row=0, column=2)


    sourceEntry1=tkinter.Entry(tableFrame)
    sourceEntry1.grid(row=1, column=0)

    chartDataEntry1=tkinter.Entry(tableFrame)
    chartDataEntry1.grid(row=1, column=1)

    nlpDataEntry1=tkinter.Entry(tableFrame)
    nlpDataEntry1.grid(row=1, column=2)

    sourceEntry2=tkinter.Entry(tableFrame)
    sourceEntry2.grid(row=2, column=0)

    chartDataEntry2=tkinter.Entry(tableFrame)
    chartDataEntry2.grid(row=2, column=1)

    nlpDataEntry2=tkinter.Entry(tableFrame)
    nlpDataEntry2.grid(row=2, column=2)

    sourceEntry3=tkinter.Entry(tableFrame)
    sourceEntry3.grid(row=3, column=0)

    chartDataEntry3=tkinter.Entry(tableFrame)
    chartDataEntry3.grid(row=3, column=1)

    nlpDataEntry3=tkinter.Entry(tableFrame)
    nlpDataEntry3.grid(row=3, column=2)
    return tableFrame

#Function to create dataframe
def create_dataframe(api, instrument):
    r = instruments.InstrumentsCandles(instrument=instrument, params={"granularity": "M1"})
    api.request(r)
    df = pd.DataFrame(r.response['candles'])
    df = df.drop(columns='volume')
    df['time'] = pd.to_datetime(df['time'])  # Converting the time from string to datetime object 
    return df

    
#Function to prepare training data for model    
def prepare_training_data(df): 
    train_cols = ['time', 'openBid', 'highBid', 'lowBid', 'closeBid']  # Training columns which will be used to train the model 
    df_train, df_test = chronological.split_into_train_test(df, train_cols)   # Split the dataset into training and test set 

	#Normalize training data using MinMaxScaler					  
    scaler = MinMaxScaler()                                                   # Instantiate an object of the minmaxscaler class  

    scaled_train = scaler.fit_transform(df[train_cols])                        # Fit and transform the training data using minmaxscaler object    

	#Create input sequence and labels for model     
    X, y = chronological.create_inputs_andlabels(scaled_train)                  # Create input sequences and labels from normalized training data    

	#Reshape input sequence for model     
    X = X.reshape(-1, 1 ,4)                                                    # Reshape the inputs in order to fit into LSTM model       

	#Return prepared inputs and labels      
    return X, y  

    
#Function to build LSTM model    
def buildLSTMModel():     

	#Instantiate a Sequential Model Object        
    model = Sequential()         

	#Add layers to model            
    model.add(LSTM(50, activation='relu', return_sequences=True))               # Add a layer of 50 neurons with relu activation function         

    model.add(Dropout (0.2))                                                    # Add a dropout layer with 20% neuron dropping rate          

    model.add (LSTM (50, activation='relu'))                                    # Add another layer of 50 neurons with relu activation function       

    model.add (Dense (25))                                                     # Add a Dense Layer with 25 Neurons            

    model.add (Dense (1))                                                      # Output Layer             

     ## Compile Model           

    model.compile(optimizer='adam', loss='mean_squared_error')                  # Compile the model with Adam Optimizer and mean squared error loss function

	#Return Model Object         
    return model  


#Function to run model and predict    
def runModel(X, y, model): 

	#Train Model       
    model.fit(X, y, batch_size=1, epochs=1)                                    # Fit the training data into the LSTM Model for 1 epoch  

	#Predict Closing Price of Instrument     
    prediction = model.predict(X)                                              # Make predictions from the trained model           

	#Return Predictions      
    return prediction 


if __name__ == "__main__":
    #Create API Object        
    api = API (access_token=" c546ba5cbc5ff51a3ddff21642565098-3068bb7c8922261d0e820442860f7dea")           # Replace YOURTOKEN with your Oanda V20 access token         

	#Create Dataframe Object       
    df = create_dataframe (api, "EUR_USD")         # Create a dataframe using the instruments candles endpoint of oandapyV20       

	#Prepare Training Data          
    X, y = prepare_training_data (df)              # Prepare the training data for input into the LSTM Model            

	#Build LSTM Model        
    model = buildLSTMModel ()                      # Build an LSTM Model with two layers and 25 neurons in dense layer         

	#Run Model and Predict      
    predictions = runModel (X, y, model)            # Run the LSTM Model on prepared training data and make predictions              

	#Display Predictions        
    print (predictions)                            # Display the predictions made by the model
    root.mainloop()
