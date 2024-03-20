'''TED FROM UCI Dataste '''

from email.quoprimime import header_decode
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np 

def get_UCI_data(save= True): 
    path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data'
    df = pd.read_csv(path, skiprows=2, index_col=False, header=None)
    target= df[9]
    data = df.drop(columns=[9])
    # One Hot encode Target 
    enc= OneHotEncoder(sparse= False)
    target=enc.fit_transform(target.values.reshape(-1, 1))
    if save: 
        pickle.dump(enc, open('../data/TicTacToeEncoder.pkl','wb'))
    enc= OneHotEncoder(sparse= False)
    data=enc.fit_transform(data.values)
    if save: 
        pickle.dump(enc, open('../data/TicTacToeEncoder_data.pkl','wb'))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    # Train, Test, Val Split
    if save: 
        np.save('../data/Tic_X_train.npy',X_train)
        np.save('../data/Tic_X_test.npy',X_test)
        np.save('../data/Tic_X_val.npy',X_val)
        np.save('../data/Tic_y_train.npy',y_train)
        np.save('../data/Tic_y_test.npy',y_test)
        np.save('../data/Tic_y_val.npy',y_val)
    return X_train,X_val, X_test, y_train, y_val, y_test

if __name__=='__main__':
    get_UCI_data()