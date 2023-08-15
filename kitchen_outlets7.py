import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
import glob
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "experiment_tracking_1" 
run_name="SFM"


def read_label(): #read house 1 and 2 labels
    label = {}
    for i in range(1, 3):
        hi = 'low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label
labels = read_label()
for i in range(1,3):
    print('House {}: '.format(i), labels[i], '\n')

def read_merge_data(house): #load watt data in pandas data frame in format of labels as columns and timestamps as rows    

    path = 'low_freq/house_{}/'.format(house)

    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][1]], 
                                    dtype = {'unix_time': 'int64', labels[house][1]:'float64'}) 
    
    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][i]], 
                                    dtype = {'unix_time': 'int64', labels[house][i]:'float64'})
        df = pd.merge(df, data, how = 'inner', on = 'unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time','timestamp'], axis=1, inplace=True)
    return df
df = {}
for i in range(1,3):
    df[i] = read_merge_data(i)

for i in range(1,3):
    print('House {} data has shape: '.format(i), df[i].shape)
    display(df[i].tail(3))

#show the days of house 1 and house 2
dates = {}
for i in range(1,3):
    dates[i] = [str(time)[:10] for time in df[i].index.values]
    dates[i] = sorted(list(set(dates[i])))
    print('House {0} data contain {1} days from {2} to {3}.'.format(i,len(dates[i]),dates[i][0], dates[i][-1]))
    print(dates[i], '\n')

# Plot 2 first day data of house 1 and 2
def plot_df(df, title):
    apps = df.columns.values
    num_apps = len(apps) 
    fig, axes = plt.subplots((num_apps+1)//2,2, figsize=(24, num_apps*2) )
    for i, key in enumerate(apps):
        axes.flat[i].plot(df[key], alpha = 0.6)
        axes.flat[i].set_title(key, fontsize = '15')
    plt.suptitle(title, fontsize = '30')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig('figures/{}_{}.png'.format(title.replace(" ", "_"), i))


for i in range(1,3):
    plot_df(df[i][:dates[i][1]], 'First 2 day data of house {}'.format(i))

# Plot total energy consumption of each appliance from two houses
fig, axes = plt.subplots(1,2,figsize=(24, 10))
plt.suptitle('Total energy consumption of each appliance', fontsize = 30)
cons1 = df[1][df[1].columns.values[2:]].sum().sort_values(ascending=False)
app1 = cons1.index
y_pos1 = np.arange(len(app1))
axes[0].bar(y_pos1, cons1.values,  alpha=0.6) 
plt.sca(axes[0])
plt.xticks(y_pos1, app1, rotation = 45)
plt.title('House 1')

cons2 = df[2][df[2].columns.values[2:]].sum().sort_values(ascending=False)
app2 = cons2.index
y_pos2 = np.arange(len(app2))
axes[1].bar(y_pos2, cons2.values, alpha=0.6)
plt.sca(axes[1])
plt.xticks(y_pos2, app2, rotation = 45)
plt.title('House 2')

# Separate house 1 data into train, validation and test data
df1_train = df[1][:dates[1][10]]
df1_val = df[1][dates[1][11]:dates[1][16]]
df1_test = df[1][dates[1][17]:]
print('df_train.shape: ', df1_train.shape)
print('df_val.shape: ', df1_val.shape)
print('df_test.shape: ', df1_test.shape)

# data for oven_3

# Using mains_1, mains_2 to predict oven_3
X_train1 = df1_train[['mains_1','mains_2']].values 
y_train1 = df1_train['oven_3'].values
X_val1 = df1_val[['mains_1','mains_2']].values
y_val1 = df1_val['oven_3'].values
X_test1 = df1_test[['mains_1','mains_2']].values
y_test1 = df1_test['oven_3'].values
print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)

#data for refrigerator_5

# Using mains_1, mains_2 to predict refrigerator_5
X_train2 = df1_train[['mains_1','mains_2']].values 
y_train2 = df1_train['refrigerator_5'].values
X_val2 = df1_val[['mains_1','mains_2']].values
y_val2 = df1_val['refrigerator_5'].values
X_test2 = df1_test[['mains_1','mains_2']].values
y_test2 = df1_test['refrigerator_5'].values
print(X_train2.shape, y_train2.shape, X_val2.shape, y_val2.shape, X_test2.shape, y_test2.shape)

#data for kitchen_outlets_7

# Using mains_1, mains_2 to predict kitchen_outlets_7
X_train3 = df1_train[['mains_1','mains_2']].values 
y_train3 = df1_train['kitchen_outlets_7'].values
X_val3 = df1_val[['mains_1','mains_2']].values
y_val3 = df1_val['kitchen_outlets_7'].values
X_test3 = df1_test[['mains_1','mains_2']].values
y_test3 = df1_test['kitchen_outlets_7'].values
print(X_train3.shape, y_train3.shape, X_val3.shape, y_val3.shape, X_test3.shape, y_test3.shape)


# data for dishwaser_6

# Using mains_1, mains_2 to predict kitchen_outlets_7
X_train4 = df1_train[['mains_1','mains_2']].values 
y_train4 = df1_train['dishwaser_6'].values
X_val4 = df1_val[['mains_1','mains_2']].values
y_val4 = df1_val['dishwaser_6'].values
X_test4 = df1_test[['mains_1','mains_2']].values
y_test4 = df1_test['dishwaser_6'].values
print(X_train4.shape, y_train4.shape, X_val4.shape, y_val4.shape, X_test4.shape, y_test4.shape)

from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y)) 
def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y)) 


#rnndisagreggator
def build_fc_model():
    fc_model = Sequential()
    fc_model.add(Conv1D(16, 4, activation="linear", input_shape=(2,1), padding="same", strides=1))
    fc_model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
    fc_model.add(Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))

# Fully Connected Layers
    fc_model.add(Dense(128, activation='tanh'))
    fc_model.add(Dense(1, activation='linear'))
    fc_model.add( Dropout(0.2) )

    fc_model.summary()

    return fc_model


# Plot real and predict appliance's consumption on six days of test data
def plot_each_app(df, dates, predict, y_test, title, look_back = 0):
    num_date = len(dates)
    fig, axes = plt.subplots(num_date,1,figsize=(24, num_date*5) )
    plt.suptitle(title, fontsize = '25')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    for i in range(num_date):
        if i == 0: l = 0
        ind = df[dates[i]].index[look_back:]
        axes.flat[i].plot(ind, y_test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'True value')
        axes.flat[i].plot(ind, predict[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Predicted value')
        axes.flat[i].legend()
        l = len(ind)
        plt.savefig('figures/{}_{}.png'.format(title.replace(" ", "_"), i))


def plot_losses(train_loss, val_loss):
    plt.rcParams["figure.figsize"] = [24,10]
    plt.title('Mean squared error of train and val set on house 1')
    plt.plot( range(len(train_loss)), train_loss, color = 'b', alpha = 0.6, label='train_loss' )
    plt.plot( range(len( val_loss )), val_loss, color = 'r', alpha = 0.6, label='val_loss' )
    plt.xlabel( 'epoch' )
    plt.ylabel( 'loss' )
    plt.legend()


# KITCHEN OUTLETS 7 TRAINING

model_3 = build_fc_model()
# Start an MLflow run
with mlflow.start_run():
    lr = 1e-5
    mlflow.log_param("Learning Rate", lr)
    adam = Adam(lr)
    epochs = 200
    mlflow.log_param("epochs", epochs)

    model_3.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', f1_m, precision_m, recall_m])
    start = time.time()

    checkpointer = ModelCheckpoint(filepath="models/kitchen_outlets_7_h1_2.hdf5", verbose=0, save_best_only=True)

    hist_3 = model_3.fit(X_train3, y_train3, batch_size=512, verbose=1, epochs=epochs, validation_split=0.33, callbacks=[checkpointer])

    loss, accuracy, f1_score, precision, recall = model_3.evaluate(X_test3, y_test3, verbose=0)
    print('Finish training. Time: ', time.time() - start)

    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    model_3 = load_model('models/kitchen_outlets_7_h1_2.hdf5', custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})
    pred_3 = model_3.predict(X_test3).reshape(-1)

    mse_loss_3 = mse_loss(pred_3, y_test3)
    mae_loss_3 = mae_loss(pred_3, y_test3)

    mlflow.log_metric("mse_loss_3", mse_loss_3)
    mlflow.log_metric("mae_loss_3", mae_loss_3)

    mlflow.log_artifact(local_path="models/kitchen_outlets_7_h1_2.hdf5", artifact_path="models_pickle")
    mlflow.sklearn.log_model(model_3, "kitchen_outlets_7_model")



train_loss = hist_3.history['loss']
val_loss = hist_3.history['val_loss']
def plot_losses(train_loss, val_loss):
    plt.rcParams["figure.figsize"] = [24,10]
    plt.title('Mean squared error of train and val set on house 1')
    plt.plot( range(len(train_loss)), train_loss, color = 'g', alpha = 0.6, label='train_loss' )
    plt.plot( range(len( val_loss )), val_loss, color = 'r', alpha = 0.6, label='val_loss' )
    plt.xlabel( 'epoch' )
    plt.ylabel( 'loss' )
    plt.legend()

plot_losses(train_loss, val_loss)


plot_each_app(df1_test, dates[1][17:], pred_3, y_test3, 
            'FC model: real and predict kitchen outlets on 6 test day of house 1', look_back = 50)








