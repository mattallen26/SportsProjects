import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
%matplotlib inline

#import dataset
pitch_data_import = pd.read_csv(r"/Users/mac/Desktop/pitchdata.csv")

#remove 111 null values in the PlateHeight and PlateSide columns 
pitch_data_import_no_null = pitch_data_import.dropna()
pitch_data_import_no_null = pitch_data_import_no_null.reset_index(drop=True)

#create two new columns that will assign binary values to the BatSide and PitcherHand columns (left is 0 right is 1)
pitch_data_import_no_null['BatSideBinary'] = [0 if x == 'L' else 1 for x in pitch_data_import_no_null['BatSide']]
pitch_data_import_no_null['PitcherHandBinary'] = [0 if x == 'L' else 1 for x in pitch_data_import_no_null['PitcherHand']]

#split the remaining dataset into a dataset for training (70% of pitches called train_set) and a dataset for testing (30% of pitches called test_set)
train_set, test_set = train_test_split(pitch_data_import_no_null, test_size = 0.3, random_state = 100)
train_set["Dataset"] = "train"
test_set["Dataset"] = "test"

#build keras sequential ml model with a binary crossentropy loss function and an adam optimizer
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model on the train dataset using the 6 input columns
model.fit(train_set[["Balls","Strikes","PitcherHandBinary","BatSideBinary","PlateHeight","PlateSide"]].values, train_set[["CalledStrike"]].values, epochs=15, batch_size=100)

#test the model on the test dataset using the same 6 inputs
_, accuracy = model.evaluate(test_set[["Balls","Strikes","PitcherHandBinary","BatSideBinary","PlateHeight","PlateSide"]].values, test_set[["CalledStrike"]].values)
print('Accuracy: %.2f' % (accuracy*100))

#use the model to predict the expected called strike probability for every pitch in both the train and test datasets and create a new column to store that value
train_set['ExpectedCalledStrikeProbability'] = model.predict(train_set[["Balls","Strikes","PitcherHandBinary","BatSideBinary","PlateHeight","PlateSide"]].values)
test_set['ExpectedCalledStrikeProbability'] = model.predict(test_set[["Balls","Strikes","PitcherHandBinary","BatSideBinary","PlateHeight","PlateSide"]].values)

#aggregate the train and test datasets back into one dataset
full_frame_w_exp_called_strike_probability = train_set.append(test_set)

#select the same columns from the original dataset with the addition of the column that labels the dataset each pitch was part of
pitch_data_w_exp_called_strike_probability = full_frame_w_exp_called_strike_probability[["GameID",'PitchNumber','Balls','Strikes','PitcherHand','BatSide','PlateHeight','PlateSide','ExpectedCalledStrikeProbability',"CalledStrike","Dataset"]]
pitch_data_w_exp_called_strike_probability = pitch_data_w_exp_called_strike_probability.sort_values(by=['GameID', 'PitchNumber'])
pitch_data_w_exp_called_strike_probability = pitch_data_w_exp_called_strike_probability.reset_index(drop = True)

#export csv with expected called strike probabilities for every pitch
pitch_data_w_exp_called_strike_probability.to_csv("/Users/mac/Desktop/Matthew_Allen_PitchData_with_Expected_Called_Strike_Probability_Dataframe.csv",index = False)

#plot the expected called strike percentage of each pitch as a function of pitch location
plt.figure(figsize = (10, 10))
plt.title("Expected Called Strike Percentage as a Function of Pitch Location")

mn = 0 
mx = 1
md1 = 0.25
md2 = 0.5
md3 = 0.75


plt.scatter(pitch_data_w_exp_called_strike_probability['PlateSide'].values.flatten(), pitch_data_w_exp_called_strike_probability['PlateHeight'].values.flatten(), c = pitch_data_w_exp_called_strike_probability['ExpectedCalledStrikeProbability'].values.flatten(), cmap = 'coolwarm', s=0.01, vmin=mn, vmax=mx)
plt.xlabel("Pitch Plate Side Location (ft)")
plt.ylabel("Pitch Plate Height Location (ft)")
plt.xlim(-5, 5)
plt.ylim(-3, 7)
plt.gca().set_aspect('equal', adjustable='box')
clb = plt.colorbar(orientation="horizontal")
clb.ax.set_xlabel('Expected Called Strike Percentage (%)')
clb.ax.xaxis.set_label_position('top')
clb.set_ticks([mn,md1,md2,md3,mx])
clb.set_ticklabels(["0%","25%","50%","75%","100%"])
plt.savefig("/Users/mac/Desktop/Matthew_Allen_PitchData_with_Expected_Called_Strike_Probability_Chart.png")
