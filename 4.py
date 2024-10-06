import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer, MinMaxScaler

data = arff.loadarff('C:/Users/ROXANA CASTILLO/Desktop/354/gym_members_exercise_tracking.arff')
df = pd.DataFrame(data[0])

df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

onehotencoder = OneHotEncoder()
onehot_encoded = onehotencoder.fit_transform(df[['Workout_Type']]).toarray()
df_onehot = pd.DataFrame(onehot_encoded, columns=onehotencoder.get_feature_names_out(['Workout_Type']))
df = df.join(df_onehot)
df.drop(columns=['Workout_Type'], inplace=True)

labelencoder = LabelEncoder()
df['Experience_Level'] = labelencoder.fit_transform(df['Experience_Level'])

discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['Age_Discretized'] = discretizer.fit_transform(df[['Age']])

scaler = MinMaxScaler()
df[['Weight (kg)', 'Height (m)', 'BMI']] = scaler.fit_transform(df[['Weight (kg)', 'Height (m)', 'BMI']])

df.head()
