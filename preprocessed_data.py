import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("Hello World")

# Read the dataset using Pandas
space_data = pd.read_csv('space_decay.csv',header='infer')

# Drop entirely empty columns
space_data = space_data.drop(columns=['DECAY_DATE'])

# Fill missing categorical values with a placeholder
space_data['OBJECT_ID'].fillna('UNKNOWN', inplace=True)
space_data['RCS_SIZE'].fillna('UNKNOWN', inplace=True)
space_data['COUNTRY_CODE'].fillna('UNKNOWN', inplace=True)

# Convert date columns to datetime format
space_data['CREATION_DATE'] = pd.to_datetime(space_data['CREATION_DATE'])
space_data['EPOCH'] = pd.to_datetime(space_data['EPOCH'])

# Read the dataset using Pandas
print(space_data.info)

# Normalize selected numerical columns
scaler = MinMaxScaler()
numerical_columns = ['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 
                     'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
                     'BSTAR', 'MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT',
                     'SEMIMAJOR_AXIS', 'PERIOD', 'APOAPSIS', 'PERIAPSIS']
space_data[numerical_columns] = scaler.fit_transform(space_data[numerical_columns])

# Encode categorical columns
space_data = pd.get_dummies(space_data, columns=['OBJECT_TYPE', 'RCS_SIZE', 'COUNTRY_CODE'])

# Save preprocessed data to a new CSV
space_data.to_csv('preprocessed_space_data.csv', index=False)