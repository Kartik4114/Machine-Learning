import loadData
from loadData import data

# Size of the dataset
data_shape = data.shape
print(f"Size of the dataset: {data_shape}")

# Names of the features
features = data.columns.tolist()
print(f"Names of Features: {features}")

# Finding missing data
missing_data = data.isnull().sum()
print("Missing Entities in Each Feature:")
print(missing_data)
