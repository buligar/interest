import os
import pandas as pd
from glob import glob
from sklearn.model_selection import  train_test_split

metadata = "HAM10000_metadata.csv"
skin_df = pd.read_csv(metadata)
dataset="C:\\Users\\bulig\\PycharmProjects\\pythonProject\\input\\HAM10000\\HAM10000_img"
# image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(dataset, '*', '*.jpg'))}
dataset = (x for x in glob(os.path.join(dataset)))
print(dataset)
# skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
# print(image_path)
# X = np.asarray(skin_df_balanced['path'].tolist())
x_train, x_test = train_test_split(dataset, test_size=0.25, random_state=42)
print(x_train)