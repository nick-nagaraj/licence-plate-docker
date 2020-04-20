from numpy.random import RandomState
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('/home/martianspeaks/Study/Research/FINAL_Test.csv')

train, test = train_test_split(df, test_size=0.5)
train.to_csv('/home/martianspeaks/Study/Research/FINAL_Test.csv', index=False)
test.to_csv('/home/martianspeaks/Study/Research/FINAL_Evaluation.csv', index=False)
