from LVQ import LVQ
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == '__main__':
  file_name = 'heart.csv'
  lr = 0.001
  epoch = 125
  seed = 42
  train_test_ratio = 0.3

  df = pd.read_csv(file_name)

  a = pd.get_dummies(df['cp'], prefix = "cp")
  b = pd.get_dummies(df['thal'], prefix = "thal")
  c = pd.get_dummies(df['slope'], prefix = "slope")

  frames = [df, a, b, c]
  df = pd.concat(frames, axis = 1)
  df = df.drop(columns = ['cp', 'thal', 'slope'])

  y = df.target.values
  x_data = df.drop(['target'], axis = 1)

  x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = train_test_ratio,random_state=0)
  x_train = np.array(x_train.T)
  y_train = y_train.T
  x_test = np.array(x_test.T)
  y_test = y_test.T

  lvq = LVQ()
  lvq.load_data(x_train, y_train)
  lvq.init_weight(2, seed)
  metrics = lvq.train(lr, epoch, verbose=True)
  lvq.save_weight('weight.bin')
  lvq.test(x_test, y_test)

  plt.plot(metrics['accuracy'])
  plt.plot(metrics['recall'])
  plt.plot(metrics['precision'])
  plt.plot(metrics['fpr'])
  plt.plot(metrics['f1'])

  plt.title('Metrics')
  plt.xlabel('epoch')
  plt.legend(['Accuracy', 'Recall', 'Precision', 'FPR', 'F1'])

  plt.show()
