import random
import pickle

class LVQ:
  def __init__(self):
    self.X = []
    self.y = []
    self.w = []
    self.lr = 0
    self.epoch = 0

  def load_data(self, X_test, y_test):
    self.X = X_test
    self.y = y_test

  def init_weight(self, class_num, seed=random.randint(0,100), file_name='', random_weight=True):
    if random_weight:
      random.seed(seed)
      self.w = [[random.random() for _ in self.X[0]] for _ in range(class_num)]
    else:
      file = open(file_name, 'rb')
      self.w = pickle.load(file)
      file.close()

  def save_weight(self, file_name):
    file = open(file_name, 'wb')
    pickle.dump(self.w, file)
    file.close()

  def distance(self, a, b):
    total = 0
    for a_, b_ in zip(a, b):
      total += (a_ - b_) ** 2
    dist = total ** 0.5
    return dist

  def compare(self, x):
    dist = []
    for w_ in self.w:
      dist.append(self.distance(x, w_))
    return dist.index(min(dist))

  def iterate(self):
    confusion = [0, 0, 0, 0]

    for x_, y_ in zip(self.X, self.y):
      cluster = self.compare(x_)
      if cluster == y_:
        if cluster == 0:
          confusion[0] += 1
        if cluster == 1:
          confusion[3] += 1
        for i in range(len(self.w[cluster])):
          self.w[cluster][i] = self.w[cluster][i] + self.lr*(x_[i] - self.w[cluster][i])
          
      elif cluster != y_:
        if cluster == 0:
          confusion[2] += 1
        if cluster == 1:
          confusion[1] += 1
        for i in range(len(self.w[cluster])):
          self.w[cluster][i] = self.w[cluster][i] - self.lr*(x_[i] - self.w[cluster][i])
    return confusion

  def score(self, conf):
    accuracy = (conf[0]+conf[3]) / (conf[0]+conf[1]+conf[2]+conf[3])
    recall = conf[3] / (conf[3]+conf[2])
    precision = conf[3] / (conf[3]+conf[1])
    fpr = conf[2] / (conf[0]+conf[2])
    f1 = 2*(precision*recall)/(precision+recall)

    return accuracy, recall, precision, fpr, f1

  def train(self, lr, epoch, verbose=False):
    self.lr = lr
    self.epoch = epoch
    metrics = {
      'accuracy': [],
      'recall': [],
      'precision': [],
      'fpr': [],
      'f1': []
    }
    for i in range(self.epoch):
      print('Epoch %d/%d' % (i+1, self.epoch))
      conf = self.iterate()
      
      accuracy, recall, precision, fpr, f1 = self.score(conf)
      metrics['accuracy'].append(accuracy)
      metrics['recall'].append(recall)
      metrics['precision'].append(precision)
      metrics['fpr'].append(fpr)
      metrics['f1'].append(f1)

      if verbose:
        print('Accuracy:', accuracy)
        print('Recall:', recall)
        print('Precision:', precision)
        print('FPR:', fpr)
        print('F1:', f1)
      print('')
    return metrics
  
  def test(self, X_test, y_test):
    confusion = [0, 0, 0, 0]
    print('TESTING')
    for x_, y_ in zip(X_test, y_test):
      cluster = self.compare(x_)
      if cluster == y_:
        if cluster == 0:
          confusion[0] += 1
        if cluster == 1:
          confusion[3] += 1
      elif cluster != y_:
        if cluster == 0:
          confusion[2] += 1
        if cluster == 1:
          confusion[1] += 1    

    accuracy, recall, precision, fpr, f1 = self.score(confusion)
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('FPR:', fpr)
    print('F1:', f1)
  
  def predict(self, x):
    prediction = []
    for x_ in x:
      cluster = self.compare(x_)
      prediction.append(cluster)
    return prediction
