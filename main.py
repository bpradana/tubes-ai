import math
import random


class LVQ:
  '''
  Inisialisasi dengan data (x, y), learning rate, dan epoch
  X merupakan data dan Y merupakan target
  '''
  def __init__(self, x, y, lr, epoch):
    self.x = x
    self.y = y
    self.w = [None, x[0].copy(), x[1].copy()]
    self.lr = lr
    self.epoch = epoch

  '''
  Fungsi yang digunakan untuk menghitung jarak antara x1 dan x2
  '''
  def dist(self, x1, x2):
    temp = 0
    for x1_, x2_ in zip(x1, x2):
      temp += (x1_ - x2_) ** 2
    return math.sqrt(temp)

  '''
  Fungsi yang digunakan untuk membandingkan antara input dengan weight 1 dan weight 2
  Jika jarak input dengan weight 1 lebih kecil dari jarak input dengan weight 2 maka me-return 1
  Jika jarak input dengan weight 2 lebih kecil dari jarak input dengan weight 1 maka me-return 2
  Jika jarak input dengan weight 1 sama dengan jarak input dengan weight 2 maka me-return random antara 1 dan 2
  Jika nilai nilai 
  '''
  def compare(self, x):
    class1 = self.dist(x, self.w[1])
    class2 = self.dist(x, self.w[2])
    if class1 < class2:
      return 1
    elif class2 < class1:
      return 2
    elif class2 == class1:
      return random.randint(1,2)

  '''
  Fungsi yang digunakan untuk melakukan iterasi pada data sesuai dengan algoritma LVQ
  '''
  def iterate(self):
    for x_, y_ in zip(self.x, self.y):
      cluster = self.compare(x_)
      if cluster == y_:
        for i in range(len(self.w[cluster])):
          self.w[cluster][i] = self.w[cluster][i] + self.lr*(x_[i] - self.w[cluster][i])
      elif cluster != y_:
        for i in range(len(self.w[cluster])):
          self.w[cluster][i] = self.w[cluster][i] - self.lr*(x_[i] - self.w[cluster][i])
      print(x_, y_, cluster)

  '''
  Fungsi untuk menjalankan algoritma LVQ sebanyak n epoch
  '''
  def run(self):
    for i in range(self.epoch):
      print('Epoch %d/%d' % (i+1, self.epoch))
      self.iterate()
      '''
      Mencetak weight 1 dan weight 2
      '''
      print('weight 1 :', self.w[1])
      print('weight 2 :', self.w[2])


if __name__ == '__main__':
  '''
  Inisialisasi data
  '''
  x = [
    [1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 1]
  ]
  y = [1, 2, 1, 1, 1, 1, 2, 2, 2, 2]

  '''
  Inisialisasi objek LVQ
  '''
  lvq = LVQ(x, y, 0.05, 3)

  '''
  Menjalankan algoritma LVQ
  '''
  lvq.run()