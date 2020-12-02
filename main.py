from LVQ import LVQ


if __name__ == '__main__':
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
  y = [0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
  lr = 0.001
  epoch = 10000

  lvq = LVQ()
  lvq.load_data(x, y)
  lvq.init_weight()
  lvq.train(lr, epoch, verbose=True)