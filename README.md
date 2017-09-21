Intro
=============
This is a studying project for implementing basic machine learning.

Setup
=============
```
virtualenv env
source env/bin/activate
make bootstrap
dummy
```

Dataset Setup
=============
MNIST (http://yann.lecun.com)
-------------
```
mdkir tmp/mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P tmp/mnist/
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P tmp/mnist/
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P tmp/mnist/
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P tmp/mnist/
gzip -d tmp/mnist/*.gz
mkdir -p ml_playground/data/dataset/mnist/
cp tmp/mnist/* ml_playground/data/dataset/mnist/
```

iris (https://archive.ics.uci.edu/ml/datasets/iris)
-------------
```
mdkir tmp/iris
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P tmp/iris/
mkdir -p ml_playground/data/dataset/iris/
cp tmp/iris/* ml_playground/data/dataset/iris/
```

CIFAR-10 (http://www.cs.toronto.edu/~kriz/cifar.html)
----------
```
mdkir tmp/cifar-10
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P tmp/cifar-10/
tar  -xvzf tmp/cifar-10/cifar-10-python.tar.gz --directory tmp/cifar-10/
mkdir -p ml_playground/data/dataset/cifar-10/
cp tmp/cifar-10/cifar-10-batches-py/* ml_playground/data/dataset/cifar-10/
```
