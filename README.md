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