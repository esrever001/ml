#!/usr/bin/env python
from __future__ import absolute_import
from ml_playground.data.iris import Iris


def main():
    iris = Iris()
    iris.Init()
    iris.PrintAll()


if __name__ == "__main__":
    main()