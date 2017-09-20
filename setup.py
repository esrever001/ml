import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="machine learning playground",
    version="0.0.4",
    author="Esrecer",
    author_email="esreverfudan@gmail.com",
    description=("A toolkit to play with different machine learning algos."),
    url="https://github.com/esrever001/ml",
    packages=find_packages(),
    long_description=read('README.md'),
        entry_points={
        'console_scripts': [
            'dummy =  ml_playground.pipelines.dummy:dummy_pipeline',
            'mnist =  ml_playground.pipelines.mnist:mnist_pipeline',
            'iris =  ml_playground.pipelines.iris:iris_pipeline',
        ]
    },
)