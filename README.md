# Instructions
1. Clone or download this repository
2. 
# Settings
There are 4 variables that can be modified by the user through the use of the following command line arguments.
* `--Fs` - Feature selection size. This is the number of features to be left after feature selection has occured.
* `--Vs` - Vocabulary size. This is the size of the vocabulary to be used in the bag-of-words.
* `--Splits` - Training, Validation, Test Splits. This is the percentage size of each of these sets.
* `--Seed` - Randomness seed. This seed is used to provide reproducable splits between the sets of data

# Example Usage
* python solver.py --Fs 500 --Vs 1500 --Splits 70 15 15 --Seed 127863182
* python solver.py --Fs 200 --Splits 50 35 15

## Default Values
* --Fs 600
* --Vs 1000
* --Splits 80 10 10
* --Seed 1337


# Dependencies
* nltk==3.6.1              `pip install nltk`
* numpy==1.20.2            `pip install numpy`
* scikit_learn==0.24.1     `pip install sklearn`
