# MNIST Number Classification
## Implementation
### Virtual environment
All the code has been written using Python 3.6.9. You can install the packages from the numbers_requirements.txt file.
### Main code
Run the code with the default settings.
```
python main.py
```

The defaults settings are:
* the dataset is normalized between 0 and 1
* the dataset is shuffle before each epoch
* the original train dataset is randomly split in 80% for the train set and 20% for the validation set
* each training batch contains 32 examples.
* there is 20 epochs
* the models and images will be saved in “data”

You can change each default parameter.

For more information, run
```
python main.py --help 
```

