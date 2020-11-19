# Digit Recognition

## Table of Contents

* [About the Project](#about-the-project)
* [Manual](#manual)
  * [Prerequisites](#prerequisites)
  * [Getting Started](#getting-started)
  * [Setting up the NN](#setting-up-the-nn)
  * [Starting up the GUI](#starting-up-the-gui)
  * [Using the Interface](#using-the-interface)
  * [Optional: Using the 'digit_recognition' Package in Python](#optional)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About the Project
This repository contains a PyPI package for the recognition of handwritten digits using a neural network (NN). Only Numpy is used for the implementation of the NN itself. The project was created in the context of a programming course at the University of Amsterdam. It therefore rather serves illustrative than practical purposes.

The most interesting code - the implementation of the NN (i.e., initializing, training, running, and evaluating) - can be found in the 'digit_recognition.py' script.

The package also includes a GUI, which allows to draw a number and have it recognized by the NN.

![alt text](https://github.com/Mittenbuhler/digit_recognition/blob/dev/gui_example.png?raw=true)

## Manual
### Prerequisits
The package was built with Python 3.7 and uses the following dependencies: 
- `numpy`
- `urllib3`
- `gzip`
- `tkinter`
- `Pillow`

These should be installed automatically during the installation if necessary. If the package does not run as intended, please ensure that the dependencies listed here have been installed correctly.

### Getting Started
You can download the package from Github or from PyPI

#### Github
First, clone the repository.
```
git clone https://github.com/Mittenbuhler/digit_recognition.git
```
Then, install the package.
```
sudo python setup.py install
```

#### PyPI
Use either pip install
```
pip install digit_recognition
```
or download the package from https://pypi.org/project/digit-recognition/ and install the package using
```
sudo python setup.py install
```

### Setting up the NN
The package does not include a trained NN, but only the necessary functions to build it. Therefore, prior to using the interface, the NN has to be set up (this has to be done only once). To do so, run the command `install_network` in the command prompt/terminal (alternative: import the ‘digit_recognition’ module in python and call the `install_network()` function).

This creates a folder in the current directory (“DR_Data”) and downloads training and test sets from the MNIST database to this folder. It uses this data to train a NN with 784, 200, 100, and 10 nodes in each of four layers, respectively. The training algorithm goes through three epochs each with 60.000 training digits (this may take a few minutes). The NN is evaluated and the accuracy as well as the recall and precision for each digit are printed to the console (**important**: accuracy should be above 95%). Finally, the NN is saved to the “DR_Data” folder.

### Starting up the GUI
To start the interface, run the command `digit_recognition` in the command prompt/terminal (alternative: import the ‘digit_recognition’ module in python and call the `run_gui()` function). Important: Ensure that the “DR_Data” folder is located in the current working directory.

### Using the Interface
The interface consists of two fields: a drawing field framed in black (left) and a feedback field showing several outputs (right). The user can draw in the drawing field by pressing the left mouse button and moving the mouse. Located below the drawing field are two buttons: the “Recognize!” button passes the drawing to the NN and displays its output in the feedback field; The “Reset” button deletes the current drawing and output. 

In the feedback field, three outputs are displayed: first, the digit recognized by the NN in the user’s drawing; second, the confidence of this recognition (i.e. the probability that the recognition is correct); and third, a possible alternative (i.e. the second most likely recognition). If the confidence is above 80%, no alternative is displayed.

**Important**: The performance of the NN is highly sensitive to size and location of the user’s digit in the drawing field. The grey rectangle in the drawing field indicates location and size for optimal performance.


### Optional: Using the 'digit_recognition' Package in Python
Alternatively to calling functions from the command prompt/terminal as described above, they can be called from python directly (import package, e.g., with `import digit_recognition`). This way, all functions from the package can be called individually. This provides greater flexibility in setting up the NN. Most notably, the user can specify the design of the NN (number of layers, number of nodes in each layer, bias nodes on or off), specify the training process (number of epochs, step size), access the confidence matrix when evaluating the NN, and run the forward propagation algorithm individually. 

For more details see https://pypi.org/project/digit-recognition/#description.

## License
Distributed under the MIT License.

## Contact
Maximilian Mittenbühler - max.mittenbuhler@gmail.com

Project link: [https://github.com/Mittenbuhler/digit_recognition](https://github.com/Mittenbuhler/digit_recognition)

## Acknowledgments
The implementation of the NN is based on information and code examples from the following sources:
- [https://www.python-course.eu/neural_networks.php](https://www.python-course.eu/neural_networks.php)
- [http://neuralnetworksanddeeplearning.com/index.html](http://neuralnetworksanddeeplearning.com/index.html)
