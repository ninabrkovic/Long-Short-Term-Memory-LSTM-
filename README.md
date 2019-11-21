# Long Short Term Memory (LSTM) 

Recurrent Neural Networks are networks with loops in them, allowing information to persist. This means that a recurrent neural network processes sequences one element at a time while retaining a memory of what has come previously in the sequence. This kind of networks are a class of neural networks that is powerful for modeling sequence data - e.g. time series. An example of a recurrent neural network is Long Short Term Memory (LSTM) networks. 

LSTM networks are a type of Recurrent Neural Network, capable of learning long-term dependencies. LSTM networks help preserve the error, that can be back-propagated through time and layers. They allow recurrent networks to continue to lean over many time steps by maintaining a more constant error. The figure below displays the structure of a typical LSTM cell with an input gate, an internal state, an forget gate and an output gate. The data flow is from left-to-right. 

![](https://i2.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/09/LSTM-diagram.png?w=669&ssl=1)

LSTM networks work well on a variety of problems and are widely used for e.g. language modeling and translation, analysis of audio and handwriting recognition and generation, which will be the focus in the notebook. LSTM networks can learn the sequences of a problem and then generate entirely new plausible sequences for the problem domain. This type of generative models are useful to study the problem domain itself.

## Case: Text generation with LSTM – “Dracula” by Bram Stoker
In the following, we will create a generative model for text, character-by-character using LSTM recurrent neural networks with Keras. The model is developed to generate plausible text sequences for a given problem. In this case, we will use the book ["Dracula" by Bram Stoker](https://www.gutenberg.org/ebooks/345) as the dataset. We are going to develop a LSTM model to learn the dependencies between characters and the conditional probabilities of characters in sequences. And then we can generate new and original sequences of characters.

First we start by importing the classes and functions we will use to train out model.

![](https://i.imgur.com/8qucU2m.png)
![](https://i.imgur.com/lx3I2kK.png)

Then we load the text file for the book and convert all of the characters to lowercase. We convert all of the characters to reduce the vocabulary that the network must learn. After loading the book, the preprocessing of the data begins. The characters cannot be modelled directly, therefore we need to convert them to integers. This is done by creating a set of all the different characters in the book and then creating a map of each character to a unique integer. Finally we summarize the dataset by printing the total numbers of characters in the book and the total number of different characters in the vocabulary.

### Preprocessing
![](https://i.imgur.com/7AF5xVU.png)

We can see that the Dracula book has 846.028 characters. After converting to lowercase, we get 69 different characters in the vocabulary for the network to learn. Next we split the book text into subsequences with a length of 100 characters. Each training pattern is comprised of 100 time steps of one character (X) followed by one character output (y). Each character in the book is learned from the 100 characters that preceded it - except the first 100 characters. Afterwards we convert the characters to integers.

![](https://i.imgur.com/yfb8cDz.png)

We can see that when we split up the dataset into training data for the network to learn, we have about 84500 training patterns. This makes sense as we excluded the first 100 characters - we then have one training pattern to predict each of the remaining characters.

Next step is to transform the list of input sequences into the form [samples, time steps, features] and rescale the integers to the range 0 to 1, so it is easier to lean by the LSTM network that uses the sigmoid activation function by default. Afterwards we need to convert the output patterns, which are the single characters converted to integers. This is so that we can configure the network to predict the probability of each of the 69 different characters in the vocabulary. Each y-value is converted to a vector with a length of 69 with all zeros except with a 1 in the column for the letter (integer) that the pattern represents.

### Defining the LSTM model 
We define the LSTM model with a single hidden layer with 256 memory units, a dropout with a probability of 20 and an output Dense layer using the softmax activation funktion to output a probability prediction for each of the 69 characters between 0 and 1. The problem is a single character classification problem with 69 classes and is defined as optimizing the log loss (cross entropy) - in this case using the 'adam' optimization algorithm for speed.

![](https://i.imgur.com/U4VtVO8.png)

### Generating text with the LSTM model 
The following section will shortly explain how to use the trained LSTM model to generate text. First the integers from before must be converted to characters, so that the predictions are understandable. Then you start off with a random sequence as input, generate the next character, update the sequence to add the generated character on the end and trim off the first character, which will be repeated for as long as you want to predict new characters. 
If you want to improve your LSTM model, you can for example change the number of characters as output for a given seed or remove all the punctuations or commas in the source text and thereby changing the models’ vocabulary. You can also change the number of training epochs or add additional layers/memory units to the model. 

## References:
* ["Understanding LSTM Networks"](https://colah.github.io/posts/)
* ["LSTM: A Search Space Odyssey"](https://arxiv.org/pdf/1503.04069.pdf)
* ["A Beginner's Guide to LSTMs and Recurrent Neural Networks"](https://skymind.ai/wiki/lstm)
* ["Recurrent neural networks and LSTM tutorial in Python and TensorFlow"](https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/)
* ["Text Generation With LSTM Recurrent Neural Networks in Python with Keras"](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
