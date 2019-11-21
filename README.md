# Long Short Term Memory (LSTM) 

Recurrent Neural Networks are networks with loops in them, allowing information to persist. This means that a recurrent neural network processes sequences one element at a time while retaining a memory of what has come previously in the sequence. This kind of networks are a class of neural networks that is powerful for modeling sequence data - e.g. time series. An example of a recurrent neural network is Long Short Term Memory (LSTM) networks. 

LSTM networks are a type of Recurrent Neural Network, capable of learning long-term dependencies. LSTM networks help preserve the error, that can be back-propagated through time and layers. They allow recurrent networks to continue to lean over many time steps by maintaining a more constant error. The figure below displays the structure of a typical LSTM cell with an input gate, an internal state, an forget gate and an output gate. The data flow is from left-to-right. 

![](https://i2.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2017/09/LSTM-diagram.png?w=669&ssl=1)

LSTM networks work well on a variety of problems and are widely used for e.g. language modeling and translation, analysis of audio and handwriting recognition and generation, which will be the focus in the notebook. LSTM networks can learn the sequences of a problem and then generate entirely new plausible sequences for the problem domain. This type of generative models are useful to study the problem domain itself.

## Case: Text generation with LSTM – “Dracula” by Bram Stoker
In the following, we will create a generative model for text, character-by-character using LSTM recurrent neural networks with Keras . The model is developed to generate plausible text sequences for a given problem. In this case, we will use the book ["Dracula" by Bram Stoker](https://www.gutenberg.org/ebooks/345) as the dataset. We are going to develop a LSTM model to learn the dependencies between characters and the conditional probabilities of characters in sequences. And then we can generate new and original sequences of characters.

First we start by importing the classes and functions we will use to train out model.

![](https://i.imgur.com/8qucU2m.png)
