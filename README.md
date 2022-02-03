# NPL-Machine-Translation
In this project, we train a neural network model to automatically translate from English words to their  transformed forms.

The rule of this transformation is as follow: 
• if the first letter is a consonant, then that letter is moved to the end of the word and “ay” is
appended, e.g., slow →lowsay
• if a word starts with a vowel, then append “way” at the end, e.g., amoeba →amoebaway.
1
• some consonant pairs like “sh” are moved together to end of the word with “ay” appended,
e.g., shallow →allowshay.

In this model, we succesfully build and train a machine translation system using data.txt provided to learn these rules implicitly. 
The model we use is based on RNN, sequence to sequence model. The encoder RNN 
compresses the input sequence into a fixed-length vector, represented by the final hidden 
state hT . The decoder RNN conditions on this vector to produce the translation, character by 
character. Input characters are passed through an embedding layer before they are fed into 
the encoder RNN; in our model, we learn a 2910 embedding matrix, where each of the 29 
characters in the vocabulary is assigned a 10-dimensional embedding. At each time step, the 
decoder RNN outputs a vector of unnormalized log probabilities given by a linear 
transformation of the decoder hidden state. When these probabilities are normalized, they 
define a distribution over the vocabulary, indicating the most probable characters for that 
time step. The model is trained via a cross-entropy loss between the decoder distribution 
and ground-truth at each time step. 
