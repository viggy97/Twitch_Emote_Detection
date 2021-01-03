# Twitch_Emote_Detection
Twitch chat emotion prediction through Emote based Sentiment Analysis  

# Using Classifiers

The code for querying our three classifiers are present in AverageBasedClassifier.py, DistributionBasedClassifier.py, and CNNBasedClassifier.py, respectively. After installing the required packages given in requirements.txt you can run python3 AverageBasedClassifier.py on the test set. The same applies to the other classifiers as well. However, to use the LSTM_Classifier, you need to first generate the embedding and feed it to the LSTM network. Each of the three classifiers has a classify_df method that classifies a Pandas DataFrame containing a message column that includes the texts to be classified. This can be used to easily estimate the sentiment of a batch of chat messages.


# Lexica 
 In our work, we used three sentiment lexica:
 Emoji Sentiment Lexicon by Kralj Novak et al.
 VADER Sentiment Lexicon by Hutto and Gilbert
 A Self-Labeled Emote Lexicon


# Labelled Data and word embeddings

For evaluating the performance of the prediction models, 2000 sampled Twitch comments which were anotated by humans from very active Twitch channels were used. 
The labeled dataset is stored as a csv file.
Gensim word to vec is used to generate Embeddings which was used for the LSTM network.

