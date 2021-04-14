''' 
Sentiment Analysis of Movie Reviews using CNN and RNN architectures 
Author - Selina Mead Miller
October 2020
'''

# Import Pre-Processing libraries
import csv
import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Import CNN Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
# IMport RNN Libraries
from tensorflow.python.keras.layers import LSTM, Bidirectional, GRU
from keras import regularizers
from sklearn.model_selection import KFold



class Sentiment_Analysis:
	
	def __init__(self, file_path):
		self.file_path = file_path 
		print('\n==========================================================================\n')
		print('\n **** Sentiment Analysis Task ****\n')
		print('\n=== Retrieving IMBD Dataset ===')
		self.dataset = pd.read_csv(file_path)
		print('No of samples = ', len(self.dataset))
		self.small_dataset = self.dataset[:500]

	'''
	Process data and assign test/training sets
	@params - movie review file
	@returns - X_train, y_train, X_test, y_test
	'''
	def pre_process(self, file):
		dataset = self.dataset
		# print('\n=== Pre Processing Data ===')
		# Check for null values and drop if any
		if dataset.isnull().values.any():
			print('removing null values')
			dataset = dataset.dropna(how='any', axis=0)
		
		# Get Movie Reviews for cleaning
		small = False # Using small dataset until all code is working
		if small:
			movie_reviews = self.small_dataset
			print('Size of Dataset: ', movie_reviews.shape[0])
		else:
			movie_reviews = self.dataset
			print('Size of Dataset: ', movie_reviews.shape[0])

		# print(movie_reviews.review[7])
		print('Pre-processing Data......')
	
		# Clean textual data from 'Review'
		stemmer = SnowballStemmer('english') 
		# Function to clean up requirements
		# Souce: 
		def process_text(sent):
			# Removing html tags
		    sentence = remove_tags(sent)
		    # Remove punctuations and numbers
		    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
		    # Single character removal
		    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
		    # Removing multiple spaces
		    sentence = re.sub(r'\s+', ' ', sentence)
		    return sentence

		TAG_RE = re.compile(r'<[^>]+>')
		def remove_tags(text):
			return TAG_RE.sub('', text)

		# Store cleaned reviews into a list to use for train/test
		X = []
		sentences = list(movie_reviews['review'])
		for sent in sentences:
			X.append(process_text(sent))

		# Convert y to num
		y = []
		sentiments = list(movie_reviews.sentiment)
		for sent in sentiments:
			if sent == 'positive':
				y.append(1)
			else:
				y.append(0)

		# Split into train and test
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# Convert text to numercal data
		'''
		word_counts: A dictionary of words and their counts.
		word_docs: A dict of words and how many documents each appeared in.
		word_index: A dic of words and their uniquely assigned integers.
		document_count:An integer count of the total number of documents that were used to fit the Tokenizer.
		'''
		# Convert text to numerical data. Each unique word has a corresponding integer assigned
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(X_train)
		X_train = tokenizer.texts_to_sequences(X_train)
		X_test = tokenizer.texts_to_sequences(X_test)

		# Number of unique words
		vocab_size = len(tokenizer.word_index) + 1  
		print('Number of Unique Words in Corpus: ', vocab_size)

		MAX_LENGTH = len(max(X_train, key=len))
		print(MAX_LENGTH)
		
		# Padding
		max_len = 100 # Test on 100
		X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
		X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
		
		return X_train, y_train, X_test, y_test, vocab_size


	def CNN(self, X_train, y_train, X_test, y_test, vocab_size):
		print('\n=== Convolutional Neural Network ===\n')

		# Max length of review for embedding
		max_length = 100 
		embedding_dim = 100

		def best_model():
			epochs = [2, 5, 10, 15, 20]
			dropout_rate = [0.1, 0.2, 0.3, 0.4]
			# learning_rates = [0.01, 0.05, 0.1]
			list_of_all_scores = list()
			list_of_scores = list()
			list_of_dropout = list()
			list_of_all_dropouts = list()
			list_of_epochs = list()

			for i in dropout_rate:
				model = Sequential()
				model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
				model.add(Conv1D(filters=max_length, kernel_size=5, padding='same', activation='relu'))
				model.add(GlobalMaxPooling1D())
				model.add(Dropout(i))
				model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
				model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

				list_of_dropout.append(i)

				for e in epochs:
					list_of_all_dropouts.append(i)
					list_of_epochs.append(e)

					model.fit(X_train, y_train, epochs=e, batch_size=128, verbose=1, validation_split=0.2)
					score = model.evaluate(X_test, y_test, verbose=1)
					list_of_all_scores.append(score)
					
					if score not in list_of_scores:
						list_of_scores.append(score)
            		
            #print('Dropout:', i, '\n', 'Epoch:', e, '\n', 'Score:', float(score))
			lowest = min(list_of_all_scores)
			num = list_of_scores.index(lowest)
			epoch = list_of_epochs[num]
			dropout = list_of_all_dropouts[num]
			print('Lowest score:', lowest, 'Epoch:', epoch, 'Dropout',  dropout)

			return epoch, dropout

		def build_model():

			# epoch, dropout = best_model()
			epoch, dropout = 5, 0.2
			print('EPOCH = ', epoch)
			print('DROPOUT = ', dropout)

			# K-Fold Cross Validation
			cross_validate = False
			if cross_validate:
				k_folds = 10
				# Define the K-fold Cross Validator
				kfold = KFold(n_splits=k_folds, shuffle=True)
				
				# Define per-fold score containers 
				acc_per_fold = []
				loss_per_fold = []
				# Merge inputs and targets
				inputs = np.concatenate((X_train, X_test), axis=0)
				targets = np.concatenate((y_train, y_test), axis=0)

				# K-fold Cross Validation model evaluation
				fold_no = 1
				for train, test in kfold.split(inputs, targets):
					# Build CNN model
					model = Sequential()
					model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
					model.add(Conv1D(filters=max_length, kernel_size=5, padding='same', activation='relu'))
					model.add(GlobalMaxPooling1D())
					model.add(Dropout(dropout))
					model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
					model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
					# print(model.summary())	

					# Generate a print
					print('------------------------------------------------------------------------')
					print(f'Training for fold {fold_no} ...')

					# Train Model
					history = model.fit(inputs[train], targets[train], batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)

					# Evaluate Model
					scores = model.evaluate(inputs[test], targets[test], verbose=0)
					test_acc, test_loss = scores[1], scores[0]
					print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {test_loss}; {model.metrics_names[1]} of {test_acc*100}%')
					acc_per_fold.append(test_acc * 100)
					loss_per_fold.append(test_loss)


					# Increase fold number
					fold_no += 1

				# == Get average scores == #
				print('------------------------------------------------------------------------')
				print('Score per fold')
				
				for i in range(0, len(acc_per_fold)):
					print('------------------------------------------------------------------------')
					print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
					print('------------------------------------------------------------------------')
					print('Average scores for all folds:')
					print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
					print(f'> Loss: {np.mean(loss_per_fold)}')
					print('------------------------------------------------------------------------')

			else:

				# Build CNN model
				model = Sequential()
				model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
				model.add(Conv1D(filters=max_length, kernel_size=5, padding='same', activation='relu'))
				model.add(GlobalMaxPooling1D())
				model.add(Dropout(dropout))
				model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
				model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
				print(model.summary())	
			
				# Train model
				history = model.fit(X_train, y_train, epochs=epoch, batch_size=128, verbose=1, validation_split=0.2)
				# Evaluate model
				loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
				print('Accuracy: %f' % (accuracy*100))
				

			def display():
				plt.plot(history.history['acc'])
				plt.plot(history.history['val_acc'])

				plt.title('model accuracy')
				plt.ylabel('accuracy')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()

				plt.plot(history.history['loss'])
				plt.plot(history.history['val_loss'])

				plt.title('model loss')
				plt.ylabel('loss')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()
			display()
			

		build_model()
	
	def LSTM(self, X_train, y_train, X_test, y_test, vocab_size):
		print('\n=== Recurrent Neural Network with LSTM ===\n')
		
		# Max length of review for embedding
		max_length = 100 
		embedding_dim = 100

		def best_model():

			epochs = [5, 10, 15, 20]
			dropout_rate = [0.1, 0.2, 0.3]
			list_of_all_scores = list()
			list_of_scores = list()
			list_of_dropout = list()
			list_of_all_dropouts = list()
			list_of_epochs = list()

			for i in dropout_rate:
				model = Sequential()
				model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
				model.add(LSTM(128))
				model.add(Dropout(i))
				model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
				model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


				list_of_dropout.append(i)

				for e in epochs:
					list_of_all_dropouts.append(i)
					list_of_epochs.append(e)

					model.fit(X_train, y_train, epochs=e, batch_size=128, verbose=1, validation_split=0.2)
					score = model.evaluate(X_test, y_test, verbose=1)
					list_of_all_scores.append(score)
					
					if score not in list_of_scores:
						list_of_scores.append(score)
            		
            #print('Dropout:', i, '\n', 'Epoch:', e, '\n', 'Score:', float(score))
			lowest = min(list_of_all_scores)
			num = list_of_scores.index(lowest)
			epoch = list_of_epochs[num]
			dropout = list_of_all_dropouts[num]
			print('Lowest score:', lowest, 'Epoch:', epoch, 'Dropout',  dropout)

			return epoch, dropout

		def build_model():

			# epoch, dropout = best_model()
			epoch, dropout = 5, 0.1 # Best 
			print('EPOCH = ', epoch)
			print('DROPOUT = ', dropout)

			# K-Fold Cross Validation
			cross_validate = False
			if cross_validate:
				k_folds = 10
				# Define the K-fold Cross Validator
				kfold = KFold(n_splits=k_folds, shuffle=True)
				
				# Define per-fold score containers 
				acc_per_fold = []
				loss_per_fold = []
				# Merge inputs and targets
				inputs = np.concatenate((X_train, X_test), axis=0)
				targets = np.concatenate((y_train, y_test), axis=0)

				# K-fold Cross Validation model evaluation
				fold_no = 1
				for train, test in kfold.split(inputs, targets):
					# Build LSTM Model
					model = Sequential()
					model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
					model.add(LSTM(128))
					model.add(Dropout(dropout))
					model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
					model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
						

					# Generate a print
					print('------------------------------------------------------------------------')
					print(f'Training for fold {fold_no} ...')

					# Train Model
					history = model.fit(inputs[train], targets[train], batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)

					# Evaluate Model
					scores = model.evaluate(inputs[test], targets[test], verbose=0)
					test_acc, test_loss = scores[1], scores[0]
					print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {test_loss}; {model.metrics_names[1]} of {test_acc*100}%')
					acc_per_fold.append(test_acc * 100)
					loss_per_fold.append(test_loss)


					# Increase fold number
					fold_no += 1

				# == Get average scores == #
				print('------------------------------------------------------------------------')
				print('Score per fold')
				
				for i in range(0, len(acc_per_fold)):
					print('------------------------------------------------------------------------')
					print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
					print('------------------------------------------------------------------------')
					print('Average scores for all folds:')
					print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
					print(f'> Loss: {np.mean(loss_per_fold)}')
					print('------------------------------------------------------------------------')

			else:

				# Build LSTM Model
				model = Sequential()
				model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
				model.add(LSTM(128))
				model.add(Dropout(dropout))
				model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
				model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
				
				# Train model
				history = model.fit(X_train, y_train, epochs=epoch, batch_size=128, verbose=1, validation_split=0.2)
				# Evaluate model
				loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
				print('Accuracy: %f' % (accuracy*100))


			def display():
				plt.plot(history.history['acc'])
				plt.plot(history.history['val_acc'])

				plt.title('model accuracy')
				plt.ylabel('accuracy')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()

				plt.plot(history.history['loss'])
				plt.plot(history.history['val_loss'])

				plt.title('Model Loss')
				plt.ylabel('loss')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()
			display()

		build_model()

	def bi_LSTM(self, X_train, y_train, X_test, y_test, vocab_size):
		print('\n=== Recurrent Neural Network with bidirectional LSTM ===\n')
		
		# Max length of review for embedding
		max_length = 100 
		embedding_dim = 10

		def best_model():

			epochs = [5, 10, 15, 20]
			dropout_rate = [0.1, 0.2, 0.3]
			list_of_all_scores = list()
			list_of_scores = list()
			list_of_dropout = list()
			list_of_all_dropouts = list()
			list_of_epochs = list()

			for i in dropout_rate:
				model = Sequential()
				model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
				model.add(Bidirectional(LSTM(20)))#, dropout=dropout)))
				model.add(Dropout(dropout))
				model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="sigmoid"))
				
				model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

				list_of_dropout.append(i)
					
				for e in epochs:
						list_of_all_dropouts.append(i)
						list_of_epochs.append(e)

						model.fit(X_train, y_train, epochs=e, batch_size=128, verbose=1, validation_split=0.2)
						score = model.evaluate(X_test, y_test, verbose=1)
						list_of_all_scores.append(score)
						
						if score not in list_of_scores:
							list_of_scores.append(score)

			#print('Dropout:', i, '\n', 'Epoch:', e, '\n', 'Score:', float(score))
			lowest = min(list_of_all_scores)
			num = list_of_scores.index(lowest)
			epoch = list_of_epochs[num]
			dropout = list_of_all_dropouts[num]
			print('Lowest score:', lowest, 'Epoch:', epoch, 'Dropout',  dropout)

			return epoch, dropout
		
		def build_model():

			# epoch, dropout = best_model()
			epoch, dropout = 5, 0.1
			print('EPOCH = ', epoch)
			print('DROPOUT = ', dropout)

			# K-Fold Cross Validation
			cross_validate = False
			if cross_validate:
				k_folds = 10
				# Define the K-fold Cross Validator
				kfold = KFold(n_splits=k_folds, shuffle=True)
				
				# Define per-fold score containers 
				acc_per_fold = []
				loss_per_fold = []
				# Merge inputs and targets
				inputs = np.concatenate((X_train, X_test), axis=0)
				targets = np.concatenate((y_train, y_test), axis=0)

				# K-fold Cross Validation model evaluation
				fold_no = 1
				for train, test in kfold.split(inputs, targets):
					# Build Bi_LSTM model
					model = Sequential()
					model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
					model.add(Bidirectional(LSTM(20)))#, dropout=dropout)))
					model.add(Dropout(dropout))
					model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="sigmoid"))
					
					model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

					# Generate a print
					print('------------------------------------------------------------------------')
					print(f'Training for fold {fold_no} ...')

					# Train Model
					history = model.fit(inputs[train], targets[train], batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)

					# Evaluate Model
					scores = model.evaluate(inputs[test], targets[test], verbose=0)
					test_acc, test_loss = scores[1], scores[0]
					print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {test_loss}; {model.metrics_names[1]} of {test_acc*100}%')
					acc_per_fold.append(test_acc * 100)
					loss_per_fold.append(test_loss)


					# Increase fold number
					fold_no += 1

				# == Get average scores == #
				print('------------------------------------------------------------------------')
				print('Score per fold')
				
				for i in range(0, len(acc_per_fold)):
					print('------------------------------------------------------------------------')
					print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
					print('------------------------------------------------------------------------')
					print('Average scores for all folds:')
					print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
					print(f'> Loss: {np.mean(loss_per_fold)}')
					print('------------------------------------------------------------------------')

			else:
				# Build Bi_LSTM model
				model = Sequential()
				model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
				model.add(Bidirectional(LSTM(20)))#, dropout=dropout)))
				model.add(Dropout(dropout))
				model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="sigmoid"))
				
				model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

				print(model.summary())	

				# Train the model
				history = model.fit(X_train, y_train, epochs=epoch, batch_size=128, verbose=1, validation_split=0.2)
				# Evaluate the model
				loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
				print('Accuracy: %f' % (accuracy * 100))



			def display():
				plt.plot(history.history['acc'])
				plt.plot(history.history['val_acc'])

				plt.title('model accuracy')
				plt.ylabel('accuracy')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()

				plt.plot(history.history['loss'])
				plt.plot(history.history['val_loss'])

				plt.title('Model Loss')
				plt.ylabel('loss')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()
			display()

		build_model()

	
	def GRU(self, X_train, y_train, X_test, y_test, vocab_size):
		print('\n=== Gated Recurrent Network ===\n')
		
		# Max length of review for embedding
		max_length = 100 
		embedding_dim = 10

		def best_model():

			epochs = [5, 10, 15, 20]
			dropout_rate = [0.1, 0.2, 0.3]
			list_of_all_scores = list()
			list_of_scores = list()
			list_of_dropout = list()
			list_of_all_dropouts = list()
			list_of_epochs = list()

			for i in dropout_rate:
		
				model = Sequential()
				model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
				model.add(GRU(50, return_sequences=True))
				model.add(GRU(1, return_sequences=False))
				model.add(Dropout(i))
				model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
				model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    
				list_of_dropout.append(i)
					
				for e in epochs:
						list_of_all_dropouts.append(i)
						list_of_epochs.append(e)

						model.fit(X_train, y_train, epochs=e, batch_size=128, verbose=1, validation_split=0.2)
						score = model.evaluate(X_test, y_test, verbose=1)
						list_of_all_scores.append(score)
						
						if score not in list_of_scores:
							list_of_scores.append(score)

			#print('Dropout:', i, '\n', 'Epoch:', e, '\n', 'Score:', float(score))
			lowest = min(list_of_all_scores)
			num = list_of_scores.index(lowest)
			epoch = list_of_epochs[num]
			dropout = list_of_all_dropouts[num]
			print('Lowest score:', lowest, 'Epoch:', epoch, 'Dropout',  dropout)

			return epoch, dropout

		def build_model():

			# epoch, dropout = best_model()
			epoch, dropout = 20, 0.3
			print('EPOCH = ', epoch)
			print('DROPOUT = ', dropout)

			# K-Fold Cross Validation
			cross_validate = False
			if cross_validate:
				k_folds = 10
				# Define the K-fold Cross Validator
				kfold = KFold(n_splits=k_folds, shuffle=True)
				
				# Define per-fold score containers 
				acc_per_fold = []
				loss_per_fold = []
				# Merge inputs and targets
				inputs = np.concatenate((X_train, X_test), axis=0)
				targets = np.concatenate((y_train, y_test), axis=0)

				# K-fold Cross Validation model evaluation
				fold_no = 1
				for train, test in kfold.split(inputs, targets):
					model = Sequential()
					model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
					model.add(GRU(50, return_sequences=True))
					model.add(GRU(1, return_sequences=False))
					model.add(Dropout(dropout))
					model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
					model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
		    
					# print(model.summary())	

					# Generate a print
					print('------------------------------------------------------------------------')
					print(f'Training for fold {fold_no} ...')

					# Train Model
					history = model.fit(inputs[train], targets[train], batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)

					# Evaluate Model
					scores = model.evaluate(inputs[test], targets[test], verbose=0)
					test_acc, test_loss = scores[1], scores[0]
					print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {test_loss}; {model.metrics_names[1]} of {test_acc*100}%')
					acc_per_fold.append(test_acc * 100)
					loss_per_fold.append(test_loss)


					# Increase fold number
					fold_no += 1

				# == Get average scores == #
				print('------------------------------------------------------------------------')
				print('Score per fold')
				
				for i in range(0, len(acc_per_fold)):
					print('------------------------------------------------------------------------')
					print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
					print('------------------------------------------------------------------------')
					print('Average scores for all folds:')
					print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
					print(f'> Loss: {np.mean(loss_per_fold)}')
					print('------------------------------------------------------------------------')


				else:
					# Train the model
					history = model.fit(X_train, y_train, epochs=epoch, batch_size=128, verbose=1, validation_split=0.2)
					# Evaluate the model
					loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
					print('Accuracy: %f' % (accuracy * 100))

			
			def display():
				plt.plot(history.history['acc'])
				plt.plot(history.history['val_acc'])

				plt.title('model accuracy')
				plt.ylabel('accuracy')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()

				plt.plot(history.history['loss'])
				plt.plot(history.history['val_loss'])

				plt.title('Model Loss')
				plt.ylabel('loss')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()
			display()

		build_model()

if __name__ == "__main__":

    file = '/Users/selina/Code/Python/SSL/CNN_vs_RNN/IMDB_Dataset.csv'
    extractor = Sentiment_Analysis(file)
    X_train, y_train, X_test, y_test, vocab_size = extractor.pre_process(file)
    extractor.CNN(X_train, y_train, X_test, y_test, vocab_size)
    # extractor.LSTM(X_train, y_train, X_test, y_test, vocab_size)
    # extractor.bi_LSTM(X_train, y_train, X_test, y_test, vocab_size)
    # extractor.GRU(X_train, y_train, X_test, y_test, vocab_size)


