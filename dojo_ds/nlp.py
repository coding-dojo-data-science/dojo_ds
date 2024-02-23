"""
nlp functions from Coding Dojo Online Data Science and Machine Learning 24-Week Program.
"""


def reference_set_seed_keras(markdown=True):
	
	ref = """
	```python
	# From source: https://keras.io/examples/keras_recipes/reproducibility_recipes/
	import tensorflow as tf
	import numpy as np

	# Then Set Random Seeds
	tf.keras.utils.set_random_seed(42)
	tf.random.set_seed(42)
	np.random.seed(42)

	# Then run the Enable Deterministic Operations Function
	tf.config.experimental.enable_op_determinism()
	```
	"""
	if markdown:
		from IPython.display import display, Markdown
		display(Markdown(ref))
	else:
		print(ref)


def make_text_vectorization_layer(train_ds,  max_tokens=None, 
								  split='whitespace',
								  standardize="lower_and_strip_punctuation",
								  output_mode="int",
								  output_sequence_length=None,
								  ngrams=None, pad_to_max_tokens=False,
								  verbose=True,
								  **kwargs,
								 ):
	"""
	Creates a text vectorization layer using TensorFlow's TextVectorization class.

	Parameters:
	- train_ds: The training dataset containing the text data.
	- max_tokens: The maximum number of tokens to keep in the vocabulary.
	- split: The method used to split the text into tokens.
	- standardize: The method used to standardize the text.
	- output_mode: The output mode of the layer.
	- output_sequence_length: The length of the output sequences.
	- ngrams: The n-grams to consider when tokenizing the text.
	- pad_to_max_tokens: Whether to pad the sequences to have the same length.
	- verbose: Whether to print the layer's parameters.
	- **kwargs: Additional keyword arguments to pass to the TextVectorization class.

	Returns:
	- text_vectorizer: The text vectorization layer.
	- int_to_str: A dictionary mapping integers to words in the vocabulary.
	"""
	import tensorflow as tf
	import numpy as np
	from pprint import pprint

	# Build the text vectorization layer
	text_vectorizer = tf.keras.layers.TextVectorization(
		max_tokens=max_tokens,
		standardize=standardize, 
		output_mode=output_mode,
		output_sequence_length=output_sequence_length,
		**kwargs
	)
	# Get just the text from the training data
	
	if isinstance(train_ds, (np.ndarray, list, tuple, pd.Series)):
		ds_texts = train_ds
	else:
		try:
			ds_texts = train_ds.map(lambda x, y: x )
		except:
			ds_texts = train_ds
			
	# Fit the layer on the training texts
	text_vectorizer.adapt(ds_texts)
	
	
	if verbose:
		# Print the params
		print( "\ntf.keras.layers.TextVectorization(" )
		config = text_vectorizer.get_config()
		pprint(config,indent=4)
		print(")")
			   
	# SAVING VOCAB FOR LATER
	# Getting list of vocab 
	vocab = text_vectorizer.get_vocabulary()
	# Save dictionaries to look up words from ints 
	int_to_str  = {idx:word for idx, word in enumerate(vocab)}
	
	return text_vectorizer, int_to_str


def batch_preprocess_texts(
	texts,
	nlp=None,
	remove_stopwords=True,
	remove_punct=True,
	use_lemmas=False,
	disable=["ner"],
	batch_size=50,
	n_process=-1,
):
	"""Efficiently preprocess a collection of texts using nlp.pipe()

	Args:
		texts (collection of strings): Collection of texts to process (e.g. df['text'])
		nlp (spacy pipe), optional): Spacy nlp pipe. Defaults to None; if None, it creates a default 'en_core_web_sm' pipe.
		remove_stopwords (bool, optional): Controls stopword removal. Defaults to True.
		remove_punct (bool, optional): Controls punctuation removal. Defaults to True.
		use_lemmas (bool, optional): Lemmatize tokens. Defaults to False.
		disable (list of strings, optional): Named pipeline elements to disable. Defaults to ["ner"]: Used with nlp.pipe(disable=disable)
		batch_size (int, optional): Number of texts to process in a batch. Defaults to 50.
		n_process (int, optional): Number of CPU processors to use. Defaults to -1 (meaning all CPU cores).

	Returns:
		list of tokens: Processed texts as a list of tokens.
	"""
	# Function implementation


import pandas as pd
def get_ngram_measures_finder(tokens, ngrams=2, measure='raw_freq', top_n=None, min_freq=1, words_colname='Words'):
	"""
	Calculate n-gram measures for a given list of tokens.

	Parameters:
	- tokens (list): List of tokens.
	- ngrams (int): Number of grams to consider (2 for bigrams, 3 for trigrams, 4 for quadgrams). Default is 2.
	- measure (str): Measure to calculate ('raw_freq' for raw frequency, 'pmi' for pointwise mutual information). Default is 'raw_freq'.
	- top_n (int): Number of top n-grams to return. Default is None (return all n-grams).
	- min_freq (int): Minimum frequency threshold for n-grams. Default is 1.
	- words_colname (str): Column name for the n-grams in the resulting DataFrame. Default is 'Words'.

	Returns:
	- df_ngrams (DataFrame): DataFrame containing the n-grams and their corresponding measure values.
	"""
	import nltk
	import pandas as pd

	if ngrams == 4:
		MeasuresClass = nltk.collocations.QuadgramAssocMeasures
		FinderClass = nltk.collocations.QuadgramCollocationFinder
	elif ngrams == 3:
		MeasuresClass = nltk.collocations.TrigramAssocMeasures
		FinderClass = nltk.collocations.TrigramCollocationFinder
	else:
		MeasuresClass = nltk.collocations.BigramAssocMeasures
		FinderClass = nltk.collocations.BigramCollocationFinder

	measures = MeasuresClass()

	finder = FinderClass.from_words(tokens)
	finder.apply_freq_filter(min_freq)
	if measure == 'pmi':
		scored_ngrams = finder.score_ngrams(measures.pmi)
	else:
		measure = 'raw_freq'
		scored_ngrams = finder.score_ngrams(measures.raw_freq)

	df_ngrams = pd.DataFrame(scored_ngrams, columns=[words_colname, measure.replace("_", ' ').title()])
	if top_n is not None:
		return df_ngrams.head(top_n)
	else:
		return df_ngrams

# def get_ngram_measures_finder(tokens=None,docs=None, ngrams=2, verbose=False,
#                               get_scores_df=False, measure='raw_freq', top_n=None,
#                              words_colname='Words'):
#     import nltk
#     if ngrams == 4:
#         MeasuresClass = nltk.collocations.QuadgramAssocMeasures
#         FinderClass = nltk.collocations.QuadgramCollocationFinder
		
#     elif ngrams == 3: 
#         MeasuresClass = nltk.collocations.TrigramAssocMeasures
#         FinderClass = nltk.collocations.TrigramCollocationFinder
#     else:
#         MeasuresClass = nltk.collocations.BigramAssocMeasures
#         FinderClass = nltk.collocations.BigramCollocationFinder

#     measures = MeasuresClass()
	
#     if (tokens is not None):
#         finder = FinderClass.from_words(tokens)
#     elif (docs is not None):
#         finder = FinderClass.from_docs(docs)
#     else:
#         raise Exception("Must provide tokens or docs")
		

#     if get_scores_df == False:
#         return measures, finder
#     else:
#         df_ngrams = get_score_df(measures, finder, measure=measure, top_n=top_n, words_colname=words_colname)
#         return df_ngrams




# def get_score_df( measures,finder, measure='raw_freq', top_n=None, words_colname="Words"):
#     import pandas as pd
#     if measure=='pmi':
#         scored_ngrams = finder.score_ngrams(measures.pmi)
#     else:
#         measure='raw_freq'
#         scored_ngrams = finder.score_ngrams(measures.raw_freq)

#     df_ngrams = pd.DataFrame(scored_ngrams, columns=[words_colname, measure.replace("_",' ').title()])
#     if top_n is not None:
#         return df_ngrams.head(top_n)
#     else:
#         return df_ngrams


	

def preprocess_text(txt, nlp=None, remove_stopwords=True, remove_punct=True, use_lemmas=False,):
	"""
	Preprocesses the given text by tokenizing and optionally removing stopwords, punctuation, and lemmatizing the tokens.

	Args:
		txt (str): The text to be processed.
		nlp (spacy pipe, optional): The Spacy nlp pipe. Defaults to None.
								   If None, it creates a default 'en_core_web_sm' pipe.
		remove_stopwords (bool, optional): Controls whether to remove stopwords. Defaults to True.
		remove_punct (bool, optional): Controls whether to remove punctuation. Defaults to True.
		use_lemmas (bool, optional): Controls whether to lemmatize tokens. Defaults to False.

	Returns:
		list: A list of tokens/lemmas after preprocessing.
	"""
	import spacy
	if nlp is None:
		nlp = spacy.load('en_core_web_sm')

	doc = nlp(txt)

	# Saving list of the token objects for stopwords and punctuation removal
	tokens = []

	for token in doc:
		# Check if should remove stopwords and if token is stopword
		if (remove_stopwords == True) & (token.is_stop == True):
			# Continue the loop with the next token
			continue

		# Check if should remove punctuation and if token is punctuation
		if (remove_punct == True) & (token.is_punct == True):
			# Continue the loop with the next token
			continue

		# Check if should remove punctuation and if token is a space
		if (remove_punct == True) & (token.is_space == True):
			# Continue the loop with the next token
			continue

		# Determine final form of output list of tokens/lemmas
		if use_lemmas:
			tokens.append(token.lemma_.lower())
		else:
			tokens.append(token.text.lower())

	return tokens





def make_custom_nlp(
	disable=["ner"],
	contractions=["don't", "can't", "couldn't", "you'd", "I'll"],
	stopwords_to_add=[],
	stopwords_to_remove=[],
	spacy_model = "en_core_web_sm"
):
	"""Returns a custom spacy nlp pipeline.
	
	Args:
		disable (list, optional): Names of pipe components to disable. Defaults to ["ner"].
		contractions (list, optional): List of contractions to add as special cases. Defaults to ["don't", "can't", "couldn't", "you'd", "I'll"].
		stopwords_to_add(list, optional): List of words to set as stopwords (word.is_stop=True)
		stopwords_to_remove(list, optional): List of words to remove from stopwords (word.is_stop=False)
		spacy_model (str, optional): Name or path of the spaCy model to load. Defaults to "en_core_web_sm".
			
	Returns:
		nlp pipeline: spacy pipeline with special cases and updated nlp.Default.stopwords
	"""
	import spacy
	# Load the English NLP model
	nlp = spacy.load(spacy_model, disable=disable)
	

	## Adding Special Cases 
	# Loop through the contractions list and add special cases
	for contraction in contractions:
		special_case = [{"ORTH": contraction}]
		nlp.tokenizer.add_special_case(contraction, special_case)

	
	## Adding stopwords
	for word in stopwords_to_add:
		# Set the is_stop attribute for the word in the vocab dict to true.
		nlp.vocab[
			word
		].is_stop = True  # this determines spacy's treatmean of the word as a stop word

		# Add the word to the list of stopwords (for easily tracking stopwords)
		nlp.Defaults.stop_words.add(word)

	
	## Removing Stopwords
	for word in stopwords_to_remove:
		
		# Ensure the words are not recognized as stopwords
		nlp.vocab[word].is_stop = False
		nlp.Defaults.stop_words.discard(word)
		
	return nlp




# def custom_preprocess_text(
# 	txt,
# 	nlp=None,
# 	nlp_creation_fn=None,
# 	nlp_fn_kwargs={},  ## THESE ARE NEW SINCE BRENDA SAW
# 	lowercase=True,
# 	remove_stopwords=True,
# 	remove_punct=True,
# 	use_lemmas=False,
# 	disable=None,
# ):
# 	"""Preprocess text into tokens/lemmas. 
	
# 	Args:
# 		txt (string): text to process
# 		nlp (spacy pipe), optional): Spacy nlp pipe. Defaults to None; if None, it creates a default 'en_core_web_sm' pipe.
# 		nlp_creation_fn (_type_, optional): Function that returns an nlp pipe. Defaults to None; Only used if nlp arg is None.
# 		nlp_fn_kwargs (dict, optional): Keyword arguments for nlp_creation_fn. Defaults to {}.
# 		remove_stopwords (bool, optional): Controls stopword removal. Defaults to True.
# 		remove_punct (bool, optional): Controls punctuation removal. Defaults to True.
# 		use_lemmas (bool, optional): lemmatize tokens. Defaults to False.
# 		disable (list of strings, optional): named pipeline elements to disable. Defaults to None;Only used if nlp is None and nlp_creation_fn is None
	
# 	Returns:
# 		list: list of tokens/lemmas
# 	"""
# 	# If nlp is none, use nlp_creation_func to make it
# 	if (nlp is None) and (nlp_creation_fn is not None):
# 		nlp = nlp_creation_fn(**nlp_fn_kwargs)
	
# 	# If nlp is none,and no nlp_creation_func, make default nlp object
# 	elif (nlp is None) & (nlp_creation_fn is None):
# 		nlp = spacy.load("en_core_web_sm")

# 	# Create the document
# 	doc = nlp(txt)
	
# 	# Saving list of the token objects for stopwords and punctuation removal
# 	tokens = []
	
# 	for token in doc:
# 		# Check if should remove stopwords and if token is stopword
# 		if (remove_stopwords == True) & (token.is_stop == True):
# 			# Continue the loop with the next token
# 			continue
	
# 		# Check if should remove punctuation and if token is punctuation
# 		if (remove_punct == True) & (token.is_punct == True):
# 			# Continue the loop with the next oken
# 			continue

# 		# Check if should remove punctuation and if token is a space
# 		if (remove_punct == True) & (token.is_space == True):
# 			# Continue the loop with the next oken
# 			continue
	
# 		## Determine final form of output list of tokens/lemmas
# 		if use_lemmas:
# 			tokens.append(token.lemma_)
# 		elif lowercase==True:
# 			tokens.append(token.text.lower())
# 		else: 
# 			tokens.append(token.text)
	
# 	return tokens


