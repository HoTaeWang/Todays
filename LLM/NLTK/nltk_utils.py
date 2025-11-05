import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

# Return the list of unique_words
def unique_word_count(freq_dist):
    unique_words = freq_dist.keys()
    print('Unique word counts: ', len(unique_words))
    return list(unique_words)
    #return len(unique_words)

# Remove stop word of the English
# Example of stop words: 'the', 'a', 'an', 'is', 'of'...
def remove_english_stopwords(freq_dist):
    english_stopwords = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    unique_words = unique_word_count(freq_dist)
    filtered_unique_words = [word for word in unique_words if word.lower() not in english_stopwords and word not in punctuation]
    print('Total word counts after removing english stopwords and punctuation', len(filtered_unique_words))
    return filtered_unique_words

    