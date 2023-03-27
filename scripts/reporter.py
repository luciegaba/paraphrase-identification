
from collections import Counter
import matplotlib.pyplot as plt

def most_common_words_with_frequency(observation_list):
    concatenated_text = " ".join([str(elem) for elem in observation_list])
    words = concatenated_text.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(30)
    top_word_list = [word[0] for word in top_words]
    top_word_freq = [word[1] for word in top_words]
    plt.bar(top_word_list, top_word_freq)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 mots les plus fréquents')
    plt.xticks(rotation=45)
    plt.show()