import re
from collections import defaultdict

class NaiveBayes:
    def __init__(self, classes):
        self.classes = classes
        self.vocab = set()
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_doc_counts = defaultdict(int)
        self.num_docs = 0

    def preprocess(self, text):
        # Remove punctuations and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text).lower()
        # Remove stop words
        stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'to', 'for', 'by', 'with', 'from', 'and'])
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def train(self, documents):
        for document, category in documents:
            tokens = self.preprocess(document)
            self.vocab.update(tokens)
            self.class_doc_counts[category] += 1
            self.num_docs += 1
            for word in tokens:
                self.class_word_counts[category][word] += 1

    def predict(self, document):
        tokens = self.preprocess(document)
        posteriors = {category: 0 for category in self.classes}
        for category in self.classes:
            prior = self.class_doc_counts[category] / self.num_docs
            posterior = prior
            for word in tokens:
                word_count = self.class_word_counts[category][word]
                total_count = sum(self.class_word_counts[category].values())
                conditional = word_count / total_count if total_count > 0 else 1  # Avoid division by zero
                posterior *= conditional
            posteriors[category] = posterior
        return max(posteriors, key=posteriors.get)


# Training data for cricket-related categories
docs = [
    ('The match was thrilling and intense', 'match'),
    ('He hit a beautiful six over the boundary', 'batting'),
    ('The bowler took a hat-trick in the match', 'bowling'),
    ('The cricket team won the series 3-0', 'team'),
    ('He is the best cricket captain in the world', 'captain'),
    ('The ground was full of spectators cheering for their team', 'stadium'),
    ('He bowled the ball at 150 km/h', 'bowling'),
    ('The batsman was dismissed after scoring a century', 'batting'),
    ('The umpire made a wrong decision during the match', 'umpire'),
    ('The cricket tournament is scheduled for next week', 'tournament'),
]

# Instantiate and train the Naive Bayes classifier
nb = NaiveBayes(['match', 'batting', 'bowling', 'team', 'captain', 'stadium', 'umpire', 'tournament'])
nb.train(docs)

# Predict the category of new documents
new_doc1 = 'The team is preparing for the next match'
new_doc2 = 'He is a great batsman'
new_doc3 = 'The bowling performance was outstanding'

category1 = nb.predict(new_doc1)
category2 = nb.predict(new_doc2)
category3 = nb.predict(new_doc3)

# Output the results
print(f'The document "{new_doc1}" belongs to the category "{category1}"')
print(f'The document "{new_doc2}" belongs to the category "{category2}"')
print(f'The document "{new_doc3}" belongs to the category "{category3}"')
