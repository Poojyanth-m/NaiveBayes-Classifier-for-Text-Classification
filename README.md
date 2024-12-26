
# Text Classifier - Naive Bayes 📚

## Project Overview

The **Text Classifier** is a machine learning project based on the **Naive Bayes algorithm**, designed to categorize text documents into predefined categories. This project demonstrates the application of Naive Bayes classification on cricket-related documents, including news, players, matches, and more. The model is built to process text data, preprocess it, and classify it into specific categories.

## Features

- **Text Preprocessing**: Cleans the text by removing punctuation, converting it to lowercase, and removing common stop words (like "a", "the", "in", etc.).
- **Naive Bayes Classifier**: A probabilistic machine learning model used to classify documents based on word frequencies.
- **Customizable**: You can easily change the dataset or categories to classify any type of text documents.

## Categories

The classifier currently works with cricket-related categories. These are:

- **Cricket News** 📰
- **Cricket Players** 🏏
- **Cricket Matches** 🎮
- **Cricket Tournaments** 🏆
- **Cricket Equipment** 🏏

## Technologies Used

- **Python** 🐍: Main programming language.
- **Naive Bayes Algorithm**: Used for classification based on conditional probabilities.
- **Regular Expressions (re module)**: Used for text preprocessing (removal of punctuation, stop words, etc.).
- **Collections module (defaultdict)**: Used for counting word frequencies and document occurrences.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Poojyanth-m/text-classifier-naive-bayes.git
   ```

2. Install Python (if not already installed). This project only requires the standard Python library, so no additional packages are needed.

## Usage

### Training the Classifier

To train the model, you need to prepare your dataset. The dataset should contain a list of documents, each paired with a category. The model will then learn the word frequencies for each category.

```python
docs = [
    ('Cricket World Cup 2023 starts today!', 'Cricket News'),
    ('Sachin Tendulkar retires from international cricket', 'Cricket Players'),
    ('India vs Australia, 3rd ODI, 2023', 'Cricket Matches'),
    ('IPL 2023 Final scheduled for next week', 'Cricket Tournaments'),
    ('Best cricket bats for beginners', 'Cricket Equipment'),
]
```

### Training the Naive Bayes Model

After preparing the training data (`docs`), instantiate the Naive Bayes classifier and train it using the `train()` method.

```python
nb = NaiveBayes(['Cricket News', 'Cricket Players', 'Cricket Matches', 'Cricket Tournaments', 'Cricket Equipment'])
nb.train(docs)
```

### Predicting the Category

Once trained, you can classify new text documents using the `predict()` method. Simply provide the text to be classified.

```python
new_doc = 'Virat Kohli scores a century in the match'
category = nb.predict(new_doc)
print(f'The document "{new_doc}" belongs to the category "{category}"')
```

### Example Output

```bash
The document "Virat Kohli scores a century in the match" belongs to the category "Cricket Players"
```

### Modifying the Dataset

To change the dataset or categories:

1. **Change Categories**: Modify the list of categories in the classifier initialization.
   ```python
   nb = NaiveBayes(['Cricket News', 'Cricket Players', 'Cricket Matches', 'Cricket Tournaments', 'Cricket Equipment'])
   ```

2. **Update Training Data**: Modify the `docs` list to include new documents and categories.

## How It Works

1. **Preprocessing**: The text is cleaned by removing punctuations, converting it to lowercase, and filtering out stop words.
2. **Training**: The model calculates word frequencies for each category in the dataset.
3. **Prediction**: The model uses the Naive Bayes algorithm to calculate the probabilities of a document belonging to each category and classifies it based on the highest probability.

## Contributing

Feel free to fork the repository, improve the code, or contribute by:

1. Forking the repo
2. Creating a new branch (`git checkout -b feature-branch`)
3. Making changes and committing (`git commit -am 'Add new feature'`)
4. Pushing to the branch (`git push origin feature-branch`)
5. Opening a pull request
