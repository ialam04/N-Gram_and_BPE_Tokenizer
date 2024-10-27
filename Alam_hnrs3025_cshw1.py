import re
import pickle
import argparse
import random
import collections


class NGramModel:
    def __init__(self, n):
        """
        Initializes the NGram model with the order and the model
        :param n: (int) The order of the NGram model
        """
        self.n = n
        self.model = {}

    def train(self, corpus):
        """
        Trains the model with corpus using NGram model based on whether the argument is bigram or unigram
        :param corpus: (str) The corpus to be trained on
        :raise Exception if invalid NGram order entered
        """

        # Splitting corpus into inout string and initializing model
        tokens = re.findall(r'\w+|[^\w\s]', corpus)
        self.model = {}
        one_word_frequency = collections.Counter(tokens)

        # Combining adjacent tokens and finding the frequency of pairs
        unigram_list = list(zip(tokens[:-1], tokens[1:]))
        unigram_frequency = collections.Counter(unigram_list)

        # Unigram
        if self.n == 1:
            # Calculate conditional probability for each adjacent word pair
            for (w1, w2), count in unigram_frequency.items():

                if w1 not in self.model:
                    self.model[w1] = {}
                self.model[w1][w2] = count / one_word_frequency[w1]

        # Bigram
        elif self.n == 2:
            # Combining adjacent three tokens and finding their frequency
            bigram_list = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
            bigram_frequency = collections.Counter(bigram_list)

            # Calculation conditional probability for last word based on first two in combined tokens
            for (w1, w2, w3), count in bigram_frequency.items():
                if w1 not in self.model:
                    self.model[w1] = {}
                if w2 not in self.model[w1]:
                    self.model[w1][w2] = {}
                self.model[w1][w2][w3] = count / unigram_frequency[(w1, w2)]
        else:
            # Throw exception for invalid ngram order
            raise Exception("Invalid NGram Model Order")

    def predict_next_word(self, input_tuple, deterministic=False):
        """
        Predicts the next word based on the input and the trained NGram Model
        :param input_tuple: (tuple) Input to use for prediction, one word for unigram and two for bigram
        :param deterministic: (boolean) Decides to sample by highest probability or randomly by probability distribution
        :return (string) The next word in the sequence for the sequence in generated text
        :raise Exception if words not in vocabulary or invalid NGram order
        """

        # Unigram
        if self.n == 1:
            previous_word = input_tuple

            # Raise an error if the input word is not in the vocabulary
            if previous_word not in self.model:
                raise Exception(f"Word '{previous_word}' not found in the unigram model")

            # Selecting word with the highest probability and returning it if it is deterministic
            if deterministic:
                next_word = max(self.model[previous_word].items(), key=lambda x: x[1])[0]
                return next_word

            # Randomly sample words and return word based on probability distribution otherwise
            else:
                words, probs = zip(*self.model[previous_word].items())
                next_word = random.choices(words, weights=probs)[0]
                return next_word

        # Bigram
        elif self.n == 2:
            w1, w2 = input_tuple[-2:]

            # Raise an error if any of the input words are not in the vocabulary
            if w1 not in self.model or w2 not in self.model[w1]:
                raise Exception(f"Words '{w1}' and '{w2}' not found in the bigram model")

            # Selecting word with the highest probability and returning it if it is deterministic
            if deterministic:
                next_word = max(self.model[w1][w2].items(), key=lambda x: x[1])[0]
                return next_word

            # Randomly sample words and return word based on probability distribution otherwise
            else:
                words, probs = zip(*self.model[w1][w2].items())
                next_word = random.choices(words, weights=probs)[0]
                return next_word
        else:
            # Throw exception for invalid ngram order
            raise Exception("Invalid NGram Model Order")


class BPEAlgorithm:
    def __init__(self):
        self.vocabulary = {}

    def train(self, corpus, k=500):
        """
        Trains the BPE algorithm based on the provided corpus to provide a vocabulary to tokenize with
        :param corpus: (str) The string to train the BPE algorithm on
        :param k: (int) Number of iterations that the trains BPE algorithm
        """

        # Splitting corpus into list and populating vocabulary as well as initializing pair counter
        corpus_split = list(corpus)
        self.vocabulary = corpus_split
        pair_counter = collections.defaultdict(int)

        # Training/algorithm k number of times, k -> training iteration number
        for i in range(k):

            # Iterating through corpus and finding most frequent pair for every iteration of k
            for j in range(len(corpus_split) - 1):
                pair = corpus_split[j] + corpus_split[j + 1]
                pair_counter[pair] += 1
            most_frequent_pair_key = max(pair_counter, key=pair_counter.get)

            # If the most frequent pair's count is only one then break
            if pair_counter[most_frequent_pair_key] <= 1:
                break

            # Add the most frequent pair to the vocabulary
            self.vocabulary.append(most_frequent_pair_key)
            n = 0
            print(i)

            # Remove the individual instances of both members of the pair in the vocabulary
            while n < len(corpus_split) - 1:
                first = corpus_split[n]
                second = corpus_split[n + 1]
                if first + second == most_frequent_pair_key:
                    corpus_split[n] = most_frequent_pair_key
                    corpus_split.pop(n + 1)
                else:
                    n += 1

    def tokenize(self, tokenize_string):
        """
        Tokenize the input string based on the vocabulary
        :param tokenize_string: (str) String to tokenize
        :return (tuple) tokens, token_ids - List of tokens combined with list of token ids for corresponding tokens
        """

        # Getting list of tokens to tokenize
        tokens = list(tokenize_string)

        # Sorting vocabulary in reverse order to get the greatest possible match for tokens
        sorted_vocabulary = sorted(self.vocabulary, key=len, reverse=True)

        # Iterating through tokens in sorted vocabulary
        for token in sorted_vocabulary:
            token_length = len(token)
            index = 0

            # While the index is still in the bounds of the tokens list
            while index <= len(tokens) - token_length:

                # Iterating through and replacing sliced token with token in vocabulary if there is a match
                if ''.join(tokens[index:index + token_length]) == token:
                    tokens[index:index + token_length] = [token]
                    index += token_length
                else:
                    index += 1

        # Assign each token its corresponding index or token_id
        token_ids = [self.vocabulary.index(token) for token in tokens]

        return tokens, token_ids


def main():
    """Main function to implement CLI and parse arguments for activities."""

    # Creating parser and defining args
    parser = argparse.ArgumentParser(description='N-gram Model and BPE tokenizer.')

    parser.add_argument('activity', type=str, choices=['train_ngram', 'predict_ngram', 'train_bpe', 'tokenize'])

    parser.add_argument('--data', type=str)

    parser.add_argument('--save', type=str)

    parser.add_argument('--load', type=str)

    parser.add_argument('--word', type=str)

    parser.add_argument('--nwords', type=int)

    parser.add_argument('--text', type=str)

    parser.add_argument('--n', type=int, choices=[1, 2])

    parser.add_argument('--k', type=int, default=500)

    parser.add_argument('--d', action='store_true')

    args = parser.parse_args()

    if args.activity == 'train_ngram':
        # Opening and reading data file
        with open(args.data, 'r', encoding='utf-8') as f:
            data = f.read()

        # Initializing and training NGram model
        model = NGramModel(args.n)
        model.train(data)

        # Saving trained NGram model to specified file path
        with open(args.save, 'wb') as f:
            pickle.dump(model, f)
        print("N-gram model trained and saved to", args.save)

    elif args.activity == 'predict_ngram':
        # Loading pre-trained NGram model
        with open(args.load, 'rb') as f:
            model = pickle.load(f)

        # Unigram model
        if model.n == 1:

            # Getting input word and raising exception if there is none
            input_word = args.word.strip()
            if not input_word:
                raise Exception("Error: Unigram requires one input word.")

            # Generating words beginning from the input word
            generated_words = [input_word]
            for _ in range(args.nwords):
                next_word = model.predict_next_word(input_word, deterministic=args.d)
                generated_words.append(next_word)
                input_word = next_word
            print('Generated text:', ' '.join(generated_words))

        # Bigram model
        elif model.n == 2:

            # Getting input word and raising exception if there is none or not 2
            input_words = tuple(args.word.strip().split())
            if len(input_words) != 2:
                raise Exception("Error: Bigram requires two input words.")

            # Generating words beginning from the input words
            generated_words = list(input_words)
            for _ in range(args.nwords):
                next_word = model.predict_next_word(tuple(generated_words[-2:]), deterministic=args.d)
                generated_words.append(next_word)
            print('Generated text:', ' '.join(generated_words))

    elif args.activity == 'tokenize':

        # Loading pre-trained BPE Model
        with open(args.load, 'rb') as f:
            model = pickle.load(f)

        # Tokenizing input text and print results
        tokens = model.tokenize(args.text)
        print('Tokens:', tokens)

    elif args.activity == 'train_bpe':

        # Opening and reading data file
        with open(args.data, 'r', encoding='utf-8') as f:
            data = f.read()

        # Initialize and train BPE Model
        model = BPEAlgorithm()
        model.train(data, k=args.k)

        # Saving trained BPE model to specific file path
        with open(args.save, 'wb') as f:
            pickle.dump(model, f)
        print(f"BPE model trained and saved to {args.save}")


if __name__ == '__main__':
    main()
