from gensim.utils import simple_preprocess
from nltk import ngrams, pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from pprint import pprint
from collections import Counter
import textstat
import spacy
from utils.helpers import load_only_prompt, load_prompt

nlp = spacy.load("en_core_web_sm")

LOGICAL_OPERATORS = {
    "and",
    "or",
    "not",
    "if",
    "then",
    "unless",
    "whether",
    "although",
    "but",
}


class FeatureExtractor:
    def __init__(
        self,
        set_id,
        word2vec_model,
        doc2vec_model,
        ngram_results,
        weighted_keywords,
    ):
        self.prompt = load_prompt(set_id)
        self.only_prompt = load_only_prompt(set_id)
        self.word2vec_model = word2vec_model
        self.doc2vec_model = doc2vec_model
        self.ngram_results = ngram_results
        self.weighted_keywords = weighted_keywords

    def word2vec(self, student_answer):
        words = simple_preprocess(student_answer)
        vectors = [
            self.word2vec_model[word] for word in words if word in self.word2vec_model
        ]
        if vectors:
            average_vector = sum(vectors) / len(vectors)
        else:
            average_vector = None

        return average_vector

    def doc2vec(self, student_answer):
        tokenized_answer = simple_preprocess(student_answer)
        vector = self.doc2vec_model.infer_vector(tokenized_answer)

        return vector

    def pos(self, student_answer):
        tokens = word_tokenize(student_answer)
        pos_tags = pos_tag(tokens)

        # Generate bi-grams, tri-grams, and tetra-grams
        bi_grams = list(ngrams(pos_tags, 2))
        tri_grams = list(ngrams(pos_tags, 3))
        tetra_grams = list(ngrams(pos_tags, 4))

        # Combine all n-grams
        all_ngrams = bi_grams + tri_grams + tetra_grams

        # Generate a vector for n-grams indicating overlap with ngram_results
        overlap_vector = [
            1 if ngram in all_ngrams else 0 for ngram in self.ngram_results
        ]
        return overlap_vector

    def prompt_overlap(self, student_answer):

        prompt_tokens = set(word_tokenize(self.prompt))
        answer_tokens = set(word_tokenize(student_answer))

        # Calculate the number of overlapping words
        overlap_count = sum(1 for word in prompt_tokens if word in answer_tokens)

        # Calculate the percentage of overlap between 0 and 1
        overlap_percentage = round((overlap_count / len(prompt_tokens)), 2)

        return [overlap_percentage]

    def weighted_domain_specific_keywords(self, student_answer):
        keywords_dict = self.weighted_keywords

        max_value = max(keywords_dict.values())
        min_value = min(keywords_dict.values())

        if max_value == min_value:
            normalized_keywords = {k: 0 for k in keywords_dict}
        else:
            normalized_keywords = {
                k: (v - min_value) / (max_value - min_value)
                for k, v in keywords_dict.items()
            }

        keyword_counts = {
            keyword: student_answer.count(keyword) for keyword in keywords_dict
        }

        weighted_keywords_multiplied = [
            normalized_keywords[keyword] * keyword_counts[keyword]
            for keyword in sorted(keywords_dict)
        ]

        return weighted_keywords_multiplied

    def lexical_overlap(self, student_answer):

        prompt_tokens = set(word_tokenize(self.prompt))
        answer_tokens = set(word_tokenize(student_answer))

        # Calculate the number of overlapping words
        overlap_count = sum(1 for word in prompt_tokens if word in answer_tokens)

        # Calculate the percentage of overlap between 0 and 1
        overlap_percentage = round((overlap_count / len(prompt_tokens)), 2)

        return [overlap_percentage]

    def stylistic_features(self, student_answer):
        def get_word_difficulty(word):
            # A placeholder function for word difficulty level,
            # In practice, you need a proper mapping of words to their difficulty levels
            difficulty_mapping = {"easy": 1, "medium": 10, "hard": 20}
            return difficulty_mapping.get(word.lower(), 1)

        # Tokenize the text into words and sentences
        words = word_tokenize(student_answer)
        sentences = sent_tokenize(student_answer)

        # Calculate the number of unique words
        unique_words = set(words)

        # Calculate Type-Token Ratio (TTR)
        ttr = len(unique_words) / len(words)

        # Calculate average word length
        average_word_length = sum(len(word) for word in words) / len(words)

        # Calculate average sentence length
        average_sentence_length = sum(
            len(word_tokenize(sentence)) for sentence in sentences
        ) / len(sentences)

        # Count complex words (words with 3 or more syllables)
        complex_words = sum(1 for word in words if textstat.syllable_count(word) >= 3)

        # Count complex sentences (sentences with more than one clause)
        # Here, we'll simply count sentences longer than a certain threshold as complex
        complex_sentences = sum(
            1 for sentence in sentences if len(word_tokenize(sentence)) > 15
        )

        # Get word difficulty levels and their frequencies
        word_difficulty_levels = [get_word_difficulty(word) for word in words]
        difficulty_counter = Counter(word_difficulty_levels)

        # Create a vector for the frequencies of words in each difficulty level
        difficulty_vector = [difficulty_counter[i] for i in range(1, 21)]

        # Create the feature vector
        feature_vector = difficulty_vector + [
            len(unique_words),
            ttr,
            average_word_length,
            average_sentence_length,
            complex_words,
            complex_sentences,
            average_word_length,
            average_sentence_length,
        ]

        return feature_vector

    def logical_operators(self, student_answer):
        # Tokenize the text into words and sentences
        words = word_tokenize(student_answer)
        sentences = sent_tokenize(student_answer)

        # Initialize feature variables
        num_logical_operators = 0
        unique_logical_operators = set()
        complex_sentences = 0
        logical_operators_positions = []
        sentence_lengths = []
        total_words = len(words)
        total_sentences = len(sentences)

        # Analyze each sentence
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            sentence_length = len(sentence_words)
            sentence_lengths.append(sentence_length)

            # Check for logical operators in the sentence
            logical_operators_in_sentence = [
                word for word in sentence_words if word.lower() in LOGICAL_OPERATORS
            ]
            if logical_operators_in_sentence:
                num_logical_operators += len(logical_operators_in_sentence)
                unique_logical_operators.update(logical_operators_in_sentence)
                logical_operators_positions.append(
                    (
                        sentence_words.index(logical_operators_in_sentence[0]),
                        sentence_length,
                    )
                )
                complex_sentences += 1 if len(logical_operators_in_sentence) > 1 else 0

        # Calculate logical operator density
        logical_operator_density = num_logical_operators / total_words

        # Create the feature vector
        feature_vector = [
            num_logical_operators,
            len(unique_logical_operators),
            logical_operator_density,
            complex_sentences,
            total_words,
            total_sentences,
        ]

        return feature_vector

    def temporal_features(self, student_answer):
        doc = nlp(student_answer)
        tense_counter = Counter()
        aspect_counter = Counter()

        for token in doc:
            if token.pos_ == "VERB" or token.pos_ == "AUX":
                tense = (
                    "present"
                    if token.morph.get("Tense") == ["Pres"]
                    else (
                        "past"
                        if token.morph.get("Tense") == ["Past"]
                        else "future" if token.morph.get("Tense") == ["Fut"] else "none"
                    )
                )
                aspect = (
                    "simple"
                    if token.morph.get("Aspect") == []
                    else (
                        "progressive"
                        if token.morph.get("Aspect") == ["Prog"]
                        else (
                            "perfect"
                            if token.morph.get("Aspect") == ["Perf"]
                            else (
                                "perfect_progressive"
                                if token.morph.get("Aspect") == ["Perf", "Prog"]
                                else "none"
                            )
                        )
                    )
                )
                tense_counter[tense] += 1
                aspect_counter[aspect] += 1

        feature_vector = [
            tense_counter["present"],
            tense_counter["past"],
            tense_counter["future"],
            aspect_counter["simple"],
            aspect_counter["progressive"],
            aspect_counter["perfect"],
            aspect_counter["perfect_progressive"],
        ]

        return feature_vector
