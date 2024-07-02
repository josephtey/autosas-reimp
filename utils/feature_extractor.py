from gensim.utils import simple_preprocess
from nltk import ngrams, pos_tag
from nltk.tokenize import word_tokenize


class FeatureExtractor:
    def __init__(self, word2vec_model, doc2vec_model, ngram_results, prompts):
        self.word2vec_model = word2vec_model
        self.doc2vec_model = doc2vec_model
        self.ngram_results = ngram_results
        self.prompts = prompts

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

    def pos(self, setnumber, student_answer):
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
            1 if ngram in all_ngrams else 0 for ngram in self.ngram_results[setnumber]
        ]
        return overlap_vector

    def prompt_overlap(self, setnumber, student_answer):
        prompt = self.prompts[setnumber]

        prompt_tokens = set(word_tokenize(prompt))
        answer_tokens = set(word_tokenize(student_answer))

        # Calculate the number of overlapping words
        overlap_count = sum(1 for word in prompt_tokens if word in answer_tokens)

        # Calculate the percentage of overlap between 0 and 1
        overlap_percentage = round((overlap_count / len(prompt_tokens)), 2)

        return [overlap_percentage]
