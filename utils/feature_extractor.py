from gensim.utils import simple_preprocess


class FeatureExtractor:
    def __init__(self, word2vec_model, doc2vec_model):
        self.word2vec_model = word2vec_model
        self.doc2vec_model = doc2vec_model

    def word2vec(self, student_answer):
        words = student_answer.split()
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
