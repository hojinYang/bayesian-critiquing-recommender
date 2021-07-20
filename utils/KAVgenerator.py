from sklearn import linear_model
import numpy as np
from tqdm import tqdm


class KAVgenerator:
    def __init__(self, m_item_keyphrase, m_item_activations, kp_threshold):
        self.m_item_keyphrase = m_item_keyphrase.copy()
        if kp_threshold < 1:
            self.m_item_keyphrase = self.m_item_keyphrase/(self.m_item_keyphrase.sum(axis=1)+1e-10) 
        self.m_item_keyphrase[self.m_item_keyphrase < kp_threshold] = 0
        self.valid_items = np.unique(self.m_item_keyphrase.nonzero()[0])

        self.m_item_activations = m_item_activations
        self.num_keyphrase = self.m_item_keyphrase.shape[1]
        self.dim_activate = self.m_item_activations.shape[1]


    def get_keyphrase_examples(self, keyphrase_id):
        return self.m_item_keyphrase[:,keyphrase_id].nonzero()[0]

    def _get_kav(self, keyphrase_id, num_neg=1,num_vec=100):
        keyphrase_examples = self.get_keyphrase_examples(keyphrase_id)
        if len(keyphrase_examples) == 0:
            #activation vectors are all zero vector
            return np.zeros(shape=(num_vec, self.dim_activate))

        vectors = []
        norms = []

        for _ in range(num_vec):
            positive_samples = np.random.choice(keyphrase_examples, num_neg)
            negative_samples = np.random.choice(np.setdiff1d(self.valid_items,keyphrase_examples),num_neg)

            v_positive_samples = self.m_item_activations[positive_samples]
            v_negative_samples = self.m_item_activations[negative_samples]
            
            X = np.vstack((v_positive_samples,v_negative_samples))
            Y = [1]*len(positive_samples) + [0]*len(negative_samples)
            lm = linear_model.LogisticRegression()
            #lm = linear_model.SGDClassifier()
            lm.fit(X, Y)
            vectors.append(lm.coef_[0])
            norms.append(np.mean(np.linalg.norm(X,ord=2, axis=1, keepdims=True)))
        return self.normalize_rows(np.vstack(vectors))

    def get_all_kav(self, num_negatives,num_kav):
        ret = []
        print("Generate Keyphrase Activation Vector")
        for k in tqdm(range(self.num_keyphrase)):
            kavs = self._get_kav(k, num_negatives, num_kav)
            ret.append(kavs)
        #kp by sample by dim
        return np.stack(ret,axis=0)

    def get_all_mean_kav(self, num_negatives,num_kav):
        all_kav = self.get_all_kav(num_negatives,num_kav)
        print(all_kav.shape)

        return np.mean(all_kav, axis=1)

    def normalize_rows(self, x):
        #return x
        """
        function that normalizes each row of the matrix x to have unit length.
        Args:
        ``x``: A numpy matrix of shape (n, m)
        Returns:
        ``x``: The normalized (by row) numpy matrix.
        """
        return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)