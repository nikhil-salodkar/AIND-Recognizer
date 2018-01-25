import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_n = self.min_n_components
        min = float("+inf")
        #calculate the BIC for each number of n_components..
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=i, covariance_type="diag",n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X,self.lengths)
                log_likelyhood = model.score(self.X, self.lengths)
                #now calculating the bic.. BIC=-2*LOGL + plogN
                p = i*i + 2*i*len(self.X[0]) - 1
                BIC = -2*log_likelyhood + p*(math.log(len(self.X[0])))
                #keeping track of the lowest value of BIC which would be the best for model..
                if BIC < min:
                    min = BIC
                    best_n = i
            except:
                pass
        #return the best model..
        return GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X,self.lengths)


class SelectorDIC(ModelSelector):

    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #In this model selector for each n component first find its model and then compare the same
        #with average of all other words which calculates DIC. The model having highest DIC value is best.
        best_model = None
        best_score = float("-inf")
        #traverse for each n_components..
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                # find the model..
                new_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                likelyhood = new_model.score(self.X, self.lengths)
                #now compare this likelyhood with all likelyhoods of other words..
                sum_score = 0; avg_score = 0; count = 0
                for other_word, sets in self.hwords.items():
                    try:
                        if other_word != self.this_word:
                            data, length = sets
                            new_score = new_model.score(data, length)
                            sum_score += new_score
                            count += 1
                    except:
                        pass
                avg_score = sum_score/count
                #now calculate DIC..
                DIC = likelyhood - avg_score
                #and keep updating the best dic score
                if DIC > best_score:
                    best_score = DIC
                    best_model = new_model
            except:
                pass
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        split_method = KFold()
        best_n = self.min_n_components
        best_score = float("-inf")
        #Iterate for all possible number of states...
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                count = 0
                total = 0
                #for each combination of folds which result due to split method, get the train and test samples..
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    #now subsets must be combined based on indices given for the folds..
                    train_set, train_length = combine_sequences(cv_train_idx, self.sequences)
                    test_set, test_length = combine_sequences(cv_test_idx, self.sequences)
                    #now create a model using the training samples selected just now..
                    new_model = GaussianHMM(i, n_iter=1000).fit(train_set, train_length)
                    #now calculate the score and test how well this newly created model is performing..
                    new_score = new_model.score(test_set, test_length)
                    total = total + new_score
                    count += 1
                avg_score = total/count
                #this average score corresponds to the performance of the model using i number of n_components..
                if(avg_score > best_score):
                    best_score = avg_score
                    best_n = i
            except:
                pass
        return GaussianHMM(best_n, n_iter=1000).fit(self.X, self.lengths)




