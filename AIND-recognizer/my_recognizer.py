import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    test_cases = list(test_set.get_all_Xlengths().values())
    for X, X_length in test_cases:
        temp_dict = {}
        max = float("-inf")
        most_likely_word = "not defined"
        #iterate through all the models and find the best model that suits the training set..
        for current_word, current_model in models.items():
            try:
                curr_prob = current_model.score(X, X_length)
                # now fill up the probabilities dictionary accordingly..
                temp_dict[current_word] = curr_prob
                if(curr_prob > max):
                    max = curr_prob
                    most_likely_word = current_word
            except:
                pass
        probabilities.append(temp_dict)
        #now that we have got our most probabilistic word, update the guesses list..
        guesses.append(most_likely_word)
    # return probabilities, guesses
    return probabilities, guesses

