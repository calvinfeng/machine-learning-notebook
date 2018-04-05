class NaiveBayesModel(object):
    def __init__(self, fps):
        self.feature_probability_set = fps

    def score_email(self, email):
        fps = self.feature_probability_set

        # General probability of a given email is spam
        prob_spam = float(fps.class_count.spam_count) / fps.class_count.ham_count

        product = prob_spam
        for code in fps.code_count:
            if code in email.codes:
                product *= fps.code_prob_ratio(code)
            else:
                product *= fps.no_code_prob_ratio(code)

        return product

    def email_scores(self, emails):
        return list(map(self.score_email, emails))


def classification_accuracy(score_cutoff, ham_scores, spam_scores):
    num_false_positives = len(list(filter(lambda score: score > score_cutoff, ham_scores)))
    num_true_positives = len(list(filter(lambda score: score > score_cutoff, spam_scores)))

    false_positive_rate = float(num_false_positives) / len(ham_scores)
    true_positive_rate = float(num_true_positives) / len(spam_scores)

    return {
        'true_positive_rate': true_positive_rate,
        'false_positive_rate': false_positive_rate
    }
