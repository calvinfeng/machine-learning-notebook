
class FeatureProbabilitySelector(object):
    """A helper class to select the best ham/spam features
    """
    @classmethod
    def best_spam_features(cls, fps, limit=20, present_features=True):
        return cls.best_features(fps, limit, -1, present_features=present_features)

    @classmethod
    def best_ham_features(cls, fps, limit=20, present_features=True):
        return cls.best_features(fps, limit, 1, present_features=present_features)

    @classmethod
    def best_features(cls, fps, limit, multiplier, present_features):
        if present_features:
            prob_ratio_fn = fps.code_prob_ratio
            code_count = lambda code: fps.code_count[code]
        else:
            prob_ratio_fn = fps.no_code_prob_ratio
            code_count = fps.no_code_count

        codes = list(fps.code_count.keys())
        features = [{
            'code': code,
            'count': code_count(code),
            'feature_probability_ratio': (prob_ratio_fn(code))
        } for code in codes]

        features.sort(key=lambda feature: multiplier * feature['feature_probability_ratio'])

        return features[:limit]

    @classmethod
    def print_feature_list(cls, feature_list, word_encoding_dictionary):
        for feature in feature_list:
            code, reach, fp_ratio = (feature['code'], feature['count'], feature['feature_probability_ratio'])
            word = word_encoding_dictionary.code_to_word(code)
            print "%s | %s | %s | %s" % (code, word, reach, fp_ratio)
