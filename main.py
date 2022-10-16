import string


class CountVectorizer:
    """
    кодирование строк при помощи векторов подсчета
    """

    def __init__(self):
        self.feature_names = []
        self.feature_names_set = set()

    def fit_transform(self, corpus):
        """
        принимает на вход список строк и преобразует их в векторы,
         кодируя путем создания вектора всех слов корпуса
        и подсчета количества повторов каждого слова в конкретной строке
        сопоставляет каждой строке - вектор из int-ов
        :param corpus: список слов
        :return: закодированный список слов
        """

        for string_ in corpus:
            words = "".join(
                x for x in string_.lower() if x not in string.punctuation
            ).split()
            for word in words:
                if word not in self.feature_names_set:
                    self.feature_names_set.add(word)
                    self.feature_names.append(word)
        return self.__transform(corpus)

    def __transform(self, corpus):
        """
        на основе имеющегося в классе вектора всех слов кодирует корпус строк
        :param corpus: список слов
        :return: закодированный список слов
        """

        code_corpus = []
        for string_ in corpus:
            code_word = [0] * len(self.feature_names)
            words = "".join(
                x for x in string_.lower() if x not in string.punctuation
            ).split()
            for word in words:
                try:
                    code_word[self.feature_names.index(word)] += 1
                except IndexError:
                    print(
                        "IndexError: transform trying to be applyied \
                     on new corpus"
                    )
                    return ["IndexError"]
            code_corpus.append(code_word)
        return code_corpus

    def get_feature_names(self):
        """
        :return: возвращает копию вектора всех слов по которым строился словарь
        """
        return self.feature_names.copy()


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(count_matrix)
    print(vectorizer.get_feature_names())
