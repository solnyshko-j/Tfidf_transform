# coding=utf-8
import string
from math import log

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class CountVectorizer:
    '''
    кодирование строк при помощи векторов подсчета

    '''

    def __init__(self):
        self.feature_names = []
        self.feature_names_set = set()


    def fit_transform(self, corpus )-> list[list[int]]:
        '''
        принимает на вход список строк и преобразует их в векторы, кодируя  путем создания вектора всех слов корпуса
        и подсчета количества повторов каждого слова в конкретной строке
        сопоставляет каждой строке - вектор из int-ов

        '''

        for string_ in corpus:
            words  = ''.join(x for x in string_.lower() if x not in string.punctuation).split()
            for word in words:
                if word not in self.feature_names_set:
                    self.feature_names_set.add(word)
                    self.feature_names.append(word)
        return self.__transform(corpus)


    def __transform(self, corpus: list[str]) -> list[list[int]]:
        '''
        на основе имеющегося в классе вектора всех слов кодирует корпус строк

        '''

        code_corpus = []
        for string_ in corpus:
            code_word = [0] * len(self.feature_names)
            words = ''.join(x for x in string_.lower() if x not in string.punctuation).split()
            for word in words:
                try:
                    code_word[self.feature_names.index(word)] += 1
                except IndexError:
                    print('IndexError: transform trying to be applyied on new corpus')
                    return ['IndexError']
            code_corpus.append(code_word)
        return code_corpus


    def get_feature_names(self)-> list[list[int]]:
        '''
        возвращает копию вектора всех слов
        '''

        return self.feature_names.copy()




class TfidfTransformer:
    def tf_transform(self, count_matrix):
        count_matrix_tf = []
        for matrix in count_matrix:
            sum_count_matrix = sum(matrix)
            matrix_tf = []
            for x in matrix:
                matrix_tf.append(round(x / sum_count_matrix,2))
            count_matrix_tf.append(matrix_tf)
        return count_matrix_tf

    def idf_transform(self, count_matrix):
        result = []
        all_docs_count = len(count_matrix) + 1
        word_all_docs_count = []
        for word_number in range(len(count_matrix[0])):
            word_count = 1
            for text in count_matrix:
                word_count += (text[word_number] > 0)
            word_all_docs_count.append(word_count)
        for x in word_all_docs_count:
            result.append(round(log(all_docs_count / x) + 1, 2))
        return result

    def fit_transform(self, count_matrix )-> list[list[int]]:
        tf = self.tf_transform(count_matrix)
        idf = self.idf_transform(count_matrix)
        result = []
        for x in tf:
            result.append([round(t*d,2) for (t, d) in list(zip(x, idf))])
        return result


class TfidfVectorizer(CountVectorizer):
    def fit_transform(self, corpus) :
        count_matrix = super().fit_transform(corpus)
        t = TfidfTransformer()
        return t.fit_transform(count_matrix)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    t = TfidfVectorizer()
    print(t.fit_transform(corpus))
    # t = tf_transform(count_matrix)
    # idf = idf_transform(count_matrix)
    # print(idf)
    # count_matrix = [
    #     [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    # ]
    # idf_matrix = idf_transform(count_matrix)
    # print(idf_matrix)
    # assert idf_matrix == [1.4, 1.4, 1.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]
    # print(list(zip(*count_matrix)))
    # print(count_matrix)

