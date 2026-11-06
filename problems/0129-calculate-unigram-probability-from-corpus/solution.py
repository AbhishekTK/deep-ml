from collections import Counter
def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    # pass
    a = corpus.split()
    c = Counter(a)
    # print(c)

    return c[word]/len(a)
