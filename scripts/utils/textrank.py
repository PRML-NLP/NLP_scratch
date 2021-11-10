from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np
from konlpy.tag import Komoran

def dict_to_mat(d, n_rows, n_cols):
    """
    convert graph to sparse matrix
    """
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

def scan_vocab(sents, tokenizer, min_count=2):
    """
    build vocabulary
        sents: sentences in documents
        tokenizer: str -> list of str
        min_count: minimun number of occurrence in given documents (recommend 2~8)
    """
    counter = Counter(w for sent in sents for w in tokenizer(sent))
    occur = {}
    for w, c in counter.items():
        if c >= min_count:
            occur[w] = c
            
    idx_to_vocab = [w for w, _ in sorted(occur.items(), key=lambda x: -x[1])]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def cooccurrence(tokenized_sents, vocab_to_idx, window=2, min_coocc=2):
    """
    Co-occurrence between words
        tokenized_sents: list of tokenized sentences
        vocab_to_idx: {word: index} dictionary
    """
    counter = defaultdict(int)
    for sent in tokenized_sents:
        # Token to id
        vocab_idx = []
        for w in sent:
            if w in vocab_to_idx:
                vocab_idx.append(vocab_to_idx[w])
        for i, idx in enumerate(vocab_idx):
            if window <= 0:
                b_idx, e_idx = 0, len(vocab_idx)
            else:
                b_idx, e_idx = max(0, i-window), min(i+window, len(vocab_idx))
            # current word: idx, co-occurrence word in window: vocab_idx[j]
            for j in range(b_idx, e_idx):
                if i==j:
                    continue
                counter[(idx, vocab_idx[j])] += 1
                counter[(vocab_idx[j], idx)] += 1
    coocc = {}
    for k, v in counter.items():
        if v >= min_coocc:
            coocc[k] = v
    n_vocabs = len(vocab_to_idx)

    return dict_to_mat(coocc, n_vocabs, n_vocabs)

def word_graph(sents, tokenizer, min_count=2, window=2, min_coocc=2):
    idx_to_vocab, vocab_to_idx = scan_vocab(sents, tokenizer, min_count)
    tokens = [tokenizer(sent) for sent in sents]
    print('Constructing graph...')
    graph = cooccurrence(tokens, vocab_to_idx, window, min_coocc)
    print('Done!')
    return graph, idx_to_vocab

def pagerank(x, df=0.85, max_iter=30):
    """
    Page Rank
        x: co-occurrence graph
    """
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1,1)
    bias = (1-df) * np.ones(A.shape[0]).reshape(-1,1)

    # iteration
    # A*R: Value to be transferred with R_j from column j to row i
    for _ in range(max_iter):
        R = df * (A * R) + bias
    
    return R

def textrank_keyword(sents, min_count, window, min_coocc, tokenizer=None, df=0.85, max_iter=30, topk=30):
    if tokenizer is None:
        tokenizer = lambda x: x.split(' ')
    graph, idx_to_vocab = word_graph(sents, tokenizer, min_count, window, min_coocc)
    R = pagerank(graph, df, max_iter).reshape(-1)
    idxes = R.argsort()[-topk:]
    keywords = [(idx_to_vocab[idx], R[idx]) for idx in reversed(idxes)]
    return keywords

if __name__=='__main__':
    document = '간혹 인공지능과 로봇을 혼동, 혼용하는 사람들도 있는데 따로 두고 생각해야 한다. 인공지능이라는 분야가 대중들과 가까워진 시기가 상당히 최근의 일이고 미디어에서 보여주는 로봇들이 대부분 당연한듯이 자칭 인공지능 기술을 탑재하고 나오는 경우가 많지만 이 둘은 애초부터 추구하는 목적 자체가 다른 분야다. 인공지능은 어떤 정보를 받아서 해석하여 결과를 출력하는 등등 정보처리 차원의 문제다. 어떤 입력을 어떻게 처리해서 주었을 때 어떤 결과가 나오는지, 어떤 의미를 가지는지, 얼마나 정확한지, 얼마나 우수한지 등등의 사안이 가장 중요한 문제다. 만일 어떤 대상을 예측하기 위한 인공지능이라면 그 대상을 얼마나 높은 정확도로 예측하는지가 중요할 것이며, 주어진 정보를 토대로 지지대나 철봉 따위의 기구를 자동 설계하는 인공지능이라면 어느정도까지 최소의 비용으로 최대의 내구력을 얻을 수 있을지가 중요할 것이다. 이렇게 인공지능은 정보로 시작해서 정보로 끝나는 분야이므로 로봇 등의 기계적인 요소와는 완전히 무관하다. 반면에 로봇공학은 전형적으로 기계적인 부분에 초점을 맞춰서 구동기를 뭘 쓸지, 로봇의 신체를 용도를 고려해 어떤 식으로 만들지, 어떤 부위의 구동기를 어떻게 제어해서 어떤식으로 물리적으로 실존하는 기계시스템을 빠르고 정확하게 운영할 수 있는지, 그래서 어떤 기계적인 성과를 거둘지가 중요한 문제다. 예를 들어 팔에 모터를 박아놓은 인간형 로봇이라면 그 모터와 모터를 움직이기 위한 동력을 효율적으로 제어하여 가능한한 정확하고 빠르게 목표한 각도로 팔을 움직여주어야 할 것이다. 사람이야 오랜기간 진화를 통해서 별 생각 없이도 손쉽게 팔을 움직일 수 있지만 어떤 기계를 물리적으로 원하는대로 움직이는 것은 여러분이 상상하는 것 이상으로 상당한 난이도가 있는 공학 기술이며, 특히나 우리가 흔히 아는 관절이 여러개이거나 구조가 복잡한 로봇들은 신속하고 정확하게 원하는 대로 움직이는 것 자체가 매우 고도의 기술이다. 왜냐하면 전기나 유압 등의 눈에 보이지도 않는 에너지를 적재적소에 정확한 양으로 공급하거나 빼야 되는데다가 기계를 움직이는 경로에 사소한 장애물이 있을 수도 있고 그 기계가 물건을 들고 있거나 내려놓는 등의 상황에는 같은 동작을 하더라도 필요한 힘이 수시로 바뀌기 때문이다. 저런 오만가지 변수를 싸그리 예측하고 무마해서 기계를 원하는대로 칼 같이 동작을 시키는 것과 할 수 있게 만드는 것은 결코 만만한 일이 아니다. 특히 로봇공학처럼 서보제어 영역으로 들어가게 되면 응답속도도 매우 중요해지는데 이 응답속도가 밀리초~마이크로초 영역이라 뭘 판단하고 다음 행동을 결정할 시간조차도 없다. 사람으로 예를 들면 발에 돌이 걸려 넘어질 때 자기도 모르게 팔을 땅에 짚게 되는데 이런 동작을 일일이 생각을 거쳐서 하려들면 뭘 해보기도 전에 땅에 코를 박게 될 것이다. 보스턴 다이나믹스의 제품들이 매우 좋은 예시인데 무척 복잡하고 정교한 움직임과 자세를 보여줌에도 불구하고 이들의 움직임은 모두 고도의 제어공학과 각종 센서, 알고리즘을 인간이 응용하고 설계하여 만든 것이며, 조종도 전부 인간이 한다. 홍보 영상에 등장하는 모든 로봇들의 움직임에 인공지능이 관여하는 부분은 없다. 인공지능이 없는데 저런 움직임이 가능하냐고 물을 수도 있는데 인간형 로봇을 예를 들어 생각해보면 어차피 걷는 방법 자체는 다리를 뻗어서 발로 땅을 짚고 다시 반대쪽 다리를 뻗어서 다시 땅을 짚는 행위의 반복이다. 여기서 무게중심이나 전방의 장애물, 땅의 형태 따위를 센서로 읽어보고 명령을 받아 하고자 하는 동작 등을 고려해서 다리를 뻗거나 발을 짚는 위치를 계산해서 조정하는 식으로 진행하게 된다. 현대에 있는 온갖 자동화기기들을 보면 알겠지만 이런 식으로 팔다리를 이리저리 움직이는 행위가 정보처리와 계산 만으로 해결할 수 있는, 분명히 지능이 필수가 아닌 자동제어의 영역임을 이해할 수 있을 것이다. 여러분이 팔다리를 움직여서 무엇을 어떻게 할지를 먼저 생각하지, 팔다리를 움직이는 방법 자체를 고민하지는 않듯이 말이다. 정리를 하자면 복잡한 정보와 대량의 데이터를 토대로 새로운 정보를 창출하는 일과 어떤 복잡한 기계 체계를 컴퓨터를 이용해 고도의 자동화 알고리즘으로 가동시키는 것은 전혀 다른 문제라는 것이다. 몸을 원하는대로 움직이는 것은 지능을 필요로 하는 문제라기보다는 기계적인 문제에 가깝기 때문에 사람에게도 매우 힘든 일이다. 여러분이 팔과 다리를 유연하게 움직일 수 있는 것도 팔 다리에 있는 수십 수백개의 크고작은 근육들이 정교하게 협동함으로써 가능한 것인데 이렇게 로봇 따위는 갖다 댈 수도 없을 만큼 복잡한 신체를 별달리 의식하지 않고 쉽게 움직일 수 있는 이유는 이런 신체제어만 전담해서 자동으로 처리하는 소뇌가 있기 때문이다. 사람의 뇌조차도 대뇌가 감당하기 힘들어서[4] 신체제어를 전문적으로 하는 부위가 따로 있는데 하물며 기계는 어떻겠는가? 이제 왜 두 분야가 별도로 분리가 되어 있는지 이해가 될 것이다. 즉 알파고처럼 컴퓨터 안에서만 돌아가는 인공지능도 존재하고 단순 알고리즘과 제어 프로그램에 의해 움직이는 협업로봇이 존재하듯이 이들은 서로 긴밀하게 묶여있는 분야가 아니고 상호보완의 관계다. 현실에서는 창작물처럼 인공지능이 무안단물마냥 모든 문제를 해결해 줄 수 없으며, 분야마다 제각기 강점과 약점이 있기 마련이다.'
    document = document.split('. ')
    komoran = Komoran()
    def komoran_tokenizer(sent):
        words = komoran.pos(sent, join=True)
        words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
        return words

    print(textrank_keyword(document, min_count=5, window=7, min_coocc=3, tokenizer=komoran_tokenizer))