from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
SOS_token = 0
EOS_token = 1
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
    def addSentence(self, sentence):
        for word in sentence.split():
            if word == '':
                print('****************',sentence)
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z0-9?&\%\-]+", r" ", s)
    return s
s="Il eût été bien de s'inscrire en 2017 !! N'est-ce pas ?"
s = normalizeString(s)
print(s)
def readLangs(questions, answers, reverse=False):
    print("Reading lines...")
    lines = open('data/chatbot-M2-DS.txt', encoding='utf-8').        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(answers)
        output_lang = Lang(questions)
    else:
        input_lang = Lang(questions)
        output_lang = Lang(answers)
    return input_lang, output_lang, pairs
MAX_LENGTH = 12
stopwords = []
def TrimWordsSentence(sentence):
    resultwords = [word for word in sentence.split() if word.lower() not in stopwords]
    resultwords = ' '.join(resultwords)
    return resultwords
def TrimWords(pairs):
    for pair in pairs:
        pair[0] = TrimWordsSentence(pair[0])
    return pairs
def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and         len(p[1].split()) < MAX_LENGTH
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
s="Il eût été bien de s'inscrire en 2017 !! N'est-ce pas ?"
print(s)
s = normalizeString(s)
print(s)
s = TrimWordsSentence(s)
print(s)
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = TrimWords(pairs)
    for pair in [pair for pair in pairs if not filterPair(pair)]:
        print('%s (%d) -> %s (%d)' % (pair[0],len(pair[0].split()),pair[1],len(pair[1].split())))
    pairs = filterPairs(pairs)
    print('')
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        if '' in output_lang.word2index: print(pair[1].split())
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
input_lang, output_lang, pairs = prepareData('questions', 'answers', False)
print(random.choice(pairs))
print(set(input_lang.word2index))
print(set(output_lang.word2index))
pairs
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]),dim=1)
        return output, hidden
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        return output, hidden, attn_weights
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split()]
def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result
def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)
teacher_forcing_ratio = 0.5
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input
            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / target_length
import time
import math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.05):
    start = time.time()
    plot_losses = []
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    decoder_input = decoder_input
    decoder_hidden = encoder_hidden
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))
    return decoded_words, decoder_attentions[:di + 1]
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1)
trainIters(encoder1, attn_decoder1, 1000, print_every=1000)
evaluateRandomly(encoder1, attn_decoder1)
mail="Bonjour, je suis intéressé par la data science. C'est quand la rentrée ? Il y a quels cours ?"
sentences = [s.strip() for s in re.split('[\.\,\?\!]' , mail)]
print(sentences[:-1])
sentences = sentences[:-1]
    sentence2= TrimWordsSentence(normalizeString(sentence))
    print(sentence2)
    output_words, attentions = evaluate(encoder1, attn_decoder1, sentence2)
    output_sentence = ' '.join(output_words)
    print(sentence)
    print('=', sentence2)
    print('->', output_sentence)
    print('')
def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    showAttention(input_sentence, output_words, attentions)
evaluateAndShowAttention(TrimWordsSentence("quelle est la date de la rentree 2018 ?"))
evaluateAndShowAttention(TrimWordsSentence("a quel metier prepare la formation ?"))
def indexesFromSentence(lang, sentence):
    indexes=[]
    for word in sentence.split(' '):
        if word in set(lang.word2index):
            indexes.append(lang.word2index[word])
        else:
            print("Unknwon word : ",word)
    while len(indexes) >= MAX_LENGTH:
        indexes.pop(random.randint(1,len(indexes)-1))
    return indexes
def SentenceFromIndexes(lang, indexes):
    sentence = []
    for i in indexes:
        sentence.append(input_lang.index2word[i])
    sentence = ' '.join(sentence)
    return sentence
new_sentence = "pourriez vous me dire quels sont les enseignants ou les chercheurs du laboratoire qui vont encadrer mon stage de fin d'étude s'il vous plaît ?"
indexes = indexesFromSentence(input_lang, TrimWordsSentence(normalizeString(new_sentence)))
sentence2 = SentenceFromIndexes(input_lang, indexes)
print(new_sentence)
print('->', sentence2)
evaluateAndShowAttention(sentence2)
from gensim.models import KeyedVectors
filename = 'data/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
result = model.most_similar(positive=['femme', 'roi'], negative=['homme'], topn=1)
print("La femme est à l'homme, ce que la", result[0][0], 'est au roi')
result = model.most_similar(positive=['femme', 'voiture'], negative=['homme'], topn=1)
print("La voiture est à l'homme, ce que la", result[0][0], 'est à la femme')
result = model.most_similar(positive=['stupidité', 'vulgarité'], negative=['intelligence'], topn=1)
print("La stupidité est à l'intelligence, ce que la", result[0][0], 'est à la beauté')
result = model.most_similar(positive=['ennui', 'femme'], negative=['passion'], topn=1)
print("La passion est à l'ennui, ce que la", result[0][0], 'est à la femme')
result = model.most_similar(positive=['travailleur', 'salarié'], negative=['employeur'], topn=1)
print("La salarié est à l'employeur, ce que la", result[0][0], 'est au travailleur')
result = model.most_similar(positive=['innovation', 'analogique'], negative=['entreprise'], topn=1)
print("L'innovation est à l'entreprise, ce que le", result[0][0], "est à l'analogique")
result = model.most_similar(positive=['commerce', 'finance'], negative=['ingénierie'], topn=1)
print("Le commerce est à l'ingénierie, ce que le", result[0][0], "est à la finance")
result = model.most_similar(positive=['littérature', 'femme'], negative=['homme'], topn=1)
print("Le littérature est à l'homme, ce que le", result[0][0], "est à la femme")
result = model.most_similar(positive=['homme'], topn=2)
print("Un homme s'intéresse à la", result[1][0])
result = model.most_similar(positive=['femme'], topn=2)
print("Une femme s'intéresse à son", result[1][0])
result = model.most_similar(positive=['intelligence'], topn=1)
print("L'intelligence est", result[0][0])
result = model.doesnt_match("manger boire déjeuner travailler".split())
print(result)
result = model.similarity('informatique', 'progrès')
print(result)
result = model.similarity('informatique', 'innovation')
print(result)
words = list(model.vocab)
print(set(input_lang.word2index).intersection(words))
l1 = len(list(set(input_lang.word2index).intersection(words)))
print('Percentage of common words w.r.t. input_lang : %0.2f%%' % (100 * l1/input_lang.n_words))
print('Percentage of common words w.r.t. word2vec vocab : %0.2f%%' % (100 * l1/len(words)))
print(words[11000:11200])
def indexesFromSentence(lang, sentence):
    indexes=[]
    for word in sentence.split():
        if word in set(lang.word2index):
            indexes.append(lang.word2index[word])
        elif word in model.vocab:
            new_word = model.most_similar(positive=[word], topn=1)[0][0]
            if new_word in set(lang.word2index):
                indexes.append(lang.word2index[new_word])
                print(word,'->',new_word)
        else:
            print("Unknwon word in model vocab: ",word)
    while len(indexes) >= MAX_LENGTH:
        indexes.pop(random.randint(1,len(indexes)-2))
    return indexes
new_sentence = "Je suis vif, auriez-vous l'amabiliité de me dire quels sont les enseignants ou les chercheurs du laboratoire qui vont encadrer mon stage de fin d'étude cette année et qui est le responsable du master s'il vous plaît ?"
trimmed_sentence = TrimWordsSentence(normalizeString(new_sentence))
indexes = indexesFromSentence(input_lang, trimmed_sentence)
print(indexes)
print(new_sentence)
print('->', trimmed_sentence)
print('->', SentenceFromIndexes(input_lang, indexes))
embedding_weights = np.zeros((n_symbols, vocab_dim))
for word,index in input_lang.word2index.items():
    try:
        embedding_weights[index, :] = model.get_vector(word)
    except KeyError:
        pass
print(embedding_weights.shape)
print(len(model.vocab))
hidden_size = vocab_dim
class EncoderRNN_with_trained_embedding(nn.Module):
    def __init__(self, input_size, n_symbols, pretrained_embed, n_layers=1):
        super(EncoderRNN_with_trained_embedding, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        pretrained_embed = torch.Tensor(pretrained_embed)
        self.embedding.weight = nn.Parameter(pretrained_embed)
        self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
class AttnDecoderRNN_with_trained_embedding(nn.Module):
    def __init__(self, hidden_size, output_size, pretrained_embed, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_with_trained_embedding, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        pretrained_embed = torch.Tensor(pretrained_embed)
        self.embedding.weight = nn.Parameter(pretrained_embed)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
encoder1 = EncoderRNN_with_trained_embedding(input_lang.n_words, hidden_size, embedding_weights)
attn_decoder1 = AttnDecoderRNN_with_trained_embedding(hidden_size, output_lang.n_words,
                                                      embedding_weights,1, dropout_p=0.1)
trainIters(encoder1, attn_decoder1, 5000, print_every=100)
evaluateRandomly(encoder1, attn_decoder1)
new_sentence = "Bonjour, je suis en intérim, je vis sur un promontoire, je porte un crucifix autour du cou, suis-je en mesure de faire un stage de fin d'étude sans souci ?"
trimmed_sentence = TrimWordsSentence(normalizeString(new_sentence))
indexes = indexesFromSentence(input_lang, trimmed_sentence)
print(indexes)
print(new_sentence)
print('->', trimmed_sentence)
final_sentence = SentenceFromIndexes(input_lang, indexes)
print('->', final_sentence)
evaluateAndShowAttention(final_sentence)
import spacy
nlp_fr = spacy.load('fr_core_news_md')
mail = "Bonjour Monsieur Aussem, je suis titulaire d'un Master 2 mention informatique et actuellement doctorante en deuxième année spécialité Bioinformatique. Je souhaite savoir si mes deux ans de doctorat pourront constituer un obstacle lors de ma demande de candidature en Master 2 mention informatique parcours Data Science au sein de votre université. Cordialement."
text_fr = nlp_fr(unidecode(mail))
for token in text_fr:
    print(token.text, token.lemma_, '\t', token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
token1 = nlp_fr(unidecode('bonjour'))
token2 = nlp_fr(unidecode('monsieur'))
print(token1.similarity(token2))
print(token1.vector)
text = []
stop_tags = {'PUNC','ADP_','DET_'}
for w in text_fr:
    if w.tag_[0:4] not in stop_tags:
        print("%s\t%s\t%s" % (w, w.tag_[0:4], w.dep_))
        text.append(w.text)
text2 = ' '.join(text)
print(text2)
doc_fr = text_fr
for sent in doc_fr.sents:
    print(sent)
words = [w.text for w in doc_fr]
print(words)
import unidecode
from unidecode import unidecode
unidecode("Il eût été bien de s'inscrire en 2017 !! N'est-ce pas ?")
import nltk
tokens = nltk.word_tokenize(unidecode(text_fr))
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
entities
import gensim
message = gensim.utils.to_unicode(text_fr, 'latin1').strip()
print(message)
message = list(gensim.utils.tokenize(message, lower=True))
print(message)
text = ' '.join(message)
print(text)
