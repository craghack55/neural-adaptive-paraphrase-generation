import nltk
from nltk import word_tokenize
from nltk import ngrams
from tensor2tensor.utils import bleu_hook
from operator import itemgetter
import string 
import numpy as np


def testBLEU():
	translation_corpus = [['how', 'do', 'i', 'learn', 'linux']]
	reference_corpus = [['how', 'do', 'i', 'learn' ,'networking', 'with', 'linux']]
	bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
	print(bleu)


def filterSentence(sentence):

	table = str.maketrans({key: None for key in string.punctuation})
				
	return sentence.translate(table).lower()

def buildIterations(filename_source, filename_target, datasetSize, output_filename):

	blocks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	prev = 0
	for b in blocks:
		limit = int(round(datasetSize * b))
		print(limit)
		i = 0
		s = open(output_filename + str(b) + "_source.txt", "w") 
		t = open(output_filename + str(b) + "_target.txt", "w")

		with open(filename_source) as finput, open(filename_target) as foutput:
			for source, target in zip(finput, foutput):
				if(i > prev):
					s.write(source)
					t.write(target)
				if(i == limit):
					break
				i = i + 1

		prev = limit

		s.close()
		t.close()	

def buildIterationsPool(filename_source, filename_target, datasetSize, output_filename):

	blocks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	for b in blocks:
		limit = int(round(datasetSize * b))
		print(limit)
		i = 0
		s = open(output_filename + str(b) + "_pool_source.txt", "w") 
		t = open(output_filename + str(b) + "_pool_target.txt", "w")

		with open(filename_source) as finput, open(filename_target) as foutput:
			for source, target in zip(finput, foutput):
				s.write(source)
				t.write(target)
				if(i == limit):
					break
				i = i + 1

		s.close()
		t.close()	

def buildCorpus(filename, output):

	source = open(output + "_source.txt", "w") 
	target = open(output + "_target.txt", "w") 

	with open(filename) as finput:
		for l in finput:
			t = l.strip().split("\t")
			if(t[0] == "1"):
				source.write(filterSentence(t[3]) + '\n')
				target.write(filterSentence(t[4]) + '\n')

	source.close()
	target.close()

def removeExtraWhite(source_file):
	s = open("at.txt", "w")
	with open(source_file) as finput:
		for l in finput:
			t = l.strip().split(" ")
			t = [x for x in t if x != '']
			t = ' '.join(t)
			s.write(t + '\n')
	s.close()


def createNGramSampling(source_file, target_file):
	sourceSentences = []
	targetSentences = []

	with open(source_file) as finput, open(target_file) as foutput:
		for source, target in zip(finput, foutput):	
			sourceSentences.append(source.rstrip())
			targetSentences.append(target.rstrip())

	sourceSentences = list(sourceSentences)
	targetSentences = list(targetSentences)

	n = 5 
	A = 5
	n_grams = {}
	sentencesAndScores = []

	for s in sourceSentences:
		l = s.split()
		for i in range(1 , n):
			a = ngrams(l, i)
			for g in a:
				if g in n_grams:
					n_grams[g] += 1
				else:
					n_grams[g] = 1

	for j in range(0, len(sourceSentences)):
		s = sourceSentences[j]
		l = s.split()
		ng = []
		total = 0
		for i in range(1, n):
			a = list(ngrams(l, i))
			total += len(a)
			for g in a:
				if(n_grams[g] >= A):
					ng.append(g)

		score = len(ng) / total
		sentencesAndScores.append([sourceSentences[j], targetSentences[j], score])

	sentencesAndScores = sorted(sentencesAndScores, key=itemgetter(2))

	source = open("r_ngram_train_source.txt", "w") 
	target = open("r_ngram_train_target.txt", "w")

	for i in sentencesAndScores:
		source.write(i[0] + '\n')
		target.write(i[1] + '\n')

	source.close()
	target.close()


def buildVocabulary(source_files, output_filename):

	# with open(source_files) as file:
	# 	for index, line in enumerate(file):
	# 		values = line.split() # Word and weights separated by space
	# 		word = values[0] # Word is first symbol on each line
	# 		word_weights = np.asarray(values[1:], dtype=np.float32)
	# 		print(word_weights)

	t = []
	for source_file in source_files:
		print(source_file)
		source = open(source_file).read()
		tokens = word_tokenize(source)
		for i in range (0, len(tokens)):
			tokens[i] = tokens[i].lower()
			t.append(tokens[i])

	set_tokes = set(t)
	tokens = list(set_tokes)
	file = open(output_filename,"w") 
	for i in tokens:
		file.write(i + '\n')

	file.close()

if __name__ == "__main__":
	#createNGramSampling("data/quora/train_source.txt", "data/quora/train_target.txt")
	buildIterationsPool("data/quora/r_ngram_train_source.txt", "data/quora/r_ngram_train_target.txt", 119445, "data/quora/")
	# nltk.download('punkt')
	# source_files = "data/glove.6B/glove.6B.50d.txt"
	# source_files = ["data/mscoco/train_source.txt", "data/quora/train_source.txt", "data/mscoco/train_target.txt", "data/quora/train_target.txt"]
	# buildVocabulary(source_files, "v.txt")
	# removeExtraWhite("data/mscoco/test_source.txt")
	# testBLEU()
	# buildCorpus("data/msr/test.txt", "data/msr/test")