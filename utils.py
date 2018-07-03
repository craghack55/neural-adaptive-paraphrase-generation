import nltk
from nltk import word_tokenize
from tensor2tensor.utils import bleu_hook
import string 


def testBLEU():
	translation_corpus = [['how', 'do', 'i', 'learn', 'linux']]
	reference_corpus = [['how', 'do', 'i', 'learn' ,'networking', 'with', 'linux']]
	bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
	print(bleu)


def filterSentence(sentence):

	table = str.maketrans({key: None for key in string.punctuation})
				
	return sentence.translate(table).lower()

def buildIterations(filename_source, filename_target, datasetSize, output_filename):

	blocks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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

	blocks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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


def buildVocabulary(source_files, output_filename):
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
	# buildIterationsPool("data/msr/train_source.txt", "data/msr/train_target.txt", 2753, "data/msr/")
	# nltk.download('punkt')
	source_files = ["data/mscoco/train_source.txt"]
	buildVocabulary(source_files, "v.txt")
	# testBLEU()
	# buildCorpus("data/msr/test.txt", "data/msr/test")