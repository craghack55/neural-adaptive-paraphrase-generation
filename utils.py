import nltk
from nltk import word_tokenize

def buildVocabulary(source_file):
	source = open(source_file).read()

	tokens = word_tokenize(source.decode('utf-8'))
	for i in range (0, len(tokens)):
		tokens[i] = tokens[i].lower()

	set_tokes = set(tokens)
	tokens = list(set_tokes)
	file = open("train_vocab.txt","w") 
	for i in tokens:
		file.write(i.encode('utf-8') + '\n')

	file.close()

if __name__ == "__main__":
	buildVocabulary("data/quora/all.txt")