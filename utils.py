import nltk
from nltk import word_tokenize
from tensor2tensor.utils import bleu_hook

def testBLEU():
	translation_corpus = [['how', 'do', 'i', 'learn', 'linux']]
	reference_corpus = [['how', 'do', 'i', 'learn' ,'networking', 'with', 'linux']]
	bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
	print(bleu)


def buildVocabulary(source_files, output_filename):
	for source_file in source_files:
		source = open(source_file).read()

		tokens = word_tokenize(source.decode('utf-8'))
		for i in range (0, len(tokens)):
			tokens[i] = tokens[i].lower()

	set_tokes = set(tokens)
	tokens = list(set_tokes)
	file = open(output_filename,"w") 
	for i in tokens:
		file.write(i.encode('utf-8') + '\n')

	file.close()

if __name__ == "__main__":
	source_files = ("data/quora/all.txt", "data/mscoco/all.txt")
	buildVocabulary(source_files, "transfer_vocab.txt")
	testBLEU()