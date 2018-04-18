
def recordVocabulary(destination):
	print("sadsdfsd")

def buildVocabulary(source_file, target_file):
	vocab = []

	with open(source_file) as finput, open(target_file) as foutput:
		for source,target in zip(finput, foutput):
			for i in source.split(" "):

				s = i.replace("?", "")
				s = s.replace(".", "")
				s = s.replace("!", "")
				s = s.replace("\n", "")

				if(s not in vocab): vocab.append(s)

			for i in target.split(" "):

				s = i.replace("?", "")
				s = s.replace(".", "")
				s = s.replace("!", "")
				s = s.replace("\n", "")

				if(s not in vocab): vocab.append(s)

	print(vocab)

if __name__ == "__main__":
	buildVocabulary("data/quora/train_source.txt", "data/quora/train_target.txt")