import sys

specials = ['<s>', '<pad>', '</s>', '<unk>']

with open(sys.argv[1], 'r', encoding='utf-8') as fin:
    with open(sys.argv[2], 'w', encoding='utf-8') as fout:
        for word in specials:
            fout.write(word+'\n')
        for line in fin:
            idx = line.rfind(' ')
            word = line[:idx]
            fout.write(word+'\n')