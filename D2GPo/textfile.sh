N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epoch

# fastText
FASTTEXT_DIR=./fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

NAME=D2GPo_supervised_nmt_wmt14_en_de.en-de.all
INPUT=./data/$NAME
OUTPUT=./model/${NAME}.fasttext
LOG=./logs/${NAME}.log

$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $INPUT -output $OUTPUT 1>$LOG 2>&1