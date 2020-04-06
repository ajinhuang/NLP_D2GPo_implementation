This is a course project.

The artical, introducing D2GPo, wrote by [Li et al]. , can be found here: https://openreview.net/pdf?id=S1efxTVYDr.
The majority of code was created by
```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
The generation of D2Gpo:

1, Have the fasttext download, and then do the skipgram by script.

  ```
  
  textfile.sh
  
  ```
   
2, Have the word vectors and their topology orders:

  ```
  
  python generate_d2gpo_vocab.py ./data-bin/wmt14ende/dict.de.txt ./data-bin/wmt14ende/d2gpo.en-de.vocab;
  python generate_d2gpo_embedding.py ./D2GPo_supervised_nmt_wmt14_en_de.en-de.all.fasttext.bin ./data-bin/wmt14ende/d2gpo.en-de.vocab ./data-bin/wmt14ende/d2gpo.en-de.vec ./data-bin/wmt14ende/d2gpo.en-de.w2vec;
  python generate_d2gpo_order.py ./data-bin/wmt14ende/d2gpo.en-de.w2vec ./data-bin/wmt14ende/d2gpo.en-de.order.txt ./data-bin/wmt14ende/d2gpo.en-de.order.idx
  
  ```
3, Generate the Gaussian distrubution (and do the softmax), change the prarm to try others:

```
  python generate_d2gpo_distribution.py gaussian 1 0 0 ./data-bin/wmt14ende/d2gpo.en-de.order.idx ./data-bin/wmt14ende/d2gpo.en-de.gaussian_1_0_0.h5
```

Do the fairseq experiments:

1, Prepare the data:

```
cd examples/translation/
bash prepare-wmt14en2de.sh --icml17
cd ../..
```

2, Preprocess the data:

```
TEXT=examples/translation/wmt14_en_de;
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

3, Train the model:
```
mkdir -p checkpoints/fconv_wmt_en_de
fairseq-train \
    data-bin/wmt14_en_de \
    --arch fconv_wmt_en_de \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/fconv_wmt_en_de
    --fp16
    --criterion d2gpo_label_smoothed_cross_entropy --label-smoothing 0.1 --d2gpo-alpha 0.1 --d2gpo-temperature 2 --d2gpo-weight-path ./data-bin/wmt14ende/d2gpo.en-de.gaussian_1_0_0.h5 --d2gpo-vocab-path ./data-bin/wmt14ende/d2gpo.en-de.vocab
```

4, Evaluate the model:

```
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
```




