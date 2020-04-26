This is a course project.

#major changes after the milestone:

    added a new function of distribution.

    fixed a mistake when super the class of LegacyCriterion.

    changed fairseq module of criterion, make it recognize the new criterion.

    added notebooks which generated graphs.

    added proof of experiments.

    paper uploaded.


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

We can generate the D2GPo using a large dataset while training on a small one. 

We provide the evaluation on two kinds of datasets: the larger one(WMT 14 de-en from 
http://statmt.org/wmt14/translation-task.htmlDownload), and the smaller one(torchnlp.datasets.iwslt). To simplify, we provide the starter code to generate the D2GPo on the europev7 monolingul dataset, and train the transformer model on iwslt 16 dataset. 

The generation of D2GPo:

1, Have the fasttext download, and then do the skipgram by the script.

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
  #or rectangular
  python generate_d2gpo_distribution.py rectengular 2 presoftmax 1 ./data-bin/d2gpo.en-de.order.idx ./data-bin/d2gpo.en-de.rec_2.h5
```

4, copy the .h5 file to the data-bin folder of fairseq

```
cp ./data-bin/d2gpo.en-de.d2gpo.en-de.rec_2.h ../fairseq/data-bin/iwslt14_en_de/d2gpo.en-de.rec_2.h
```


Do the fairseq experiments:

1, Prepare the data:

```
cd examples/translation/
bash prepare-iwslt.sh
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
! cd fairseq; CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    data-bin/iwslt14_en_de \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 4096 \
    --fp16 \
    --max-epoch 50 \
    --seed 1 \
    --criterion d2gpo_label_smoothed_cross_entropy \
    --label-smoothing 0.1 --d2gpo-alpha 0.1 \
    --d2gpo-temperature 2 \
    --d2gpo-weight-path ./data-bin/d2gpo.en-de.rec_2.h5
    --d2gpo-vocab-path ./data-bin/iwslt14_en_de/d2gpo.en-de.vocab
```

4, Evaluate the model:

```
!cd fairseq; fairseq-generate data-bin/iwslt14_en_de \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --quiet
```




