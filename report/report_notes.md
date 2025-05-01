# Notes for the report

### Why we work with sentences
We are searching for the scope inside of cue context. It is defined as a sentence. A function that parsers every period on the text and sees its inmediate context determines which of them are ortographic periods and which not (e.g. the period on `dr. *****` would not be considered).

### Uncertainty `no` and `sin`
there is a problem (to us) with the training data. some CUEs are duplicates on both lexicon. most notably, `no` and `sin` are also used on the uncertaity lexicon. the real problem comes when taking into account that these two words represent the vast majority of the cases in our training dataset.

If they are included on both lexicons, we would have duplicates predicions, thus increasing our false postives for the uncertainties.

They are only used on `UNC` less than 10 times each. We decided to remove them from the uncertainty lexicon.