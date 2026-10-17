| Family | Model | Config | Seeds | Accuracy | Macro-F1 | 5-fold CV F1 | OOD Acc | OOD Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| baseline | majority |  | 1 | 0.351 | 0.052 |  |  |  |
| baseline | tfidf-word|nb | C=0.003 | 1 | 0.866 | 0.679 | 0.700 |  |  |
| baseline | tfidf-word|logreg | C=0.3 | 1 | 0.897 | 0.807 | 0.760 |  |  |
| baseline | tfidf-word+char|logreg | C=0.3 | 1 | 0.901 | 0.819 | 0.776 |  |  |
| baseline | tfidf-word+char|svm | C=0.1 | 1 | 0.908 | 0.833 | 0.805 | 0.801 | 0.722 |
| baseline | tfidf-word|svm | C=0.1 | 1 | 0.905 | 0.833 | 0.773 |  |  |
| baseline | tfidf-char|logreg | C=0.3 | 1 | 0.901 | 0.852 | 0.789 |  |  |
| baseline | tfidf-char|svm | C=0.3 | 1 | 0.912 | 0.866 | 0.806 |  |  |
| embeddings | avg-fasttext-kir | C=0.1 | 1 | 0.821 | 0.698 |  |  |  |
| embeddings | avg-w2v-kir | C=0.1 | 1 | 0.844 | 0.704 |  |  |  |
| embeddings | avg-fasttext-kir+kin | C=0.3 | 1 | 0.851 | 0.714 |  |  |  |
| embeddings | avg-w2v-kir+kin | C=0.1 | 1 | 0.840 | 0.729 |  |  |  |
| rnn | bilstm-maxmean-random | lr=0.002, max_len=400 | 1 | 0.759 | 0.518 |  |  |  |
| rnn | bilstm-maxmean-w2v-kir | lr=0.002, max_len=400 | 1 | 0.878 | 0.705 |  |  |  |
| rnn | bilstm-maxmean-w2v-kir+kin | lr=0.002, max_len=400 | 1 | 0.874 | 0.771 |  |  |  |
