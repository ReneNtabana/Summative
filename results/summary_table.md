| Family | Model | Config | Seeds | Accuracy | Macro-F1 | 5-fold CV F1 | OOD Acc | OOD Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| baseline | majority |  | 1 | 0.351 | 0.052 |  |  |  |
| baseline | tfidf-word|nb | C=0.003 | 1 | 0.866 | 0.679 | 0.700 |  |  |
| baseline | tfidf-word|logreg | C=0.3 | 1 | 0.897 | 0.807 | 0.760 |  |  |
| baseline | tfidf-word+char|logreg | C=0.3 | 1 | 0.901 | 0.819 | 0.776 |  |  |
| baseline | tfidf-word+char|svm | C=0.1 | 1 | 0.908 | 0.833 | 0.805 |  |  |
| baseline | tfidf-word|svm | C=0.1 | 1 | 0.905 | 0.833 | 0.773 |  |  |
| baseline | tfidf-char|logreg | C=0.3 | 1 | 0.901 | 0.852 | 0.789 |  |  |
| baseline | tfidf-char|svm | C=0.3 | 1 | 0.912 | 0.866 | 0.806 |  |  |
| embeddings | avg-fasttext-kir | C=0.1 | 1 | 0.821 | 0.698 |  |  |  |
| embeddings | avg-w2v-kir | C=0.1 | 1 | 0.844 | 0.704 |  |  |  |
| embeddings | avg-fasttext-kir+kin | C=0.3 | 1 | 0.851 | 0.714 |  |  |  |
| embeddings | avg-w2v-kir+kin | C=0.1 | 1 | 0.840 | 0.729 |  |  |  |
| rnn | bilstm-attn-random | lr=0.002, max_len=400 | 3 | 0.762 ± 0.069 | 0.585 ± 0.054 |  |  |  |
| rnn | bilstm-maxmean-random | lr=0.002, max_len=400 | 3 | 0.785 ± 0.025 | 0.603 ± 0.084 |  |  |  |
| rnn | bilstm-attn-w2v-kir+kin | lr=0.002, max_len=400 | 3 | 0.835 ± 0.057 | 0.684 ± 0.058 |  |  |  |
| rnn | bilstm-maxmean-w2v-kir | lr=0.002, max_len=400 | 3 | 0.858 ± 0.019 | 0.706 ± 0.015 |  |  |  |
| rnn | bilstm-maxmean-w2v-kir+kin | lr=0.002, max_len=400 | 3 | 0.854 ± 0.029 | 0.717 ± 0.071 |  |  |  |
| rnn | bilstm-attn-fasttext-kir+kin | lr=0.002, max_len=400 | 3 | 0.880 ± 0.015 | 0.721 ± 0.052 |  |  |  |
| rnn | bilstm-maxmean-fasttext-kir+kin | lr=0.002, max_len=400 | 3 | 0.859 ± 0.019 | 0.736 ± 0.021 |  |  |  |
| transformer | xlm-roberta-base_lr5e-5-ep15 | lr=5e-05, max_len=512 | 1 | 0.252 | 0.099 |  |  |  |
| transformer | xlm-roberta-base | lr=3e-05, max_len=512 | 1 | 0.569 | 0.299 |  | 0.466 | 0.350 |
| transformer | afriberta_base-fullFT | lr=3e-05, max_len=512 | 1 | 0.893 | 0.774 |  | 0.830 | 0.755 |
| transformer | afriberta_base | lr=3e-05, max_len=512 | 3 | 0.893 ± 0.014 | 0.804 ± 0.025 |  | 0.830 | 0.753 |
| transformer | afro-xlmr-base | lr=3e-05, max_len=512 | 3 | 0.910 ± 0.012 | 0.829 ± 0.034 |  | 0.872 | 0.799 |

OOD scores for Transformers are from the seed-42 model.
