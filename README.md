# Robustness Against Polarity Bias in Decoupled Group Recommendations Evaluation
## Abstract
Group recommendations are a specific case of recommender systems, where instead of recommending for each individual independently
a shared recommendations are produced for a group of users. Usually, group recommendation techniques are built on top of common
"single-user" RS and the resulting group recommendation should both reflect the overall utility of the recommendation as well as
fairness among the utilities of individual group members.
Off-line evaluations of group recommendations were so far resolved either as a tightly coupled pair with the underlying RS or in a
decoupled fashion. In the latter case, the relevance scores estimated by underlying RS serves as a ground truth for the evaluation of
group aggregator. Both coupled and decoupled evaluation may suffer from different biases that provide illicit advantages to some
classes of group recommending strategies. In this paper, we focus on the decoupled evaluation protocol and possible polarity bias of
the underlying recommender system. We consider polarity bias as the situation when RS in average under-estimate or over-estimate
the true utility of recommended items.
In experimental part, we propose and evaluate several de-biasing strategies and show to what extent are individual group RS robust
against the polarity bias.

## Contacts
- Patrik Dokoupil - *patrik.dokoupil at matfyz.cuni.cz*
- Ladislav Peška - *ladislav.peska at matfyz.cuni.cz*

## About this repository
- This repository contains source codes for generating results used in the paper and jupyter notebooks for the analysis of the results. These scripts expect outputs of base recommender (ALS Matrix Factorization) and their aggregations (via group recommender algorithms) to be present. All these raw results were obtained by using implementation from one of the author's previous publication [1] and are available in the following repository: https://github.com/LadislavMalecek/UMAP2021

### Project structure
- [ml1m](./ml1m/) folder contains everything related to evaluation on ML1M dataset
    - 
- [kgrec](./kgrec/) folder contains everything related to evaluation on KGREC dataset

### Requirements
Having all the data in the format produced by https://github.com/LadislavMalecek/UMAP2021
The code was tested with Python 3.9.6 (but slightly older version should work fine as well), sklearn and numpy.

### Parameters
- `--rating_normalization` specifies the type of global normalization to use. Only possible value at the moment is `"norm_shift_nonlinear"` which normalizes by `max(EPSILON, rating + C)`
- `--user_rating_normalization` specifies the user level normalization, possible values are `"u_norm_min_max_scaler"` (meaning min max normalization to [0, 1]) and `""` (meaning identity)
- `--use_quadratic_amplification` when specifies, quadratic amplification is used
- `--normalization_c` the constant `C` used in the global normalization
- `--use_all_constants` if specified, the algorithm will iterate over predefined, fixed set of constants that were used in the paper.
- `--path_prefix` path to the root of the repository. If you run the scripts from the repository root itself, just pass `"."`.
- `--group_types` string with comma separated group types, default is `"sim"` which corresponds to similar group.
- `--group_sizes` string with comma separated group sizes, default is `"2,4,8"`

### Running the project

```
cd evaluation
python3 compute_metrics_aggregatedResults.py --rating_normalization "norm_shift_nonlinear" --user_rating_normalization "u_norm_min_max_scaler" --normalization_c -0.5 --group_types "sim,div" --group_sizes "2,4,8" --path_prefix "/mnt/0"
```

## References
[1] Ladislav Peska and Ladislav Malecek. 2021. Coupled or Decoupled Evaluation for Group Recommendation Methods?. In Proceedings of the Perspectives
on the Evaluation of Recommender Systems Workshop 2021 co-located with the 15th ACM Conference on Recommender Systems (RecSys 2021), Amsterdam,
The Netherlands, September 25, 2021 (CEUR Workshop Proceedings, Vol. 2955), Eva Zangerle, Christine Bauer, and Alan Said (Eds.). CEUR-WS.org.
http://ceur-ws.org/Vol-2955/paper1.pdf