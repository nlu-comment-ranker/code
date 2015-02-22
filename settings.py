##
# Various constants and parameter lists
# used to configure classifier.py

GRIDSEARCH_PARAMS = {
    "svr": {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100, 500],
        # "degree": [0.5, 1, 2, 3, 4, 5],
        "gamma": [0.0, 0,00001, 0.001, 0.01, 0.1, 1.0],
        "epsilon": [0.1],
        "tol": [1e-1]
        },
    "rf": {
        "n_estimators": [10, 30, 100],
        "criterion": ["mse"],
        "max_features": ["auto"],
        "max_depth": [5, 10, 20]
    },
    "elasticnet": {
        "alpha": [0.1, 1, 10, 100, 500],
        "l1_ratio": [0, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
    }
}

STANDARD_PARAMS = {
    "svr": {
        "epsilon": 0.1,
        "C": 1,
        "tol": 0.1,
        "gamma": 0.0,
        "kernel": "rbf"
    },
    "rf": {
        "n_estimators": 100,
        "criterion": "mse",
        "max_features": "auto",
        "max_depth": 5
    }
}

# This is somewhat redundant with features.py,
# but better reflects the feature groups we tested
# in our original paper
FG = {
    "length": ["n_chars", "n_words",
               "n_sentences", "n_paragraphs",
               "n_uppercase"],
    "tok_n": ["tok_n_links", "tok_n_emph",
              "tok_n_nums", "tok_n_quote"],
    "distro": ["SMOG", "entropy"],
    "pos_n": ["pos_n_noun", "pos_n_nounproper",
              "pos_n_verb", "pos_n_adj", "pos_n_adv",
              "pos_n_inter", "pos_n_wh", "pos_n_particle",
              "pos_n_numeral"],
    "pos_f": ["pos_f_noun", "pos_f_nounproper",
              "pos_f_verb", "pos_f_adj", "pos_f_adv",
              "pos_f_inter", "pos_f_wh", "pos_f_particle",
              "pos_f_numeral"],
    "metadata": ["position_rank",
                 "timedelta",
                 # 'num_reports', # appears to be missing?
                 # 'distinguished' # string feature; can't convert to float
                 ],
    "posterior": ['gilded', 'num_replies', 'convo_depth'],
    "context": ["parent_term_overlap",
                "parent_jaccard_overlap",
                "parent_tfidf_overlap",
                "informativeness_thread",
                "informativeness_global"],
    "user": ["is_mod", "is_gold", "has_verified_email"],
    "user_local": [
        "user_local_comment_count",
        "user_local_comment_pos_karma",
        "user_local_comment_neg_karma",
        "user_local_comment_net_karma",
        "user_local_comment_avg_pos_karma",
        "user_local_comment_avg_neg_karma",
        "user_local_comment_avg_net_karma",
        "user_local_sub_count",
        "user_local_sub_pos_karma",
        "user_local_sub_neg_karma",
        "user_local_sub_net_karma",
        "user_local_sub_avg_pos_karma",
        "user_local_sub_avg_neg_karma",
        "user_local_sub_avg_net_karma"],
    "user_global": [
        "user_global_comment_count",
        "user_global_comment_pos_karma",
        "user_global_comment_neg_karma",
        "user_global_comment_net_karma",
        "user_global_comment_avg_pos_karma",
        "user_global_comment_avg_neg_karma",
        "user_global_comment_avg_net_karma",
        "user_global_sub_count",
        "user_global_sub_pos_karma",
        "user_global_sub_neg_karma",
        "user_global_sub_net_karma",
        "user_global_sub_avg_pos_karma",
        "user_global_sub_avg_neg_karma",
        "user_global_sub_avg_net_karma"]
}

FG_CUSTOM = {
    "baseline": FG["length"],
    "user_only": (FG["user"]
                  + FG["user_local"]
                  + FG["user_global"]),
    "metadata_only": [
        "timedelta"
    ],
    "all_text": (FG["length"] + FG["tok_n"] + FG["distro"]
                 + FG["pos_n"] + FG["pos_f"] + FG["metadata"]
                 + FG["context"]),
    "all_text-june":[
        "n_chars",
        "n_words",
        "n_sentences",
        "n_paragraphs",
        "n_uppercase",
        "tok_n_links",
        "tok_n_emph",
        "tok_n_nums",
        "tok_n_quote",
        "SMOG",
        "entropy",
        "pos_f_noun",
        "pos_f_nounproper",
        "pos_f_verb",
        "pos_f_adj",
        "pos_f_adv",
        "pos_f_inter",
        "pos_f_wh",
        "pos_f_particle",
        "pos_f_numeral",
        "parent_term_overlap",
        "parent_jaccard_overlap",
        "parent_tfidf_overlap",
        "informativeness_thread",
        "informativeness_global",
        "gilded",
        "num_replies",
        "convo_depth",
        "timedelta"
    ],
    "combo": [
        "n_chars",
        "n_words",
        "n_sentences",
        "n_paragraphs",
        "SMOG",
        "entropy",
        "pos_f_noun",
        "pos_f_verb",
        "parent_tfidf_overlap",
        "informativeness_thread",
        "user_local_comment_count",
        "user_local_comment_net_karma",
        "user_local_comment_avg_net_karma",
        "user_local_sub_count",
        "user_local_sub_net_karma",
        "user_local_sub_avg_net_karma",
        "user_global_comment_count",
        "user_global_comment_net_karma",
        "user_global_comment_avg_net_karma",
        "user_global_sub_count",
        "user_global_sub_net_karma",
        "user_global_sub_avg_net_karma"
        ],
    "all": [
        "n_chars",
        "n_words",
        "n_sentences",
        "n_paragraphs",
        "n_uppercase",
        "tok_n_links",
        "tok_n_emph",
        "tok_n_nums",
        "tok_n_quote",
        "SMOG",
        "entropy",
        "pos_f_noun",
        "pos_f_nounproper",
        "pos_f_verb",
        "pos_f_verb",
        "pos_f_adj",
        "pos_f_adv",
        "pos_f_inter",
        "pos_f_wh",
        "pos_f_particle",
        "pos_f_numeral",
        "parent_term_overlap",
        "parent_jaccard_overlap",
        "parent_tfidf_overlap",
        "informativeness_thread",
        "informativeness_global",
        "timedelta",
        "is_mod",
        "is_gold",
        "has_verified_email",
        "user_local_comment_count",
        "user_local_comment_net_karma",
        "user_local_comment_avg_net_karma",
        "user_local_sub_count",
        "user_local_sub_net_karma",
        "user_local_sub_avg_net_karma",
        "user_global_comment_count",
        "user_global_comment_net_karma",
        "user_global_comment_avg_net_karma",
        "user_global_sub_count",
        "user_global_sub_net_karma",
        "user_global_sub_avg_net_karma"
        ],
    "all-june": [
        "n_chars",
        "n_words",
        "n_sentences",
        "n_paragraphs",
        "n_uppercase",
        "tok_n_links",
        "tok_n_emph",
        "tok_n_nums",
        "tok_n_quote",
        "SMOG",
        "entropy",
        "pos_f_noun",
        "pos_f_nounproper",
        "pos_f_verb",
        "pos_f_verb",
        "pos_f_adj",
        "pos_f_adv",
        "pos_f_inter",
        "pos_f_wh",
        "pos_f_particle",
        "pos_f_numeral",
        "parent_term_overlap",
        "parent_jaccard_overlap",
        "parent_tfidf_overlap",
        "informativeness_thread",
        "informativeness_global",
        "gilded",
        "num_replies",
        "convo_depth",
        "timedelta",
        "is_mod",
        "is_gold",
        "has_verified_email",
        "user_local_comment_count",
        "user_local_comment_net_karma",
        "user_local_comment_avg_net_karma",
        "user_local_sub_count",
        "user_local_sub_net_karma",
        "user_local_sub_avg_net_karma",
        "user_global_comment_count",
        "user_global_comment_net_karma",
        "user_global_comment_avg_net_karma",
        "user_global_sub_count",
        "user_global_sub_net_karma",
        "user_global_sub_avg_net_karma"
        ]
}

FG.update(FG_CUSTOM)
FG = {k:set(l) for k,l in FG.items()} # remove possible duplicates
FEATURE_GROUPS = FG