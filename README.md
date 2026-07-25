SMS Spam Classifier — TF-IDF + Logistic Regression

A binary text classifier that flags SMS messages as spam or ham (legitimate), built with a classical NLP pipeline: TF-IDF vectorization + Logistic Regression. Built primarily as a learning project to understand the full pipeline end-to-end, including where a "good accuracy" number can hide a weak model.

SMS Spam Collection Dataset — 5,572 SMS messages labeled ham or spam (4,825 ham / 747 spam after loading; 5,169 after removing 403 duplicate rows).

Pipeline:

. Load data, clean/lowercase labels, map spam → 1, ham → 0
. Drop 403 duplicate rows (present in the raw dataset — left in, they leak across train/test)
. Stratified train/test split (80/20), preserving the ~87/13 ham/spam class ratio in both sets
. TfidfVectorizer fit on training text only, then used to transform test text (avoids vocabulary/IDF leakage from test data)
. LogisticRegression trained on the TF-IDF matrix
. Evaluated with precision/recall/F1 per class and a confusion matrix — not just accuracy, given the class imbalance

Results:

Accuracy alone was misleading. The default model looked strong at 96.7% accuracy, but was missing 1 in 4 real spam messages (recall 0.748) — because ham dominates the dataset (~87%), a model can score high just by leaning "ham." Rebalancing class weights improved spam recall to 0.908, at the cost of some precision (more real messages incorrectly flagged as spam). Which model is "better" depends on the cost of a missed spam message vs. a wrongly-flagged real message — not something accuracy alone can tell you.

What the model gets wrong:

. False negatives (real spam missed): tend to avoid typical spam vocabulary ("FREE," "WIN," urgency language), reading more like casual or ambiguous text.
. False positives (real messages flagged as spam): often contain words that are statistically spam-associated even in legitimate context (e.g. a scam-warning message flagged for words like "terrorist" or "sim card").
