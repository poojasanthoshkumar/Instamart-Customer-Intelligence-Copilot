Instamart Customer Intelligence & Support Copilot

**1. Project Overview**

The Instamart Customer Intelligence & Support Copilot is a reproducible data analytics and machine-learning project built using Python, pandas, NumPy, SQLite, and scikit-learn. It establishes a cleaned data foundation, analyses customer purchasing behaviour, identifies customer segments, and predicts customer lapse risk.

The project is organised into four parts:

Part A — Data Foundation and Analytics: Raw-data auditing, data cleaning, SQLite database construction, SQL analysis, and exploratory data analysis.

Part B — Customer Intelligence: RFM analysis, K-Means customer segmentation, lapse prediction, model evaluation, oversampling, feature interpretation, and retention prioritisation.

Part C — Support-Ticket Triage: PII masking and support-ticket classification.

Part D — Policy RAG and Refund Agent: Policy retrieval and refund-support automation.

The supplied generate_data.py and mock_llm.py files are retained unchanged.

**2. Technology Stack**

Python 3.11 or later

pandas and NumPy

SQLite 3.25 or later

scikit-learn

joblib

VS Code

Git and GitHub

3. Project Structure

Instamart_Copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   └── orders.csv
│   └── instamart.db
├── sql/
│   └── queries.sql
├── outputs/
│   ├── customer_segments.csv
│   ├── kmeans_evaluation.csv
│   ├── lapse_dataset.csv
│   ├── lapse_model.joblib
│   ├── lapse_model_comparison.json
│   ├── logistic_oversampling_comparison.csv
│   ├── logistic_regression_coefficients.csv
│   ├── pca_coordinates.csv
│   ├── random_forest_feature_importance.csv
│   ├── retention_list.csv
│   ├── rfm_scaled.csv
│   ├── rfm_table.csv
│   └── segment_profiles.csv
├── generate_data.py
├── mock_llm.py
├── audit_data.py
├── clean_customers.py
├── clean_products.py
├── clean_orders.py
├── build_database.py
├── validate_database.py
├── run_sql_queries.py
├── pandas_analysis.py
├── build_rfm.py
├── cluster_customers.py
├── create_segments.py
├── build_lapse_dataset.py
├── train_lapse_model.py
├── oversample_lapse_model.py
├── oversample_logistic_regression.py
├── analyse_lapse_features.py
├── check_scaler_mean.py
├── finalize_lapse_outputs.py
├── eda_report.txt
├── requirements.txt
├── .gitignore
└── README.md

The outputs/ directory contains generated results and trained model artefacts. Some intermediate or additional output files may not be listed in this overview.

4. Setup and Reproduction

4.1 Prerequisites

Install Python 3.11 or later and Git. Clone the repository and navigate to the project directory.

git clone https://github.com/poojasanthoshkumar/Instamart-Customer-Intelligence-Copilot.git
cd Instamart-Customer-Intelligence-Copilot

4.2 Create the virtual environment

python -m venv .venv
.\.venv\Scripts\Activate.ps1

4.3 Install dependencies

python -m pip install -r requirements.txt

4.4 Generate and audit raw data

python generate_data.py
python audit_data.py

4.5 Clean the datasets

python clean_customers.py
python clean_products.py
python clean_orders.py

4.6 Build and validate the database

python build_database.py
python validate_database.py

The expected validation message is ALL CHECKPOINTS PASSED.

4.7 Run SQL and pandas analyses

python run_sql_queries.py
python pandas_analysis.py

Expected completion messages:

ALL SEVEN SQL QUERIES EXECUTED SUCCESSFULLY

ALL PANDAS ANALYSIS CHECKS PASSED

4.8 Reproduce Part B

After completing Part A, run the following commands from the project root:

python build_rfm.py
python cluster_customers.py
python create_segments.py
python build_lapse_dataset.py
python train_lapse_model.py
python oversample_lapse_model.py
python oversample_logistic_regression.py
python analyse_lapse_features.py
python check_scaler_mean.py
python finalize_lapse_outputs.py

These scripts generate the RFM dataset, clustering evaluation, customer profiles, lapse cohort, model evaluation results, feature interpretation outputs, and retention list.

The supplied data-generation and mock-LLM scripts should not be modified.

5. Part A — Data Foundation and Analytics

5.1 Data cleaning

Customers

Standardised city names by trimming whitespace and applying title case.

Standardised gender values to male, female, or unknown.

Treated ages outside the 10–100 range as invalid and set them to missing.

Imputed missing ages using the median age within each city.

Converted signup dates to datetime format.

Products

Retained 40 product records and verified product ID uniqueness.

Checked for missing values and standardised category data where required.

Orders

Converted order timestamps to datetime.

Removed 119 exact duplicate records and retained the latest record for each order ID, producing 12,309 unique orders.

Converted order amounts to numeric values and treated invalid amounts as missing.

Replaced missing order amounts using product price, quantity, and discount:

price_inr * quantity * (1 - discount_pct / 100)

Treated negative delivery times as missing.

Imputed missing delivery times for completed orders using the median for the corresponding product category.

Kept delivery times missing for cancelled orders.

5.2 Database design

The SQLite database, data/instamart.db, contains three tables:

customers

products

orders

The database is built from the cleaned CSV files. Order timestamps are stored as text in YYYY-MM-DD HH:MM:SS format.

5.3 Key results

Metric

Result

Customers

1,200

Products

40

Unique orders

12,309

Completed orders

11,110

Cancelled orders

1,199

Completed-order revenue

INR 2,998,668.80

Completed-order mean amount

INR 269.91

Completed-order median amount

INR 135.00

Upper IQR outliers

897

The pandas pivot-table revenue total matches the SQL revenue total. The seven SQL queries cover customer spending, city and category revenue, above-average product prices, customers without completed orders, spending relative to the customer average, city-wise customer rankings, and cumulative daily revenue.

5.4 Reproducibility and integrity

The supplied generate_data.py and mock_llm.py files are retained unchanged. Cleaning scripts generate the cleaned CSV files and SQLite database. Validation scripts check key row counts, missing-value rules, revenue totals, database tables, and query outputs.

Part A establishes the cleaned data layer, database, SQL analytics, pandas cross-checks, and EDA report used by subsequent parts.

6. Part B — Customer Intelligence

6.1 RFM analysis

RFM analysis was performed using completed orders as of the snapshot date 2025-10-01 00:00:00.

Recency: Whole days since the customer's most recent completed order.

Frequency: Number of distinct completed orders.

Monetary: Sum of completed-order amounts.

The resulting RFM dataset contains 1,175 customers.

Metric

Result

Customers analysed

1,175

Median recency

19 days

Minimum recency

0 days

Maximum recency

180 days

Mean frequency

9.46 orders

Maximum frequency

33 orders

Mean monetary value

INR 2,552.06

Maximum monetary value

INR 14,766.55

Output: outputs/rfm_table.csv

6.2 K-Means clustering and segment profiles

RFM features were standardised using StandardScaler. K-Means was evaluated for K values from 2 to 8 using n_init=10 and random_state=42. Inertia and silhouette scores were recorded, and K=4 was selected.

The following table summarises the resulting customer segments.

Segment

Customers

Mean Recency (days)

Mean Frequency

Mean Monetary (INR)

At-Risk Customers

436

28.86

3.77

1,057.62

Premium

138

24.12

6.77

7,129.70

Loyal Customers

419

8.77

19.40

3,410.02

Dormant Customers

182

109.04

2.20

685.99

The Premium segment is identified as the cluster with the highest mean monetary value. The is_premium flag is true for customers assigned to that segment and false for customers assigned to the other segments.

The first two PCA components explain 63.4% and 21.8% of the variance, respectively, for a combined 85.2%.

Outputs:

outputs/kmeans_evaluation.csv

outputs/rfm_scaled.csv

outputs/customer_segments.csv

outputs/segment_profiles.csv

outputs/pca_coordinates.csv

6.3 Lapse cohort definition

The lapse cohort includes customers with at least one completed order before 2025-09-01.

A customer is labelled as lapsed if they have no completed order during 2025-09-01 to 2025-09-30. Otherwise, the customer is labelled not lapsed.

Metric

Result

Cohort size

1,156

Not-lapsed customers

713

Lapsed customers

443

Lapsed proportion

38.32%

Training rows

924

Test rows

232

The train-test split is stratified and uses random_state=42.

6.4 Feature engineering and leakage prevention

Features include historical recency, frequency, monetary value, average discount, cancellation rate, average delivery time, favourite product category, city, age, gender, and customer tenure.

The feature-engineering cutoff is 2025-09-01. No order data from 2025-09-01 onwards entered the model features. Orders in September were used only to construct the target label.

The segment labels generated in Task 3 were not used as lapse-model features.

Preprocessing uses StandardScaler for numerical features and OneHotEncoder(drop="first") for categorical features. Preprocessing is fitted on the training partition only and then applied to the test partition.

No encoder or scaler was fitted on test rows. No resampler was fitted on or applied to the test set. Oversampling was performed only on the training partition, and all final test metrics were calculated on the unchanged held-out test set.

The full-cohort mean frequency is 8.0285, while the training-fitted scaler's mean frequency is 8.1147. The difference is expected because the scaler is fitted using training rows only.

6.5 Lapse-model comparison

Logistic Regression and Random Forest were evaluated using five-fold GridSearchCV with F1 scoring.

Logistic Regression grid

C: [0.01, 0.1, 1, 10, 100]

max_iter=1000

Random Forest grid

max_depth: [3, 5, 8]

min_samples_leaf: [5, 10, 20]

n_estimators=100

random_state=42

Model

Best Parameters

Best CV F1

Test Precision

Test Recall

Test F1

Logistic Regression

C = 0.1

0.6469

0.6180

0.6180

0.6180

Random Forest

max_depth = 8, min_samples_leaf = 5

0.6812

0.6000

0.6404

0.6196

Precision, recall, and F1 are reported for the positive class: lapsed customers. The selected Random Forest achieved a test F1 of 0.6196, exceeding the required minimum of 0.55.

The baseline predicts every customer as not lapsed. Its accuracy is approximately 61.6%, but its F1 score for the lapsed class is zero. The trained models can identify lapsed customers, whereas the baseline identifies none.

Output files:

outputs/lapse_dataset.csv

outputs/lapse_model_comparison.json

outputs/lapse_model.joblib

6.6 Oversampling comparison

Random oversampling with replacement was used to balance the training classes. The original training counts were 570 not-lapsed and 354 lapsed customers. After oversampling, each class contained 570 rows.

The Logistic Regression model was refitted on the balanced training data and evaluated on the unchanged test set.

Logistic Regression

Precision

Recall

F1

Original training distribution

0.6180

0.6180

0.6180

Oversampled training distribution

0.5738

0.7865

0.6635

Oversampling improved recall and F1 for lapsed customers but reduced precision. This trade-off can help identify more customers at risk while potentially increasing unnecessary retention outreach.

Outputs:

outputs/logistic_oversampling_comparison.csv

outputs/logistic_oversampling_details.json

outputs/lapse_logistic_oversampled.joblib

outputs/oversampling_comparison.csv

outputs/lapse_model_oversampled.joblib

6.7 Feature interpretation

Random Forest feature importances and Logistic Regression coefficients were exported for interpretation. The leading Random Forest features include frequency, monetary value, cancellation rate, recency, and average delivery time.

Logistic Regression coefficients indicate the direction of association between a feature and the model's predicted lapse class, conditional on the fitted model and its other features. They should not be interpreted as causal effects.

Outputs:

outputs/random_forest_feature_importance.csv

outputs/logistic_regression_coefficients.csv

6.8 Retention list and model reproducibility

A retention list was generated by ranking Premium-segment test customers according to predicted lapse probability. Two of the ten listed customers were labelled as lapsed.

Output: outputs/retention_list.csv

Saved models were reloaded and their predictions compared with predictions from the original fitted models to check reproducibility.

6.9 Part B limitations

The Premium flag is a post-hoc whole-window label. It is based on the RFM snapshot as of 2025-10-01, while lapse prediction uses a historical cutoff of 2025-09-01. The Premium-filtered retention list is therefore illustrative and should not be interpreted as a fully leakage-free prospective retention evaluation.

Other limitations include:

The dataset is synthetic, so results may differ on real customer data.

The test partition measures performance on a held-out subset of the generated cohort, not on a future external cohort.

Oversampling increases recall but reduces precision, which may increase unnecessary retention interventions.

The retention list should be validated through prospective evaluation before operational use.

6.10 Part B completion

Part B implements RFM analysis, K-Means segmentation, lapse cohort construction, cross-validated model selection, training-only oversampling, feature interpretation, saved-model checks, and retention prioritisation. The key outputs are stored under outputs/.


## 7. Part C — Support-Ticket Triage

### 7.1 Objective and input validation

Part C implements a privacy-aware support-ticket triage pipeline using the supplied `mock_llm.py` interface. It classifies eligible tickets into billing, delivery, technical, or fraud categories, validates model responses, tracks token usage, and routes tickets to the appropriate support destination.

The input dataset contains 164 tickets. Empty or whitespace-only ticket texts are assigned `skipped_empty`, while tickets exceeding the 400-token input threshold are assigned `flagged_long`. These checks occur before any model call.

| Input status | Tickets |
|---|---:|
| Eligible for classification | 156 |
| Empty tickets skipped | 4 |
| Overlength tickets flagged | 4 |
| Total | 164 |

### 7.2 PII masking and privacy safeguards

Before classification, the pipeline masks email addresses, ten-digit Indian mobile numbers, and order identifiers using the placeholders `[EMAIL]`, `[PHONE]`, and `[ORDER_ID]`, respectively. The regular expressions are applied in the prescribed order.

| PII type | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order identifiers | 132 |
| Total substitutions | 278 |

Validation detected zero remaining matches for the configured PII patterns after redaction. The model's user message contains only the redacted ticket text; customer identifiers, original ticket text, and other metadata are not added to that message.

The input audit is saved as `outputs/triage_input_audit.csv`.

### 7.3 Classification prompts and response validation

Two prompt strategies were evaluated using the same deterministic mock model:

- **Zero-shot classification:** Provides category definitions and the required JSON response schema without examples.
- **Few-shot classification:** Uses the same instructions and schema, supplemented by labelled examples covering fraud, billing, and technical issues.

Responses are parsed as JSON and accepted as `classified_json` when the category belongs to the allowed set. If JSON parsing fails, a regular-expression fallback accepts a response as `classified_keyword` only when exactly one allowed category word is found. Unsupported categories, including `shipping`, and other unparseable responses are assigned `flagged_invalid_output` for human review.

Transient model errors are retried up to three times per ticket, with exponential backoff and jitter. Retry events are logged. The supplied `mock_llm.py` file remains unchanged.

### 7.4 Evaluation results

Both full evaluation runs processed 164 records, including the eight tickets excluded by the input guardrails. Classification metrics are calculated only for the 147 tickets receiving valid category predictions.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| JSON classifications | 135 | 135 |
| Keyword-fallback classifications | 12 | 12 |
| Invalid outputs | 9 | 9 |
| Validly classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy among classified tickets | 85.71% | 100.00% |
| Total response tokens | 12,364 | 20,944 |
| Tokens per classified ticket | 84.11 | 142.48 |
| Transient errors recovered | 19 | 19 |

The zero-shot confusion matrix showed that the errors were concentrated in fraud and technical tickets predicted as billing. The few-shot prompt achieved perfect accuracy on the classified subset under the mock model's deterministic keyword rules. This improvement came with higher token consumption because the system prompt included examples.

These findings describe performance on the supplied synthetic evaluation environment. They should not be interpreted as a guarantee of accuracy on real-world customer tickets.

### 7.5 Token-budget demonstration

A separate zero-shot run used a token budget of 6,000. Before each model call, the pipeline checked whether accumulated response-token usage had reached the budget. Once the threshold was reached, remaining eligible tickets were assigned `budget_exhausted` and routed for human review.

The run classified 72 tickets, used 6,038 response tokens, and routed 80 tickets to human review because the budget was exhausted. The final permitted response can take the total above the threshold because the check occurs before the call.

Output: `outputs/triage_budget_demo.csv`.

The budget demonstration records the mock model's reported response-token usage. It does not represent a monetary cost estimate for a hosted language model.

### 7.6 Final routing and priority rules

The final output uses the few-shot prompt with a 100,000-token budget and applies the following rules in order:

1. Tickets with `skipped_empty`, `flagged_long`, `flagged_invalid_output`, or `budget_exhausted` status are routed to `human_review` with `NORMAL` priority and no category.
2. Validly classified fraud tickets are routed to `security_team` with `HIGH` priority.
3. Other validly classified tickets belonging to premium customers are routed to `premium_desk` with `HIGH` priority.
4. All remaining validly classified tickets are routed to `support_queue` with `NORMAL` priority.

Customer segments are joined using `customer_id` from `outputs/customer_segments.csv`. Missing segment records are treated as non-premium.

The final output contains 164 unique ticket records.

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Premium desk | 11 |
| Human review | 17 |
| Total | 164 |

Output: `outputs/triage_results.csv`.

The output contains `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`.

### 7.7 Human review and limitations

Human review is required for empty tickets, overlength tickets, invalid model responses, and tickets for which the token budget prevents classification. These records are not assigned a predicted category. Valid classifications are routed according to the fraud and premium-customer rules.

The pipeline uses a deterministic mock LLM rather than a production model. Its accuracy, response patterns, retry behaviour, and token counts are specific to the supplied implementation. PII checks cover the configured regular expressions and do not guarantee detection of every possible form of sensitive information. A production deployment would require broader privacy validation, human-reviewed evaluation data, monitoring, and testing across real customer language.

### 7.8 Part C completion

Part C implements PII redaction, input guardrails, zero-shot and few-shot classification, retry handling, response validation, token-budget enforcement, evaluation, and customer-aware routing. The required final and budget-demonstration CSV outputs are generated under `outputs/`. The supplied mock model remains unchanged.