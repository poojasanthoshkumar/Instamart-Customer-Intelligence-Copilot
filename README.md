# Instamart Customer Intelligence & Support Copilot

An end-to-end data analytics, machine-learning, customer intelligence, and AI-assisted customer-support project built using Python, pandas, NumPy, SQLite, and scikit-learn.

## 1. Project Overview

The Instamart Customer Intelligence & Support Copilot demonstrates how data analytics and intelligent automation can improve customer understanding, retention planning, support-ticket management, and policy-based customer service in an online grocery delivery environment.

The project is divided into four major components:

- **Part A – Data Foundation and Analytics:** Data auditing, cleaning, database construction, SQL analysis, and exploratory data analysis.
- **Part B – Customer Intelligence:** RFM analysis, K-Means customer segmentation, lapse prediction, model comparison, oversampling, and retention prioritisation.
- **Part C – Support-Ticket Triage:** PII masking, ticket classification, response validation, token-budget management, and customer-aware ticket routing.
- **Part D – Policy RAG and Refund Agent:** Policy-document processing, TF-IDF retrieval, RAG response generation, SQLite-based order lookup, simulated refund processing, and crash-recovery testing.

The supplied `generate_data.py` and `mock_llm.py` files are retained unchanged.

## 2. Technology Stack

| Technology | Application |
|---|---|
| Python 3.11+ | Project implementation |
| pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| SQLite | Relational database |
| scikit-learn | Machine learning and TF-IDF retrieval |
| joblib | Model persistence |
| Matplotlib | Data visualisation |
| Git | Version control |
| GitHub | Project repository |

## 3. Repository Structure

```text
Instamart-Customer-Intelligence-Copilot/
│
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── tickets.csv
│   │   ├── policy_eval.csv
│   │   └── agent_scenarios.csv
│   │
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   └── orders.csv
│   │
│   ├── policies/
│   │   ├── delivery_policy.txt
│   │   ├── escalation_policy.txt
│   │   ├── payment_policy.txt
│   │   ├── privacy_policy.txt
│   │   ├── product_policy.txt
│   │   └── refund_policy.txt
│   │
│   └── instamart.db
│
├── sql/
│   └── queries.sql
│
├── src/
│   └── part_d/
│       ├── evaluate_agent.py
│       ├── evaluate_policy.py
│       ├── policy_processor.py
│       ├── policy_retriever.py
│       ├── rag_pipeline.py
│       ├── refund_agent.py
│       └── test_recovery.py
│
├── outputs/
│   ├── rfm_table.csv
│   ├── rfm_scaled.csv
│   ├── kmeans_evaluation.csv
│   ├── customer_segments.csv
│   ├── segment_profiles.csv
│   ├── pca_coordinates.csv
│   ├── lapse_dataset.csv
│   ├── retention_list.csv
│   ├── lapse_model.joblib
│   ├── lapse_model_oversampled.joblib
│   ├── lapse_model_comparison.json
│   ├── logistic_regression_coefficients.csv
│   ├── random_forest_feature_importance.csv
│   ├── triage_results.csv
│   ├── triage_zero_shot.csv
│   ├── triage_few_shot.csv
│   ├── triage_budget_demo.csv
│   ├── triage_input_audit.csv
│   │
│   └── part_d/
│       ├── policy_chunks.json
│       ├── policy_evaluation_results.csv
│       └── agent_evaluation_results.csv
│
├── generate_data.py
├── mock_llm.py
│
├── clean_customers.py
├── clean_products.py
├── clean_orders.py
├── build_database.py
├── validate_database.py
│
├── run_sql_queries.py
├── pandas_analysis.py
│
├── build_rfm.py
├── cluster_customers.py
├── create_segments.py
├── build_lapse_dataset.py
├── train_lapse_model.py
├── oversample_lapse_model.py
├── oversample_logistic_regression.py
├── finalize_lapse_outputs.py
│
├── part_c_classify.py
├── part_c_triage.py
├── part_c_finalize.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

Additional temporary and runtime files may be generated during execution. The structure above describes the principal project files and deliverables.

## 4. Setup and Reproduction

### 4.1 Prerequisites

- Python 3.11 or later
- Git
- Visual Studio Code or another Python-compatible IDE

### 4.2 Clone the Repository

```powershell
git clone https://github.com/poojasanthoshkumar/Instamart-Customer-Intelligence-Copilot.git
cd Instamart-Customer-Intelligence-Copilot
```

### 4.3 Create and Activate a Virtual Environment

```powershell
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4.4 Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.5 Generate the Datasets

```powershell
python generate_data.py
```

This generates the synthetic customer, product, order, support-ticket, policy-evaluation, and agent-scenario datasets.

### 4.6 Execute the Project

The four components should be executed in their respective workflow order.

**Part A – Data Foundation and Analytics**

```powershell
python clean_customers.py
python clean_products.py
python clean_orders.py
python build_database.py
python validate_database.py
python run_sql_queries.py
python pandas_analysis.py
```

**Part B – Customer Intelligence**

```powershell
python build_rfm.py
python cluster_customers.py
python create_segments.py
python build_lapse_dataset.py
python train_lapse_model.py
python oversample_lapse_model.py
python oversample_logistic_regression.py
python finalize_lapse_outputs.py
```

**Part C – Support-Ticket Triage**

```powershell
python part_c_classify.py
python part_c_triage.py
python part_c_finalize.py
```

**Part D – Policy RAG and Refund Agent**

```powershell
python .\src\part_d\policy_processor.py
python .\src\part_d\evaluate_policy.py
python .\src\part_d\rag_pipeline.py
python .\src\part_d\refund_agent.py
python .\src\part_d\evaluate_agent.py
python .\src\part_d\test_recovery.py
```

For Part D, ensure that the SQLite database, policy documents, and evaluation datasets are available before execution.

The evaluation scripts and recovery tests should be run after the relevant project components have been generated.

## 5. Part A – Data Foundation and Analytics

### 5.1 Data Cleaning

The raw datasets were audited and cleaned to establish a consistent analytical foundation.

**Customer data**

- Standardised city names and gender values.
- Identified invalid ages outside the 10–100 range.
- Imputed missing ages using city-wise median values.
- Converted signup dates into datetime format.

**Product data**

- Retained 40 product records.
- Validated product identifiers and uniqueness.
- Checked missing values and standardised product information.

**Order data**

- Converted order timestamps into datetime format.
- Removed 119 exact duplicate records.
- Retained the latest record for each order ID, producing 12,309 unique orders.
- Converted order amounts into numeric values.
- Imputed missing order amounts using product price, quantity, and discount.
- Treated negative delivery times as missing.
- Imputed missing delivery times for completed orders using product-category medians.

### 5.2 Database Construction

A SQLite database was constructed using the cleaned customer, product, and order datasets.

Database: `data/instamart.db`

Tables:

- `customers`
- `products`
- `orders`

### 5.3 Analytical Results

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Unique orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |
| Mean completed-order amount | INR 269.91 |
| Median completed-order amount | INR 135.00 |
| Upper IQR outliers | 897 |

Seven SQL queries were implemented to investigate:

1. Customer spending.
2. City-wise and category-wise revenue.
3. Above-average product prices.
4. Customers without completed orders.
5. Spending relative to customer averages.
6. City-wise customer rankings.
7. Cumulative daily revenue.

The SQL revenue calculations were cross-checked against pandas-based analytical results.

## 6. Part B – Customer Intelligence

### 6.1 RFM Analysis

RFM analysis was performed using completed orders with a snapshot date of 1 October 2025.

- **Recency:** Number of days since the customer's latest completed order.
- **Frequency:** Number of distinct completed orders.
- **Monetary:** Total completed-order spending.

The resulting RFM dataset contained 1,175 customers.

| Metric | Result |
|---|---:|
| Customers analysed | 1,175 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Mean frequency | 9.46 orders |
| Maximum frequency | 33 orders |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

Output: `outputs/rfm_table.csv`

### 6.2 K-Means Customer Segmentation

RFM variables were standardised using StandardScaler.

K-Means clustering was evaluated for K values between 2 and 8 using inertia and silhouette scores.

A four-cluster solution was selected.

| Segment | Customers | Mean Recency | Mean Frequency | Mean Monetary (INR) |
|---|---:|---:|---:|---:|
| At-Risk Customers | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Loyal Customers | 419 | 8.77 | 19.40 | 3,410.02 |
| Dormant Customers | 182 | 109.04 | 2.20 | 685.99 |

PCA was used for dimensionality reduction and visualisation. The first two components explained a combined 85.2% of the variance.

Outputs:

- `outputs/rfm_scaled.csv`
- `outputs/customer_segments.csv`
- `outputs/segment_profiles.csv`
- `outputs/pca_coordinates.csv`
- `outputs/kmeans_evaluation.csv`

### 6.3 Customer Lapse Prediction

The lapse cohort included customers with at least one completed order before 1 September 2025.

A customer was labelled as lapsed if no completed order occurred during September 2025.

| Metric | Result |
|---|---:|
| Cohort size | 1,156 |
| Not-lapsed customers | 713 |
| Lapsed customers | 443 |
| Lapsed proportion | 38.32% |
| Training rows | 924 |
| Test rows | 232 |

Historical features were constructed using data before the prediction cutoff to reduce temporal leakage.

Features included:

- Historical recency.
- Purchase frequency.
- Monetary value.
- Average discount.
- Cancellation rate.
- Average delivery time.
- Favourite product category.
- Customer demographics and tenure.

Preprocessing was fitted using training data only. Oversampling was restricted to the training partition, and final evaluation used the unchanged test partition.

### 6.4 Model Comparison

Logistic Regression and Random Forest models were evaluated using five-fold GridSearchCV with F1 scoring.

| Model | Best CV F1 | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6469 | 0.6180 | 0.6180 | 0.6180 |
| Random Forest | 0.6812 | 0.6000 | 0.6404 | 0.6196 |

The Random Forest achieved a test F1 score of 0.6196 for the lapsed class, exceeding the specified minimum of 0.55.

### 6.5 Oversampling Comparison

Random oversampling with replacement was applied to the training data.

| Logistic Regression | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.6180 | 0.6180 | 0.6180 |
| Oversampled training distribution | 0.5738 | 0.7865 | 0.6635 |

Oversampling improved recall and F1 while reducing precision, demonstrating the trade-off between identifying more at-risk customers and generating unnecessary retention interventions.

### 6.6 Feature Interpretation and Retention Prioritisation

Random Forest feature importance and Logistic Regression coefficients were exported for interpretation.

The leading Random Forest features included:

- Purchase frequency.
- Monetary value.
- Cancellation rate.
- Recency.
- Average delivery time.

A retention list was generated by ranking eligible Premium-segment test customers according to predicted lapse probability.

Output: `outputs/retention_list.csv`

### 6.7 Part B Limitations

The project uses synthetic data, and the model has been evaluated on a held-out subset rather than an independent future customer cohort.

The Premium label is based on the complete RFM snapshot, whereas lapse prediction uses an earlier historical cutoff. Therefore, the Premium-filtered retention list is illustrative and should not be interpreted as a fully leakage-free prospective evaluation.

## 7. Part C – Support-Ticket Triage

### 7.1 Objective

Part C implements an automated support-ticket processing workflow using the supplied deterministic mock language model.

The pipeline classifies customer tickets into:

- Billing.
- Delivery.
- Technical.
- Fraud.

It also incorporates input validation, PII masking, response validation, retry handling, token-budget management, and customer-aware routing.

### 7.2 Input Validation

The input dataset contained 164 tickets.

| Input Status | Tickets |
|---|---:|
| Eligible for classification | 156 |
| Empty tickets skipped | 4 |
| Overlength tickets flagged | 4 |
| Total | 164 |

Empty and overlength tickets were handled before model invocation.

### 7.3 PII Masking

The pipeline masks:

- Email addresses.
- Ten-digit Indian mobile numbers.
- Order identifiers.

Placeholders:

- `[EMAIL]`
- `[PHONE]`
- `[ORDER_ID]`

| PII Type | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order identifiers | 132 |
| Total | 278 |

Validation detected zero remaining matches for the configured PII patterns after redaction.

### 7.4 Classification and Evaluation

Two prompt strategies were evaluated:

- **Zero-shot:** Classification instructions and the required JSON schema without examples.
- **Few-shot:** Classification instructions supplemented with labelled examples.

Model responses were validated through JSON parsing and a restricted keyword fallback. Invalid outputs were flagged for human review.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| JSON classifications | 135 | 135 |
| Keyword-fallback classifications | 12 | 12 |
| Invalid outputs | 9 | 9 |
| Validly classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total response tokens | 12,364 | 20,944 |
| Tokens per classified ticket | 84.11 | 142.48 |
| Transient errors recovered | 19 | 19 |

The few-shot approach achieved higher accuracy on the selected synthetic evaluation data, with increased token consumption.

### 7.5 Token-Budget Management

A separate zero-shot run used a response-token budget of 6,000.

The pipeline classified 72 tickets and routed 80 tickets to human review after the budget threshold was reached.

The final permitted response could take total usage above the threshold because the budget was checked before each model call.

### 7.6 Customer-Aware Routing

The final routing pipeline used few-shot classification with a 100,000-token budget.

Routing rules:

1. Empty, overlength, invalid, and budget-exhausted tickets were assigned to human review.
2. Fraud tickets were routed to the security team with high priority.
3. Other validly classified tickets belonging to Premium customers were routed to the premium desk with high priority.
4. Remaining validly classified tickets were routed to the standard support queue.

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Premium desk | 11 |
| Human review | 17 |
| Total | 164 |

Output: `outputs/triage_results.csv`

### 7.7 Part C Limitations

The workflow uses a deterministic mock LLM rather than a production language model. Classification accuracy and token usage therefore reflect the supplied synthetic evaluation environment.

PII masking covers the configured patterns and does not guarantee detection of every possible sensitive-information format.

Real-world deployment would require broader privacy validation, human-reviewed evaluation datasets, monitoring, and testing against diverse customer language.

## 8. Part D – Policy RAG and Refund Agent

### 8.1 Objective

Part D implements a policy-grounded Retrieval-Augmented Generation pipeline and a simulated refund-support agent.

The workflow combines policy-document processing, TF-IDF retrieval, response generation, SQLite order lookup, refund eligibility assessment, and simulated transaction handling.

The implementation uses the supplied mock language model and does not require an external commercial LLM API.

### 8.2 Policy Processing

The policy processor reads the policy documents stored in `data/policies/` and divides their contents into smaller retrievable text chunks.

The generated knowledge base is saved as:

`outputs/part_d/policy_chunks.json`

The latest recorded policy-processing run generated **17 policy chunks**.

### 8.3 TF-IDF Retrieval

A TF-IDF-based retrieval system identifies relevant policy information using heading and content similarity.

The combined retrieval score uses:

- Heading similarity: 0.65
- Content similarity: 0.35

Query expansion improves retrieval when customer terminology differs from the wording in the policy documents.

The retrieval pipeline supports related terms such as damaged, missing, wrong, expired, duplicate, and eligibility.

### 8.4 RAG Pipeline

The RAG workflow follows these stages:

1. Receive a customer policy-related query.
2. Retrieve relevant policy chunks.
3. Rank retrieved content using similarity scores.
4. Supply the retrieved policy context to the mock language model.
5. Generate a response grounded in the available policy information.

This approach helps keep generated customer-support responses aligned with the supplied policy documents.

### 8.5 Policy Retrieval Evaluation

Policy retrieval was evaluated using 14 predefined queries.

| Metric | Result |
|---|---:|
| Evaluation queries | 14 |
| Passed queries | 14 |
| Failed queries | 0 |
| Accuracy on selected queries | 100% |

All 14 predefined policy evaluation queries passed during the recorded evaluation run.

Output:

`outputs/part_d/policy_evaluation_results.csv`

This result applies to the selected evaluation queries and does not establish general retrieval performance for all possible customer questions.

### 8.6 Refund-Support Agent

The refund-support agent integrates policy retrieval with SQLite-based order lookup.

Its workflow includes:

1. Receive a customer-support request.
2. Retrieve order information from SQLite.
3. Retrieve relevant policy information.
4. Evaluate the request against available policy conditions.
5. Calculate the eligible simulated refund.
6. Process the simulated refund operation.
7. Record execution state and handle escalation where required.

The agent uses structured tool execution and policy-grounded decision-making rather than relying exclusively on free-form language generation.

Refund operations are simulated locally. No actual financial transaction or payment-gateway interaction takes place.

### 8.7 Official Agent Evaluation

The refund agent is evaluated using seven predefined scenarios.

The evaluation covers different order and refund situations, including eligibility checks and cases requiring rejection or escalation.

Execution:

```powershell
python .\src\part_d\evaluate_agent.py
```

Output:

`outputs/part_d/agent_evaluation_results.csv`

### 8.8 Crash Recovery and Fault-Tolerance Testing

An additional isolated recovery-testing script validates the refund agent's ability to recover from interrupted execution.

Execution:

```powershell
python .\src\part_d\test_recovery.py
```

The recovery workflow validates:

- Checkpoint creation before simulated interruption.
- Simulated crash after refund processing.
- Resumption using the saved checkpoint.
- Prevention of duplicate refund ledger entries when resuming the same run.
- Safe handling of an already completed run.
- Timeout fallback.
- Maximum model-call iteration fallback.
- Token-budget fallback.

The crash-recovery test specifically simulates an interruption after a refund tool has processed a transaction and saved the checkpoint. The agent is then resumed using the saved execution state.

The test verifies that the resumed workflow completes without recording the same refund twice.

**Validation result:** All five recovery test categories passed during the recorded validation run.

The recovery tests are separate from the seven official agent evaluation scenarios.

The test script uses isolated temporary storage for runtime records. It does not modify the existing project refund ledger or persistent checkpoint files.

This validates checkpoint-based recovery and duplicate prevention within a resumed run. It does not establish distributed transaction guarantees or cross-run financial idempotency.

### 8.9 Runtime Safety and Reproducibility

The refund agent generates runtime records such as:

- Agent execution logs.
- Simulated refund transaction records.
- Persistent execution checkpoints.

These are runtime artefacts rather than required analytical outputs. They are excluded from version control through `.gitignore`.

The recovery tests use temporary storage to avoid changing the main project runtime state.

The committed source files, datasets, policy documents, model artefacts, and evaluation outputs provide the reproducible project deliverables.

## 9. Project Limitations

The following limitations should be considered when interpreting the results:

- The datasets represent synthetic Instamart-style customer and transaction information rather than real commercial records.
- Customer segmentation depends on the selected RFM features, scaling approach, and number of clusters.
- Lapse prediction was evaluated on a held-out subset of the generated cohort rather than an independent future customer population.
- The Premium-filtered retention list is illustrative and has temporal-label limitations.
- The support-ticket classification workflow uses a deterministic mock language model.
- PII masking is limited to the configured regular-expression patterns.
- TF-IDF retrieval performance depends on policy-document quality and customer-query wording.
- Refund operations are simulated and do not interact with actual payment systems.
- Checkpoint-based recovery demonstrates resumption and duplicate prevention within the same resumed run, but does not provide distributed transaction guarantees.

## 10. Conclusion

The Instamart Customer Intelligence & Support Copilot integrates data engineering, database analytics, machine learning, customer segmentation, support-ticket triage, and policy-grounded automation into a unified project.

Part A establishes a validated data foundation through cleaning, SQLite database construction, SQL queries, and pandas-based analysis.

Part B applies RFM analysis, customer segmentation, and lapse prediction to understand customer behaviour and prioritise retention opportunities.

Part C demonstrates privacy-conscious support-ticket classification, response validation, token-budget management, and customer-aware ticket routing.

Part D integrates policy-document retrieval with a simulated refund-support agent, including structured order lookup, policy-grounded decisions, checkpoint management, and recovery testing.

Together, these components demonstrate how data analytics and machine-learning techniques can support customer intelligence, operational decision-making, and automated customer-support workflows in an online grocery delivery environment.

## 11. Repository and Submission

**GitHub Repository:**

https://github.com/poojasanthoshkumar/Instamart-Customer-Intelligence-Copilot

The repository contains the project source code, datasets, policy documents, analytical outputs, model artefacts, and evaluation scripts required to examine the four project components.

**Project status:** The implementation and evaluation workflows have been developed and committed to the `main` branch. Final submission readiness should be confirmed against the assignment rubric and the committed repository contents.

---

**Author:** Pooja S. S

**Project:** Instamart Customer Intelligence & Support Copilot