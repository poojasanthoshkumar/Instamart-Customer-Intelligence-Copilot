# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

The Instamart Customer Intelligence & Support Copilot is a reproducible data analytics and machine-learning project developed using Python, pandas, NumPy, SQLite, and scikit-learn.

The project addresses four major operational challenges in an online grocery delivery platform:

- Maintaining reliable customer, product, and order data through data cleaning and database validation.
- Understanding customer purchasing behaviour through RFM analysis, customer segmentation, and lapse prediction.
- Improving support-ticket handling through automated classification, privacy protection, and customer-aware ticket routing.
- Supporting policy-based customer service through document retrieval, RAG response generation, and a simulated refund-support agent.

The project is organised into four parts:

- **Part A — Data Foundation and Analytics:** Raw-data auditing, data cleaning, SQLite database construction, SQL analysis, and exploratory data analysis.
- **Part B — Customer Intelligence:** RFM analysis, K-Means customer segmentation, lapse prediction, model evaluation, oversampling, feature interpretation, and retention prioritisation.
- **Part C — Support-Ticket Triage:** PII masking, ticket classification, response validation, token-budget management, and customer-aware ticket routing.
- **Part D — Policy RAG and Refund Agent:** Policy processing, TF-IDF-based retrieval, RAG response generation, SQLite-based order lookup, simulated refund automation, and escalation handling.

The supplied `generate_data.py` and `mock_llm.py` files are retained unchanged.

## 2. Technology Stack

- Python 3.11 or later
- pandas
- NumPy
- SQLite 3.25 or later
- scikit-learn
- joblib
- Matplotlib
- VS Code
- Git and GitHub

## 3. Project Structure

```text
Instamart_Copilot/
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

The `outputs/` directory contains generated analytical results, model artefacts, and evaluation reports. Additional temporary and runtime files may be generated during execution.

## 4. Setup and Reproduction

### 4.1 Prerequisites

Install Python 3.11 or later and Git.

Clone the repository:

```powershell
git clone https://github.com/poojasanthoshkumar/Instamart-Customer-Intelligence-Copilot.git

cd Instamart-Customer-Intelligence-Copilot
```

### 4.2 Create a virtual environment

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 4.3 Install dependencies

```powershell
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

### 4.4 Generate raw data

```powershell
python generate_data.py
```

The supplied data-generation script creates the synthetic customer, product, order, ticket, policy-evaluation, and agent-scenario datasets.

The supplied `generate_data.py` and `mock_llm.py` files are retained unchanged.

### 4.5 Data Cleaning and Database Construction

Run the cleaning scripts from the project root in the following order.

**Step 1: Clean customer data**

```powershell
python clean_customers.py
```

Standardises city and gender values, identifies invalid ages, imputes missing ages using city-wise medians, and standardises signup dates.

**Step 2: Clean product data**

```powershell
python clean_products.py
```

Standardises product information and validates product identifiers and prices.

**Step 3: Clean order data**

```powershell
python clean_orders.py
```

Removes duplicate records, retains the latest record for each order ID, handles missing order amounts, and processes delivery-time values.

**Step 4: Build the SQLite database**

```powershell
python build_database.py
```

Creates `data/instamart.db` using the cleaned customer, product, and order datasets.

**Step 5: Validate the database**

```powershell
python validate_database.py
```

Checks database structure, records, and data integrity.

### 4.6 Part A — Database Analytics and Exploratory Analysis

After database construction and validation, execute the SQL and pandas analyses.

**SQL analysis**

```powershell
python run_sql_queries.py
```

Executes the analytical SQL queries defined in the project.

**Pandas analysis**

```powershell
python pandas_analysis.py
```

Performs complementary data analysis and cross-checks the SQL-derived results.

The SQL and pandas analyses cover customer spending, city-wise revenue, product pricing, customer purchase behaviour, customer rankings, and cumulative revenue patterns.

### 4.7 Part B — Customer Intelligence

Execute the scripts from the project root in the following order.

**Step 1: Build the RFM table**

```powershell
python build_rfm.py
```

Builds the customer-level RFM dataset using completed orders and the snapshot date of 1 October 2025.

The three RFM measures are:

- **Recency:** Whole days since the customer's most recent completed order.
- **Frequency:** Number of distinct completed orders.
- **Monetary:** Sum of completed-order amounts.

Output:

`outputs/rfm_table.csv`

**Step 2: Evaluate K-Means clustering**

```powershell
python cluster_customers.py
```

Standardises RFM features and evaluates cluster counts from K=2 to K=8 using inertia and silhouette scores.

Outputs:

- `outputs/kmeans_evaluation.csv`
- `outputs/rfm_scaled.csv`

**Step 3: Create customer segments**

```powershell
python create_segments.py
```

Fits a four-cluster K-Means model, profiles customer groups, assigns business-oriented segment labels, and generates PCA coordinates for visualisation.

Outputs:

- `outputs/customer_segments.csv`
- `outputs/segment_profiles.csv`
- `outputs/pca_coordinates.csv`

The segments are assigned descriptive labels based on their spending, purchase frequency, and recency characteristics.

**Step 4: Build the customer lapse dataset**

```powershell
python build_lapse_dataset.py
```

Creates a leakage-aware dataset using historical customer information before 1 September 2025.

Customer lapse labels are determined using completed-order activity during September 2025.

Historical purchasing behaviour, cancellation rate, average discount, delivery time, customer demographics, and favourite product category are used as predictive features.

Output:

`outputs/lapse_dataset.csv`

**Step 5: Train and evaluate baseline models**

```powershell
python train_lapse_model.py
```

Trains and compares logistic regression and random forest models for customer lapse prediction.

The workflow uses a stratified train-test split and evaluates model performance using classification metrics.

**Step 6: Evaluate oversampling approaches**

```powershell
python oversample_lapse_model.py

python oversample_logistic_regression.py
```

Evaluates the effect of oversampling on model performance and compares the resulting models with their baseline counterparts.

Oversampling is restricted to training data to avoid introducing synthetic observations into the test set.

**Step 7: Finalise model verification and retention prioritisation**

```powershell
python finalize_lapse_outputs.py
```

Verifies saved model reload consistency and generates a ranked retention list of high-risk Premium customers using the selected oversampled model.

Output:

`outputs/retention_list.csv`

The Part B workflow generates customer-level analytical datasets, segmentation results, model artefacts, evaluation reports, and retention recommendations under `outputs/`.

### 4.8 Part C — Support-Ticket Triage

Execute the three support-ticket processing scripts from the project root in the following order.

**Step 1: Ticket classification**

```powershell
python part_c_classify.py
```

Processes support tickets and performs automated ticket classification.

**Step 2: Ticket triage**

```powershell
python part_c_triage.py
```

Performs customer-aware ticket processing and evaluates different classification and response-generation settings.

**Step 3: Finalise ticket processing**

```powershell
python part_c_finalize.py
```

Validates and finalises ticket-processing outputs, including routing and audit information.

The Part C workflow addresses:

- Personally identifiable information masking.
- Automated ticket classification.
- Zero-shot and few-shot processing comparisons.
- Token-budget management.
- Response validation.
- Customer-aware ticket prioritisation and routing.

Generated outputs include:

- `outputs/triage_results.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_input_audit.csv`

### 4.9 Part D — Policy RAG and Refund Agent

Run the following commands from the project root with the virtual environment activated.

Ensure that `data/instamart.db`, the policy documents, and the predefined evaluation datasets are available.

**Step 1: Process policy documents**

```powershell
python .\src\part_d\policy_processor.py
```

Processes the policy documents and converts their contents into retrievable text chunks.

Output:

`outputs/part_d/policy_chunks.json`

**Step 2: Evaluate policy retrieval**

```powershell
python .\src\part_d\evaluate_policy.py
```

Evaluates policy retrieval using 14 predefined policy queries.

Output:

`outputs/part_d/policy_evaluation_results.csv`

**Step 3: Run the RAG pipeline**

```powershell
python .\src\part_d\rag_pipeline.py
```

Retrieves relevant policy information using TF-IDF-based similarity and generates responses through the supplied mock language model.

**Step 4: Run the refund-support agent**

```powershell
python .\src\part_d\refund_agent.py
```

Demonstrates:

- Customer order lookup using SQLite.
- Retrieval of relevant refund-policy information.
- Order eligibility assessment.
- Refund calculation.
- Simulated refund processing.
- Escalation handling.

The refund-support agent uses the project database and policy information to guide its decisions.

**Step 5: Evaluate the refund agent**

```powershell
python .\src\part_d\evaluate_agent.py
```

Evaluates the refund agent using seven predefined scenarios.

Output:

`outputs/part_d/agent_evaluation_results.csv`

The Part D workflow uses a simulated refund-processing environment. It does not initiate actual financial transactions.

**Step 6: Validate crash recovery and safety limits**

```powershell
python .\src\part_d\test_recovery.py
```

Runs isolated recovery and fault-tolerance tests for the refund-support agent.

The recovery test validates:

- Simulated interruption immediately after refund processing.
- Checkpoint creation before interruption.
- Successful resumption of the interrupted agent run.
- Prevention of duplicate refund ledger entries when resuming the same run.
- Safe handling of an already completed run.
- Timeout fallback.
- Maximum model-call iteration fallback.
- Token-budget fallback.

The crash-recovery test verifies that the agent can resume from its saved checkpoint and complete the refund workflow without recording the same refund twice.

All five recovery test categories passed during validation. The recovery tests use a temporary directory for runtime records and automatically remove the temporary test files after execution.

These recovery tests are separate from the seven predefined official agent evaluation scenarios.

### 4.10 Runtime Files and Reproducibility

The project generates certain runtime records during agent execution, including:

- Agent execution logs.
- Refund transaction simulation records.
- Persistent agent checkpoint files.

These files are runtime artefacts rather than required analytical result tables. They are excluded from version control through `.gitignore` to avoid committing unnecessary execution state.

The recovery test redirects runtime paths to temporary storage, ensuring that its simulated refund transactions do not modify the existing project refund ledger or checkpoint files.

The committed analytical outputs, policy-processing results, evaluation reports, and model artefacts provide reference results for the project.

## 5. Part A — Data Foundation and Analytics

### 5.1 Data Cleaning

**Customers**

- Standardised city names by trimming whitespace and applying title case.
- Standardised gender values to male, female, or unknown.
- Treated ages outside the 10–100 range as invalid and set them to missing.
- Imputed missing ages using the median age within each city.
- Converted signup dates to datetime format.

**Products**

- Retained 40 product records and verified product ID uniqueness.
- Standardised product identifiers, product names, and category values.
- Validated product prices and checked for missing or invalid values.

**Orders**

- Converted order timestamps to datetime.
- Removed 119 exact duplicate records.
- Retained the latest record for each order ID, producing 12,309 unique orders.
- Converted order amounts to numeric values and treated invalid amounts as missing.
- Replaced missing order amounts using product price, quantity, and discount:

  `price_inr × quantity × (1 − discount_pct / 100)`

- Treated negative delivery times as missing.
- Imputed missing delivery times for completed orders using the median for the corresponding product category.
- Kept delivery times missing for cancelled orders.

### 5.2 Database Design

The SQLite database, `data/instamart.db`, contains three tables:

- `customers`
- `products`
- `orders`

The database is constructed from the cleaned CSV datasets. Order timestamps are stored as text in `YYYY-MM-DD HH:MM:SS` format.

### 5.3 Key Results

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Unique orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |
| Completed-order mean amount | INR 269.91 |
| Completed-order median amount | INR 135.00 |
| Upper IQR outliers | 897 |

The pandas pivot-table revenue total matches the SQL revenue total.

The seven SQL queries cover:

1. Customer spending.
2. City and category revenue.
3. Above-average product prices.
4. Customers without completed orders.
5. Spending relative to the customer average.
6. City-wise customer rankings.
7. Cumulative daily revenue.

### 5.4 Reproducibility and Integrity

The supplied data-generation and mock-LLM scripts are retained unchanged.

The cleaning and database artefacts provide the data foundation for subsequent analytical tasks. Validation scripts check key row counts, missing-value rules, revenue totals, database tables, and query outputs.

Part A establishes the cleaned data layer, SQLite database, SQL analytics, pandas cross-checks, and exploratory analysis used by subsequent parts.

## 6. Part B — Customer Intelligence

### 6.1 RFM Analysis

RFM analysis was performed using completed orders as of the snapshot date `2025-10-01 00:00:00`.

The resulting RFM dataset contains 1,175 customers.

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

### 6.2 K-Means Clustering

RFM features were standardised before applying K-Means clustering.

Cluster counts from K=2 to K=8 were evaluated using:

- Inertia to examine within-cluster variation.
- Silhouette score to assess cluster separation.

A four-cluster solution was used for customer segmentation.

Outputs:

- `outputs/kmeans_evaluation.csv`
- `outputs/rfm_scaled.csv`

### 6.3 Customer Segmentation

A four-cluster K-Means model was fitted using standardised recency, frequency, and monetary values.

Customer groups were profiled using mean and median RFM values.

Business-oriented segment labels were assigned to support customer engagement and retention strategies:

- Premium customers.
- Loyal customers.
- At-risk customers.
- Dormant customers.
- Occasional customers, where applicable to the profile classification.

The highest-spending cluster was designated as the Premium segment. Other segment labels were assigned using recency and frequency characteristics.

Recommended actions include personalised loyalty benefits, re-engagement campaigns, win-back offers, and repeat-purchase incentives.

PCA was used to generate two-dimensional coordinates for visualising the customer clusters.

Outputs:

- `outputs/customer_segments.csv`
- `outputs/segment_profiles.csv`
- `outputs/pca_coordinates.csv`

### 6.4 Customer Lapse Prediction

A leakage-aware customer lapse dataset was constructed using a temporal feature and label design.

Historical customer behaviour before 1 September 2025 was used to build predictive features. Customer lapse was labelled based on the absence of completed orders during September 2025.

Predictive features included:

- Recency.
- Purchase frequency.
- Monetary value.
- Average discount.
- Cancellation rate.
- Average delivery time.
- Customer age.
- Customer tenure.
- City.
- Gender.
- Favourite product category.

The resulting dataset was used to train and evaluate customer lapse prediction models.

Output: `outputs/lapse_dataset.csv`

### 6.5 Model Training and Evaluation

Logistic regression and random forest models were trained and compared.

A stratified train-test split was used to maintain the class distribution across training and testing datasets.

Oversampling experiments were performed to examine their effect on the classification of the minority lapse class. Synthetic oversampled observations were restricted to the training data.

Model evaluation and interpretation included classification performance comparisons, logistic regression coefficients, and random forest feature importance.

Generated artefacts include:

- `outputs/lapse_model.joblib`
- `outputs/lapse_model_oversampled.joblib`
- `outputs/lapse_model_comparison.json`
- `outputs/logistic_regression_coefficients.csv`
- `outputs/random_forest_feature_importance.csv`

### 6.6 Retention Prioritisation

The finalisation workflow verifies saved model consistency and identifies Premium customers with elevated lapse probabilities.

The selected oversampled model is used to calculate lapse probabilities for the eligible Premium customers in the test set.

The ten highest-ranked retention candidates are saved for targeted customer engagement.

Output: `outputs/retention_list.csv`

## 7. Part C — Support-Ticket Triage

### 7.1 Ticket Classification

The support-ticket workflow processes customer complaints and classifies them into relevant support categories.

The classification process uses the supplied mock language model and predefined ticket data.

### 7.2 Privacy Protection

Personally identifiable information is masked during ticket processing to reduce unnecessary exposure of customer information.

An input audit is generated to support examination of the ticket-processing workflow.

### 7.3 Response Processing and Triage

The workflow evaluates different response-generation approaches, including zero-shot and few-shot processing.

Token-budget management is incorporated to control response-generation inputs.

Customer-aware ticket routing and response validation are used to support more structured support operations.

### 7.4 Outputs

The Part C workflow generates:

- `outputs/triage_results.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_input_audit.csv`

These outputs provide the classification results, triage information, processing comparisons, and audit records.

## 8. Part D — Policy RAG and Refund Agent

### 8.1 Policy Document Processing

Policy documents are processed into text chunks to support retrieval during customer-support interactions.

The processed policy information is saved in:

`outputs/part_d/policy_chunks.json`

### 8.2 Retrieval-Augmented Generation

The RAG pipeline uses TF-IDF-based vectorisation and cosine similarity to identify policy passages relevant to a customer query.

Relevant retrieved information is supplied to the mock language model to generate a policy-grounded response.

Query expansion is used to improve retrieval when customer terminology differs from the wording used in policy documents.

### 8.3 Policy Retrieval Evaluation

Policy retrieval is evaluated using 14 predefined queries.

The evaluation checks whether relevant policy information can be retrieved for different customer-support situations.

Output:

`outputs/part_d/policy_evaluation_results.csv`

### 8.4 Refund-Support Agent

The refund-support agent integrates policy retrieval with SQLite-based order lookup.

The agent examines order details, retrieves relevant policy information, evaluates refund eligibility, and calculates simulated refund amounts.

The workflow also includes escalation handling for situations that require additional review.

Refund operations are simulated locally and do not represent real customer transactions.

### 8.5 Agent Evaluation

The refund-support agent is evaluated using seven predefined scenarios.

The evaluation assesses whether the agent follows the expected policy and decision-making behaviour for different order situations.

Output:

`outputs/part_d/agent_evaluation_results.csv`

### 8.6 Crash Recovery and Fault Tolerance

An additional isolated recovery-testing script validates the agent's ability to recover from interrupted execution.

The test simulates a crash immediately after the refund tool has processed a simulated transaction and saved its checkpoint.

The same run is then resumed using the saved checkpoint. The test verifies successful completion and ensures that the refund ledger does not contain duplicate entries for the same run and order.

Additional checks cover:

- Resumption of an already completed run.
- Timeout handling.
- Maximum model-call limit.
- Token-budget limit.

All five recovery test categories passed during the recorded validation run.

The recovery tests use temporary storage and do not modify the existing project refund ledger, agent execution logs, or persistent checkpoints.

This validates checkpoint-based recovery and duplicate prevention within a resumed run. It does not establish exactly-once transaction guarantees across separate runs or interruptions during an incomplete file write.

### 8.7 Part D Reproducibility

The Part D workflow uses:

- SQLite for order lookup.
- Policy documents for business rules.
- TF-IDF and cosine similarity for retrieval.
- The supplied mock language model for response generation.
- Local runtime records for simulated agent execution.
- Checkpoint files for interrupted-run recovery.
- Isolated temporary storage for recovery testing.

The workflow is designed to demonstrate an offline policy-grounded customer-support system without requiring an external commercial language-model API.

## 9. Project Limitations

- The project uses synthetic Instamart-style datasets rather than actual commercial customer records.
- Customer lapse predictions depend on the defined historical feature period and September 2025 labelling window.
- K-Means segmentation is dependent on the selected features, scaling approach, and cluster count.
- The ticket-classification and response-generation workflow uses the supplied mock language model.
- Policy retrieval performance depends on the quality and wording of the available policy documents and customer queries.
- Refund processing is simulated and does not interact with an actual payment gateway or financial system.
- Checkpoint-based recovery protects against repeated processing within the same resumed run, but does not provide distributed transaction guarantees or cross-run financial idempotency.

## 10. Conclusion

The Instamart Customer Intelligence & Support Copilot integrates data engineering, database analytics, machine learning, customer segmentation, support-ticket triage, and policy-grounded automation into a unified project.

Part A establishes a validated data foundation and analytical database. Part B applies customer analytics and predictive modelling to identify purchasing patterns and prioritise retention opportunities. Part C demonstrates structured and privacy-conscious support-ticket processing. Part D integrates policy retrieval with a simulated refund-support agent and checkpoint-based recovery.

Together, these components demonstrate how data analytics and machine-learning techniques can support customer intelligence, operational decision-making, and automated customer-support workflows in an online grocery delivery environment.

## 11. Repository

**GitHub:** https://github.com/poojasanthoshkumar/Instamart-Customer-Intelligence-Copilot

The repository contains the project source code, datasets, policy documents, analytical outputs, model artefacts, and evaluation results required to examine the implemented workflows.