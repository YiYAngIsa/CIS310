# Customer & Sales Analytics — Streamlit Dashboard

An interactive data application based on the three supplied CSV files:

- `data/orders.csv` — 9,994 order lines, Windows-1252 encoded
- `data/customers(1).csv` — 4,910 customer-location records (793 unique customers)
- `data/returns.csv` — 296 returned order IDs

## What's included

- KPI summary: total sales, profit, profit margin, distinct orders, order return rate
- **Multi-series line chart:** monthly sales by product category (and an overall sales line)
- **Q4 shading:** visually emphasize October–December in each year
- **Pie chart:** percentage contribution to aggregate category profit, where values are positive
- **Box plot:** distribution of profit per order line by category (optional outliers)
- **Bar chart:** order-level return rate for each product category
- **Bar chart:** total profit by customer segment
- **Regional sales:** confirmed sales by region, with unassignable sales explicitly labeled
- Inclusive date range filter and selectors for product categories and customer segments
- Source quality checks and downloadable filtered customer–order data
- Optional replacement of input CSVs through sidebar file uploads

## Important data handling

1. A customer can appear in multiple customer-location records. The application **does not directly join the raw 4,910-row customers table** to orders. Instead it uses one consistent customer profile per `Customer ID`, keeping `Segment` and `Country` and only assigning `Region` when all records for that customer have the same region. Otherwise geography is labeled unassigned.
2. The `Returned` field in returns is **order-level**, not line-item-level. A returned order containing multiple categories counts toward each category's **order-return incidence**. This is not proof all its line items were returned.
3. `orders.csv` contains an invalid order date (`2/29/2025` for `CA-2025-111283`). It is parsed as a missing date and excluded from date-filtered charts and KPIs; the source issue appears in the Data Quality tab.
4. The app computes a **consistent date-filtered subset** for all charts. Hence the default all-valid-dates profit totals can differ slightly from a prior offline chart that used every row regardless of invalid dates.
5. The dataset contains records with future dates relative to parts of 2026. They are displayed as supplied, not represented as independently verified transactions.

## Project structure

```text
streamlit_customer_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    ├── orders.csv
    ├── customers(1).csv
    └── returns.csv
```

## Run locally

Recommended: Python 3.12.

```bash
# From this folder
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens in your browser, normally at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud — recommended for an easily shared site

1. Create a new GitHub repository. Keep **all the project files and their folders** in the repository, including `app.py`, `requirements.txt`, and `data/` if you have permission to publish the data.
2. Visit **https://share.streamlit.io** and sign in with GitHub. Connect/authorize your repository if needed.
3. Choose **Create app → Deploy a public app from GitHub** (the exact wording may change). Select your repository, your branch (usually `main`), and **Main file path:** `app.py`.
4. In Advanced settings, select **Python 3.12** if available; no secrets are needed for the bundled CSV files.
5. Click **Deploy**. When it finishes, the service gives you a sharable address ending in **`.streamlit.app`**.
6. Edit the GitHub repository to update the application. The service rebuilds or refreshes the deployed app automatically.

**Privacy warning:** If you make both your repository and your app public, data bundled in the repository and downloadable from the dashboard may be accessible to others. Don't publish customer-identifying, private, or proprietary information without authorization. You can instead omit the `data/` directory from the repository and require users to upload the three CSVs in the sidebar. For restricted dashboards, use app sharing/privacy controls or a private deployment.

The app can run without bundled `data/` files if users upload the missing three files in the sidebar. If you omit data files, the site launches to an upload prompt rather than a populated dashboard.

Official references:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app

## Alternative deployment: Docker / your own cloud server

Build and run using the included `Dockerfile`:

```bash
docker build -t customer-analytics-streamlit .
docker run --rm -p 8501:8501 customer-analytics-streamlit
```

Then visit `http://localhost:8501`. For a public website, put the container behind an HTTPS-enabled cloud host and use the host's configured domain, network/firewall, and authentication rules. See https://docs.streamlit.io/deploy/tutorials/docker.

## Alternative deployment: Streamlit in Snowflake

If the relevant data is already hosted in Snowflake, Streamlit in Snowflake supports managed internal apps with role-based access controls. This code uses local CSVs, so a Snowflake deployment would typically involve adapting the data-loading layer to use Snowflake tables and following Snowflake's app publishing process:
https://docs.streamlit.io/deploy/snowflake

## Dependency versions

`requirements.txt` pins Streamlit, pandas, and Plotly to known releases. No API keys or external service accounts are required for the bundled file data.
