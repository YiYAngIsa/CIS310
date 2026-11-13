"""Interactive sales, customer, and returns dashboard.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

The app uses data/orders.csv, data/customers(1).csv, and data/returns.csv
unless users provide alternative CSVs through the sidebar.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------- App configuration --------------------------- #
st.set_page_config(
    page_title="Customer & Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CATEGORY_ORDER = ["Furniture", "Office Supplies", "Technology"]
CATEGORY_COLORS = {
    "Furniture": "#447DB4",
    "Office Supplies": "#E58B3D",
    "Technology": "#269672",
}
SEGMENT_ORDER = ["Consumer", "Corporate", "Home Office"]
REGION_ORDER = ["West", "Central", "East", "South"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #F3F6FA; border-radius: 12px; padding: 14px;}
    .mini-note {font-size:0.88rem; color:#536375;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Data preparation --------------------------- #
def _read_csv(contents: bytes, label: str, *, windows_encoding: bool = False) -> pd.DataFrame:
    """Read CSV preserving identifier strings and handling source encoding."""
    encodings = ["cp1252", "utf-8-sig"] if windows_encoding else ["utf-8-sig", "cp1252"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(
                BytesIO(contents),
                encoding=encoding,
                dtype={
                    "Customer ID": "string",
                    "Order ID": "string",
                    "Product ID": "string",
                    "Postal Code": "string",
                },
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode {label}: {last_error}")


def _require_columns(frame: pd.DataFrame, expected: set[str], filename: str) -> None:
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {', '.join(sorted(missing))}")


@st.cache_data(show_spinner=False)
def prepare_data(order_bytes: bytes, customer_bytes: bytes, returns_bytes: bytes):
    """Validate, normalize and join data without multiplying the order rows."""
    orders = _read_csv(order_bytes, "orders.csv", windows_encoding=True)
    customers = _read_csv(customer_bytes, "customers(1).csv")
    returns = _read_csv(returns_bytes, "returns.csv")

    _require_columns(
        orders,
        {
            "Row ID", "Order ID", "Order Date", "Ship Date", "Customer ID",
            "Category", "Sales", "Quantity", "Discount", "Profit",
        },
        "orders.csv",
    )
    _require_columns(
        customers,
        {"Customer ID", "Segment", "Country", "Region", "City", "State", "Postal Code"},
        "customers(1).csv",
    )
    _require_columns(returns, {"Order ID", "Returned"}, "returns.csv")

    for frame, id_fields in (
        (orders, ("Customer ID", "Order ID")),
        (customers, ("Customer ID",)),
        (returns, ("Order ID",)),
    ):
        for column in id_fields:
            frame[column] = frame[column].astype("string").str.strip()

    for column in ("Sales", "Profit", "Discount", "Quantity"):
        orders[column] = pd.to_numeric(orders[column], errors="coerce")

    # Explicit source date format; 2/29/2025 is invalid and becomes NaT.
    orders["Order Date Parsed"] = pd.to_datetime(
        orders["Order Date"], format="%m/%d/%Y", errors="coerce"
    )
    orders["Ship Date Parsed"] = pd.to_datetime(
        orders["Ship Date"], format="%m/%d/%Y", errors="coerce"
    )
    date_issues = orders.loc[
        orders["Order Date Parsed"].isna() | orders["Ship Date Parsed"].isna(),
        ["Row ID", "Order ID", "Order Date", "Ship Date"],
    ].copy()

    if orders["Customer ID"].isna().any() or customers["Customer ID"].isna().any():
        raise ValueError("Missing Customer IDs prevent reliable joining.")
    if orders["Order ID"].isna().any():
        raise ValueError("Missing Order IDs prevent reliable return matching.")

    # Customer IDs occur in several geographic locations. These attributes
    # must not be joined as if there were only one location per customer.
    grouped = customers.groupby("Customer ID", dropna=False)
    segment_counts = grouped["Segment"].nunique(dropna=False)
    country_counts = grouped["Country"].nunique(dropna=False)
    conflicting = (segment_counts > 1) | (country_counts > 1)
    if conflicting.any():
        raise ValueError(
            f"{int(conflicting.sum())} customers have conflicting Segment or Country values. "
            "Resolve those inconsistencies before analysis."
        )

    profiles = (
        customers[["Customer ID", "Segment", "Country"]]
        .drop_duplicates(subset="Customer ID")
        .copy()
    )
    region_counts = grouped["Region"].nunique(dropna=False)
    single_region = grouped["Region"].first().where(region_counts.eq(1))
    profiles["Region (confirmed)"] = profiles["Customer ID"].map(single_region)
    profiles["Region count"] = profiles["Customer ID"].map(region_counts).astype("Int64")

    joined = orders.merge(
        profiles,
        on="Customer ID",
        how="left",
        validate="many_to_one",
        indicator="Customer Join",
    )
    if len(joined) != len(orders):
        raise AssertionError("Join unexpectedly changed the number of order lines")

    returned_ids = set(
        returns.loc[
            returns["Returned"].astype("string").str.strip().str.lower().eq("yes"),
            "Order ID",
        ].dropna().unique()
    )
    joined["Returned"] = joined["Order ID"].isin(returned_ids)
    joined["Customer Matched"] = joined["Customer Join"].eq("both")
    joined["Region Display"] = joined["Region (confirmed)"].fillna(
        "Unassigned (multiple or missing regions)"
    )

    order_ids = set(orders["Order ID"].dropna().unique())
    meta = {
        "source_order_lines": len(orders),
        "unique_orders": orders["Order ID"].nunique(),
        "unique_customers": customers["Customer ID"].nunique(),
        "customer_location_rows": len(customers),
        "ambiguous_region_customers": int(region_counts.gt(1).sum()),
        "unmatched_order_lines": int((~joined["Customer Matched"]).sum()),
        "unmatched_unique_orders": int(joined.loc[~joined["Customer Matched"], "Order ID"].nunique()),
        "returns_not_in_orders": len(returned_ids - order_ids),
        "duplicated_row_ids": int(orders["Row ID"].duplicated().sum()),
        "invalid_order_dates": int(orders["Order Date Parsed"].isna().sum()),
        "invalid_ship_dates": int(orders["Ship Date Parsed"].isna().sum()),
    }
    return joined, customers, profiles, date_issues, meta


def _read_bytes(uploaded, fallback_filename: str) -> bytes | None:
    if uploaded is not None:
        return uploaded.getvalue()
    fallback = DATA_DIR / fallback_filename
    if fallback.exists():
        return fallback.read_bytes()
    return None


def money(value: float) -> str:
    return f"${value:,.2f}"


def style_figure(fig: go.Figure, *, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=20, t=60, b=24),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E9EDF2", zeroline=False)
    return fig


def shade_q4(fig: go.Figure, years: list[int]) -> go.Figure:
    for year in years:
        fig.add_vrect(
            x0=f"{year}-10-01",
            x1=f"{year + 1}-01-01",
            fillcolor="#E8B66B",
            opacity=0.18,
            line_width=0,
            layer="below",
        )
    return fig


def aggregate_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Month", "Category", "Sales"])
    monthly = frame.copy()
    monthly["Month"] = monthly["Order Date Parsed"].dt.to_period("M").dt.to_timestamp()
    return monthly.groupby(["Month", "Category"], as_index=False)["Sales"].sum()


def category_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rate of returned ORDER IDs among order IDs containing each category."""
    dedup = frame[["Category", "Order ID", "Returned"]].drop_duplicates(
        ["Category", "Order ID"]
    )
    result = (
        dedup.groupby("Category", as_index=False)
        .agg(Orders=("Order ID", "nunique"), Returned_Orders=("Returned", "sum"))
    )
    result["Return Rate (%)"] = (100 * result["Returned_Orders"] / result["Orders"]).round(2)
    return result.sort_values("Return Rate (%)", ascending=False)


def report_csv(frame: pd.DataFrame) -> bytes:
    safe = frame.drop(columns=["Customer Join"], errors="ignore")
    return safe.to_csv(index=False, date_format="%m/%d/%Y").encode("utf-8-sig")


# ------------------------ Custom chart builder helpers -------------------- #
CUSTOM_CHART_TYPES = (
    "Scatter plot", "Line chart", "Bar chart", "Histogram", "Box plot",
    "Violin plot", "Pie chart",
)
CUSTOM_DATA_VIEWS = (
    "Order line items", "Unique orders", "Customer summaries", "Monthly by category",
)
AGGREGATIONS = ("Sum", "Average", "Median", "Count rows", "Distinct orders")
TIME_GRAINS = ("Day", "Week", "Month", "Quarter", "Year")


def prepare_custom_view(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    """Create safe, filter-aware chart datasets at clearly defined levels."""
    work = frame.copy()
    # Leave the original source and shared filtered frame untouched.
    work["Order Date"] = work["Order Date Parsed"]
    work["Ship Date"] = work["Ship Date Parsed"]
    work["Region"] = work["Region Display"]
    work["Returned"] = work["Returned"].map({True: "Yes", False: "No"})

    if view == "Order line items":
        cols = [
            "Row ID", "Order ID", "Customer ID", "Order Date", "Ship Date",
            "Ship Mode", "Category", "Sub-Category", "Product ID", "Product Name",
            "Segment", "Country", "Region", "Returned", "Sales", "Profit",
            "Quantity", "Discount",
        ]
        return work[[c for c in cols if c in work]].copy()

    if view == "Unique orders":
        # One order can have products in several categories; don't randomly
        # choose one category for it.
        result = (
            work.groupby("Order ID", as_index=False, dropna=False)
            .agg(
                **{
                    "Customer ID": ("Customer ID", "first"),
                    "Order Date": ("Order Date", "min"),
                    "Ship Date": ("Ship Date", "min"),
                    "Category": (
                        "Category",
                        lambda values: values.iloc[0]
                        if values.nunique(dropna=False) == 1 else "Multiple categories",
                    ),
                    "Segment": ("Segment", "first"),
                    "Region": ("Region", "first"),
                    "Returned": ("Returned", "first"),
                    "Sales": ("Sales", "sum"),
                    "Profit": ("Profit", "sum"),
                    "Quantity": ("Quantity", "sum"),
                    "Discount": ("Discount", "mean"),
                    "Order Lines": ("Order ID", "size"),
                }
            )
        )
        return result

    if view == "Customer summaries":
        return (
            work.groupby("Customer ID", as_index=False, dropna=False)
            .agg(
                **{
                    "Segment": ("Segment", "first"),
                    "Country": ("Country", "first"),
                    "Region": ("Region", "first"),
                    "Sales": ("Sales", "sum"),
                    "Profit": ("Profit", "sum"),
                    "Quantity": ("Quantity", "sum"),
                    "Discount": ("Discount", "mean"),
                    "Order Count": ("Order ID", "nunique"),
                    "Order Lines": ("Order ID", "size"),
                }
            )
        )

    if view == "Monthly by category":
        work["Month"] = work["Order Date"].dt.to_period("M").dt.to_timestamp()
        return (
            work.groupby(["Month", "Category"], as_index=False, dropna=False)
            .agg(
                **{
                    "Sales": ("Sales", "sum"),
                    "Profit": ("Profit", "sum"),
                    "Quantity": ("Quantity", "sum"),
                    "Order Count": ("Order ID", "nunique"),
                    "Order Lines": ("Order ID", "size"),
                }
            )
        )

    raise ValueError(f"Unknown chart data view: {view}")


def classify_chart_columns(frame: pd.DataFrame):
    """Return numeric, temporal and categorical columns for type-safe inputs."""
    numeric = [
        c for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c])
        and not pd.api.types.is_bool_dtype(frame[c])
    ]
    temporal = [
        c for c in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[c])
    ]
    categorical = [c for c in frame.columns if c not in numeric + temporal]
    return numeric, temporal, categorical


def custom_chart_default(options, preferred: str) -> int:
    return list(options).index(preferred) if preferred in options else 0


def group_chart_data(
    frame: pd.DataFrame,
    x: str,
    y: str,
    group: str | None,
    aggregation: str,
    date_grain: str,
) -> pd.DataFrame:
    """Aggregate explicit X/(color) groups; don't join raw lines in a time plot."""
    table = frame.copy()
    if pd.api.types.is_datetime64_any_dtype(table[x]):
        freqs = {"Day": "D", "Week": "W-SUN", "Month": "M", "Quarter": "Q", "Year": "Y"}
        table["Chart X"] = table[x].dt.to_period(freqs[date_grain]).dt.start_time
    else:
        table["Chart X"] = table[x]

    group_columns = ["Chart X"]
    if group is not None and group != x:
        table["Chart Group"] = table[group]
        group_columns.append("Chart Group")

    if aggregation == "Count rows":
        plotted = table.groupby(group_columns, dropna=False, observed=True).size().reset_index(name="Value")
    elif aggregation == "Distinct orders":
        if "Order ID" not in table:
            raise ValueError("Distinct orders is available only for order-level datasets.")
        plotted = (
            table.groupby(group_columns, dropna=False, observed=True)["Order ID"]
            .nunique().reset_index(name="Value")
        )
    else:
        rule = {"Sum": "sum", "Average": "mean", "Median": "median"}[aggregation]
        plotted = (
            table.groupby(group_columns, dropna=False, observed=True)[y]
            .agg(rule).reset_index(name="Value")
        )
    return plotted.sort_values("Chart X")


def make_custom_chart(
    frame: pd.DataFrame,
    kind: str,
    *,
    x: str,
    y: str | None = None,
    group: str | None = None,
    size: str | None = None,
    aggregation: str = "Sum",
    date_grain: str = "Month",
    bins: int = 25,
    top_n: int = 20,
    orientation: str = "Vertical",
    highlight_quarter: bool = False,
) -> tuple[go.Figure | None, pd.DataFrame, str | None]:
    """Build a Plotly figure plus the underlying plotted data and any warning.

    These functions are independent of Streamlit widgets so they can be tested.
    """
    if frame.empty:
        return None, frame.copy(), "No records match the selected filters."
    if kind == "Scatter plot":
        required = [x, y] + ([group] if group else []) + ([size] if size else [])
        raw = frame.dropna(subset=required).copy()
        if raw.empty:
            return None, raw, "No complete records for the selected variables."
        if size and (raw[size] < 0).any():
            return None, raw, "Bubble size must be nonnegative. Choose a different size variable."
        fig = px.scatter(
            raw, x=x, y=y, color=group, size=size, opacity=0.66,
            hover_data=[c for c in ("Order ID", "Customer ID") if c in raw and c not in required],
            labels={x: x, y: y}, render_mode="auto",
        )
        return style_figure(fig, height=540), raw, None

    if kind == "Histogram":
        raw = frame.dropna(subset=[x] + ([group] if group else [])).copy()
        if raw.empty:
            return None, raw, "No values available for this histogram."
        fig = px.histogram(raw, x=x, color=group, nbins=bins, barmode="overlay", opacity=0.75)
        fig.update_yaxes(title_text="Number of records")
        return style_figure(fig, height=520), raw, None

    if kind in ("Box plot", "Violin plot"):
        raw = frame.dropna(subset=[x, y] + ([group] if group else [])).copy()
        if raw.empty:
            return None, raw, "No complete records for the selected variables."
        if kind == "Box plot":
            fig = px.box(raw, x=x, y=y, color=group, points=False)
        else:
            fig = px.violin(raw, x=x, y=y, color=group, box=True)
        return style_figure(fig, height=540), raw, None

    raw = frame.dropna(subset=[x] + ([group] if group else [])).copy()
    if raw.empty:
        return None, raw, "No complete records for the selected variables."
    result = group_chart_data(raw, x, y, group, aggregation, date_grain)
    if result.empty:
        return None, result, "No values remain after aggregation."

    if kind == "Line chart":
        if result["Chart X"].nunique() > 240:
            return None, result, "Too many X values for a readable line chart. Choose a date or less granular variable."
        fig = px.line(
            result, x="Chart X", y="Value",
            color="Chart Group" if "Chart Group" in result else None,
            markers=True,
        )
        if (highlight_quarter and pd.api.types.is_datetime64_any_dtype(result["Chart X"])
                and date_grain not in ("Quarter", "Year")):
            years = sorted(result["Chart X"].dt.year.unique())
            shade_q4(fig, [int(year) for year in years])
        fig.update_layout(xaxis_title=x, yaxis_title=f"{aggregation} of {y}" if y else aggregation)
        return style_figure(fig, height=530), result, None

    if kind == "Bar chart":
        # Protect readability for high-cardinality columns, e.g., Product Name.
        if result["Chart X"].nunique() > top_n:
            most_important = (
                result.groupby("Chart X", dropna=False)["Value"].sum()
                .abs().sort_values(ascending=False).head(top_n).index
            )
            result = result[result["Chart X"].isin(most_important)].copy()
            notice = f"Showing the {top_n} largest X groups by absolute aggregated value."
        else:
            notice = None
        color = "Chart Group" if "Chart Group" in result else None
        if orientation == "Horizontal":
            fig = px.bar(result, x="Value", y="Chart X", color=color, orientation="h", barmode="group")
            fig.update_layout(xaxis_title=f"{aggregation} of {y}" if y else aggregation, yaxis_title=x)
        else:
            fig = px.bar(result, x="Chart X", y="Value", color=color, barmode="group")
            fig.update_layout(xaxis_title=x, yaxis_title=f"{aggregation} of {y}" if y else aggregation)
        return style_figure(fig, height=540), result, notice

    if kind == "Pie chart":
        # A slice of a pie must represent a strictly positive proportion.
        if result["Value"].isna().any() or (result["Value"] < 0).any():
            return None, result, (
                "The selected grouping produces negative or missing values. "
                "A pie chart cannot represent these faithfully. Try a bar chart, "
                "or select a nonnegative metric."
            )
        result = result[result["Value"] > 0].copy()
        if result.empty:
            return None, result, "All values are zero, so there are no pie slices to display."
        notice = None
        if len(result) > 12:
            # Preserve the total by combining the smallest slices into Other.
            result = result.sort_values("Value", ascending=False)
            other_value = float(result.iloc[11:]["Value"].sum())
            result = result.iloc[:11].copy()
            result = pd.concat(
                [result, pd.DataFrame({"Chart X": ["Other"], "Value": [other_value]})],
                ignore_index=True,
            )
            notice = "Grouped the smallest slices into 'Other' to keep the pie readable."
        fig = px.pie(result, names="Chart X", values="Value", hole=0)
        fig.update_traces(textinfo="label+percent", textposition="inside")
        return style_figure(fig, height=540), result, notice

    raise ValueError(f"Unknown chart type: {kind}")

# -------------------------------- Sidebar -------------------------------- #
st.title("📊 Customer & Sales Analytics")
st.caption("Interactive sales, profitability, returns, and customer dashboard")

with st.sidebar:
    st.header("Dashboard filters")
    with st.expander("Use different CSV files (optional)"):
        st.caption("Upload replacements, or use the three files bundled with the project.")
        order_upload = st.file_uploader("Orders CSV", type="csv", key="orders_upload")
        customer_upload = st.file_uploader("Customers CSV", type="csv", key="customers_upload")
        return_upload = st.file_uploader("Returns CSV", type="csv", key="returns_upload")

order_bytes = _read_bytes(order_upload, "orders.csv")
customer_bytes = _read_bytes(customer_upload, "customers(1).csv")
returns_bytes = _read_bytes(return_upload, "returns.csv")

if any(x is None for x in (order_bytes, customer_bytes, returns_bytes)):
    st.info(
        "Add **orders.csv**, **customers(1).csv**, and **returns.csv** to the `data/` "
        "folder, or upload the missing files using the sidebar."
    )
    st.stop()

try:
    data, customer_locations, customer_profiles, date_issues, quality = prepare_data(
        order_bytes, customer_bytes, returns_bytes
    )
except (ValueError, KeyError, pd.errors.ParserError) as exc:
    st.error(f"Could not prepare the datasets: {exc}")
    st.stop()

valid_dates = data["Order Date Parsed"].dropna()
if valid_dates.empty:
    st.error("No valid order dates found. Check the Order Date field.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()
all_categories = [c for c in CATEGORY_ORDER if c in set(data["Category"].dropna())]
all_categories += sorted(set(data["Category"].dropna()) - set(all_categories))
all_segments = [s for s in SEGMENT_ORDER if s in set(data["Segment"].dropna())]
all_segments += sorted(set(data["Segment"].dropna()) - set(all_segments))

with st.sidebar:
    dates = st.date_input(
        "Order dates (inclusive)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="MM/DD/YYYY",
    )
    selected_categories = st.multiselect(
        "Product categories", options=all_categories, default=all_categories
    )
    selected_segments = st.multiselect(
        "Customer segments", options=all_segments, default=all_segments
    )
    highlight_q4 = st.checkbox("Shade Q4 (October–December)", value=True)
    st.divider()
    st.caption("All charts use the selected order-date and category/segment filters.")

if len(dates) != 2:
    st.info("Select both the start and end dates in the sidebar.")
    st.stop()
if not selected_categories or not selected_segments:
    st.info("Select at least one category and customer segment.")
    st.stop()

start_date, end_date = dates
filtered = data.loc[
    data["Order Date Parsed"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    & data["Category"].isin(selected_categories)
    & data["Segment"].isin(selected_segments)
].copy()
if filtered.empty:
    st.warning("There are no orders with this selection. Try expanding the filters.")
    st.stop()

# ---------------------------------- KPIs ---------------------------------- #
unique_orders = filtered["Order ID"].nunique()
returned_orders = filtered.loc[filtered["Returned"], "Order ID"].nunique()
return_rate = returned_orders / unique_orders if unique_orders else 0.0
sales_total = float(filtered["Sales"].sum())
profit_total = float(filtered["Profit"].sum())
profit_margin = profit_total / sales_total if sales_total else 0.0

kpis = st.columns(5)
kpis[0].metric("Total sales", money(sales_total))
kpis[1].metric("Total profit", money(profit_total))
kpis[2].metric("Profit margin", f"{profit_margin:.1%}")
kpis[3].metric("Unique orders", f"{unique_orders:,}")
kpis[4].metric("Order return rate", f"{return_rate:.1%}")

st.caption(
    f"{start_date:%b %d, %Y} – {end_date:%b %d, %Y} | "
    f"{len(filtered):,} order lines | "
    f"{filtered['Customer ID'].nunique():,} customers | "
    f"{returned_orders:,} returned orders"
)
if quality["invalid_order_dates"]:
    st.info(
        f"{quality['invalid_order_dates']} source order line(s) have invalid/missing dates "
        "and cannot be allocated to a month. They are excluded from date-filtered charts."
    )

sales_tab, profit_tab, return_tab, region_tab, custom_tab, quality_tab = st.tabs(
    ["📈 Sales trends", "💰 Profitability", "↩️ Returns", "🗺️ Regions", "🎨 Custom charts", "🧹 Data quality & exports"]
)

# ------------------------------- Sales trends ------------------------------ #
with sales_tab:
    st.subheader("Monthly sales by product category")
    st.caption("Visualization type: **Multi-series line chart**")
    monthly_sales = aggregate_monthly(filtered)
    fig = px.line(
        monthly_sales,
        x="Month",
        y="Sales",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"Category": CATEGORY_ORDER},
        markers=True,
        labels={"Sales": "Monthly sales (USD)", "Month": "Order month"},
    )
    if highlight_q4:
        shade_q4(fig, list(range(start_date.year, end_date.year + 1)))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", rangemode="tozero")
    fig.update_xaxes(tickformat="%b %Y", dtick="M3")
    style_figure(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Pale orange bands mark Q4 (October–December). Hover over any point for its value.")

    with st.expander("View total monthly sales (all selected categories)"):
        totals = monthly_sales.groupby("Month", as_index=False)["Sales"].sum()
        total_fig = px.line(
            totals, x="Month", y="Sales", markers=True,
            labels={"Sales": "Total monthly sales (USD)", "Month": "Order month"},
        )
        if highlight_q4:
            shade_q4(total_fig, list(range(start_date.year, end_date.year + 1)))
        total_fig.update_traces(line_color="#3867A9")
        total_fig.update_yaxes(tickprefix="$", tickformat=",.0f", rangemode="tozero")
        style_figure(total_fig)
        st.plotly_chart(total_fig, use_container_width=True)

    st.markdown("**Total sales by category (selected period)**")
    sales_by_category = (
        filtered.groupby("Category", as_index=False)["Sales"].sum()
        .sort_values("Sales", ascending=False)
    )
    sales_by_category["Sales"] = sales_by_category["Sales"].round(2)
    st.dataframe(sales_by_category, hide_index=True, use_container_width=True)

# -------------------------------- Profitability ---------------------------- #
with profit_tab:
    left, right = st.columns(2)
    prof_by_cat = (
        filtered.groupby("Category", as_index=False)["Profit"].sum()
        .sort_values("Profit", ascending=False)
    )
    with left:
        st.subheader("Profit share by category")
        if (prof_by_cat["Profit"] > 0).all():
            st.caption("Visualization type: **Pie chart** (share of net profit)")
            pie = px.pie(
                prof_by_cat, values="Profit", names="Category", hole=0.0,
                color="Category", color_discrete_map=CATEGORY_COLORS,
            )
            pie.update_traces(textposition="inside", textinfo="percent+label")
            style_figure(pie)
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.caption("Visualization type: **Bar chart** (net profits include zero/negative values)")
            st.warning("A pie chart would misrepresent zero or negative category profits for these filters.")
            alt = px.bar(prof_by_cat, x="Category", y="Profit", color="Category",
                         color_discrete_map=CATEGORY_COLORS)
            alt.update_yaxes(tickprefix="$")
            style_figure(alt)
            st.plotly_chart(alt, use_container_width=True)

    with right:
        st.subheader("Profit by customer segment")
        st.caption("Visualization type: **Column chart**")
        segments = (
            filtered.groupby("Segment", as_index=False)["Profit"].sum()
            .sort_values("Profit", ascending=False)
        )
        segment_fig = px.bar(
            segments, x="Segment", y="Profit", color="Segment", text="Profit",
            labels={"Profit": "Total profit (USD)"},
        )
        segment_fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        segment_fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        segment_fig.update_layout(showlegend=False)
        style_figure(segment_fig)
        st.plotly_chart(segment_fig, use_container_width=True)

    st.subheader("Profit distribution per order line")
    st.caption("Visualization type: **Box plot** (median, interquartile range, and whiskers)")
    show_outliers = st.checkbox("Show individual outlier points", value=False)
    box = px.box(
        filtered,
        x="Category", y="Profit", color="Category",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"Category": CATEGORY_ORDER},
        points="outliers" if show_outliers else False,
        labels={"Profit": "Profit per order line (USD)"},
    )
    box.update_layout(showlegend=False)
    box.update_yaxes(tickprefix="$", tickformat=",.0f")
    style_figure(box, height=460)
    st.plotly_chart(box, use_container_width=True)
    st.caption("Outliers can be shown for detail; hiding them makes typical profit levels easier to compare.")

# --------------------------------- Returns -------------------------------- #
with return_tab:
    st.subheader("Return rate by product category")
    st.caption("Visualization type: **Horizontal bar chart**")
    returns_by_cat = category_returns(filtered)
    ret_fig = px.bar(
        returns_by_cat,
        x="Return Rate (%)", y="Category", orientation="h",
        color="Category", color_discrete_map=CATEGORY_COLORS,
        text="Return Rate (%)",
        custom_data=["Returned_Orders", "Orders"],
    )
    ret_fig.update_traces(
        texttemplate="%{text:.2f}%", textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<br>Returned orders: %{customdata[0]}"
        "<br>Orders with category: %{customdata[1]}<extra></extra>",
    )
    ret_fig.update_xaxes(ticksuffix="%", range=[0, max(1, returns_by_cat["Return Rate (%)"].max() * 1.25)])
    ret_fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    style_figure(ret_fig)
    st.plotly_chart(ret_fig, use_container_width=True)
    st.dataframe(returns_by_cat, hide_index=True, use_container_width=True)
    st.info(
        "Returns are identified at the **order** level, not the individual product-line level. "
        "An order with multiple categories counts in each category it contains. "
        "The rate is returned unique orders / all unique orders containing that category. "
        "A returned multi-category order does not prove that every item in it was returned."
    )

# --------------------------------- Regions -------------------------------- #
with region_tab:
    st.subheader("Sales by customer region")
    st.caption("Visualization type: **Horizontal bar chart** — only unambiguous region assignments")
    is_known = filtered["Region (confirmed)"].notna()
    assigned_sales = float(filtered.loc[is_known, "Sales"].sum())
    unassigned_sales = float(filtered.loc[~is_known, "Sales"].sum())
    unassigned_pct = 100 * unassigned_sales / sales_total if sales_total else 0

    c1, c2 = st.columns(2)
    c1.metric("Regionally assignable sales", money(assigned_sales))
    c2.metric("Unassignable sales", f"{unassigned_pct:.1f}%")
    st.warning(
        "A Customer ID appears in multiple regions in the source customer table. "
        "Orders have no order-level region or address key, so most sales cannot "
        "be assigned reliably to a single region. The chart below is **not** a "
        "complete comparison of regional sales."
    )
    region_sums = (
        filtered.loc[is_known]
        .groupby("Region (confirmed)", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=True)
    )
    if region_sums.empty:
        st.info("No orders have reliably assignable regions with the selected filters.")
    else:
        fig_region = px.bar(
            region_sums, x="Sales", y="Region (confirmed)", orientation="h",
            text="Sales", labels={"Sales": "Confirmed sales (USD)", "Region (confirmed)": "Region"},
        )
        fig_region.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_region.update_xaxes(tickprefix="$", tickformat=",.0f")
        style_figure(fig_region)
        st.plotly_chart(fig_region, use_container_width=True)
    st.caption(
        f"{quality['ambiguous_region_customers']:,} of "
        f"{quality['unique_customers']:,} customers have multiple listed regions. "
        "A transaction-level address or region field would enable a reliable full regional comparison."
    )

# ---------------------------- Custom charts tab --------------------------- #
with custom_tab:
    st.subheader("Build your own visualization")
    st.caption(
        "Choose a chart type and select the variables you want to compare. "
        "Every chart uses the date, category and segment filters in the sidebar."
    )
    source_view, chart_selection = st.columns([1, 1])
    with source_view:
        custom_view = st.selectbox(
            "Level of analysis",
            CUSTOM_DATA_VIEWS,
            help="A row represents an order line, an entire order, a customer, or a month/category total.",
            key="custom_view",
        )
    with chart_selection:
        custom_kind = st.selectbox("Visualization type", CUSTOM_CHART_TYPES, key="custom_kind")

    custom_data = prepare_custom_view(filtered, custom_view)
    numeric_cols, datetime_cols, categorical_cols = classify_chart_columns(custom_data)
    group_choices = ["(None)"] + [c for c in categorical_cols if c not in ("Product Name",)]

    if custom_view == "Unique orders":
        st.info(
            "Orders containing more than one product category are labeled "
            "'Multiple categories' instead of being assigned to a single category."
        )
    if custom_view == "Monthly by category":
        st.caption("One row represents one month and one category; sales and profit are already summed.")

    chart_x = None
    chart_y = None
    chart_group = None
    chart_size = None
    chart_agg = "Sum"
    chart_grain = "Month"
    chart_bins = 25
    chart_top_n = 20
    chart_orientation = "Vertical"

    if custom_kind == "Scatter plot":
        st.markdown("**Scatter plot:** compare two numeric measures, optionally coloring and sizing markers.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            chart_x = st.selectbox("X axis", numeric_cols + datetime_cols,
                                   index=custom_chart_default(numeric_cols + datetime_cols, "Sales"), key="scatter_x")
        with col2:
            chart_y = st.selectbox("Y axis", numeric_cols,
                                   index=custom_chart_default(numeric_cols, "Profit"), key="scatter_y")
        with col3:
            color = st.selectbox("Color / group", group_choices,
                                 index=custom_chart_default(group_choices, "Category"), key="scatter_group")
            chart_group = None if color == "(None)" else color
        with col4:
            nonnegative = [c for c in numeric_cols if (custom_data[c].dropna() >= 0).all()]
            size_options = ["(None)"] + nonnegative
            bubble = st.selectbox("Bubble size (optional)", size_options,
                                  index=custom_chart_default(size_options, "Quantity"), key="scatter_size")
            chart_size = None if bubble == "(None)" else bubble

    elif custom_kind in ("Line chart", "Bar chart", "Pie chart"):
        if custom_kind == "Line chart":
            st.markdown("**Line chart:** group by time (or another ordered variable) and aggregate a measure.")
            x_options = datetime_cols + categorical_cols
            default_x = "Month" if "Month" in datetime_cols else "Order Date"
        else:
            st.markdown("**Bar / pie chart:** group records by a category and summarize a measure.")
            x_options = categorical_cols + datetime_cols
            default_x = "Category"
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_x = st.selectbox("Group by / X axis", x_options,
                                   index=custom_chart_default(x_options, default_x), key="aggregate_x")
        with col2:
            chart_agg_options = list(AGGREGATIONS) if "Order ID" in custom_data else list(AGGREGATIONS[:-1])
            chart_agg = st.selectbox("Aggregation", chart_agg_options, key="aggregate_method")
        with col3:
            if chart_agg in ("Count rows", "Distinct orders"):
                chart_y = None
                st.caption("No Y metric needed for this aggregation.")
            else:
                chart_y = st.selectbox("Y / value", numeric_cols,
                                       index=custom_chart_default(numeric_cols, "Sales"), key="aggregate_y")
        if custom_kind != "Pie chart":
            opt1, opt2, opt3 = st.columns(3)
            with opt1:
                optional_color = ["(None)"] + [c for c in categorical_cols if c != chart_x]
                color = st.selectbox("Color / split by", optional_color,
                                     index=custom_chart_default(optional_color, "Category" if custom_kind == "Line chart" else "(None)"),
                                     key="aggregate_color")
                chart_group = None if color == "(None)" else color
            with opt2:
                if chart_x in datetime_cols:
                    chart_grain = st.selectbox("Time grouping", TIME_GRAINS,
                                               index=custom_chart_default(TIME_GRAINS, "Month"), key="time_grain")
                else:
                    st.caption("Time grouping appears when X is a date.")
            with opt3:
                if custom_kind == "Bar chart":
                    chart_orientation = st.selectbox("Orientation", ("Vertical", "Horizontal"), key="bar_orientation")
                    chart_top_n = st.slider("Maximum X groups", min_value=5, max_value=50,
                                             value=20, step=5, key="bar_top_n")
                else:
                    st.caption("Q4 highlighting follows the sidebar setting.")
        else:
            st.caption("Pie charts work only with nonnegative group totals; otherwise use a bar chart.")

    elif custom_kind == "Histogram":
        st.markdown("**Histogram:** explore the distribution of one numeric measure.")
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_x = st.selectbox("Numeric measure", numeric_cols,
                                   index=custom_chart_default(numeric_cols, "Sales"), key="hist_x")
        with col2:
            color = st.selectbox("Color / group", group_choices, key="hist_color")
            chart_group = None if color == "(None)" else color
        with col3:
            chart_bins = st.slider("Number of bins", min_value=5, max_value=80, value=25, key="hist_bins")

    else:  # Box plot and Violin plot
        st.markdown("**Distribution chart:** compare a numeric variable across groups.")
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_x = st.selectbox("Category / X axis", categorical_cols,
                                   index=custom_chart_default(categorical_cols, "Category"), key="distribution_x")
        with col2:
            chart_y = st.selectbox("Numeric / Y axis", numeric_cols,
                                   index=custom_chart_default(numeric_cols, "Profit"), key="distribution_y")
        with col3:
            color = st.selectbox("Color / group", ["(None)"] + [c for c in categorical_cols if c != chart_x],
                                 key="distribution_color")
            chart_group = None if color == "(None)" else color

    # Plotly can technically color thousands of IDs, but the resulting chart
    # is unreadable and potentially too heavy for a free cloud app.
    problem = None
    if chart_group and custom_data[chart_group].nunique(dropna=True) > 25:
        problem = "The color field has over 25 groups. Please choose a lower-cardinality field or '(None)'."
    if (custom_kind in ("Box plot", "Violin plot") and chart_x is not None
            and custom_data[chart_x].nunique(dropna=True) > 40):
        problem = "The category field has over 40 groups. Please select a shorter category grouping."
    if custom_kind == "Line chart" and chart_x in categorical_cols and custom_data[chart_x].nunique() > 80:
        problem = "This X axis has over 80 categories. Use a date or a lower-cardinality categorical variable."

    if problem:
        st.warning(problem)
    else:
        chart_fig, chart_table, chart_note = make_custom_chart(
            custom_data, custom_kind, x=chart_x, y=chart_y,
            group=chart_group, size=chart_size, aggregation=chart_agg,
            date_grain=chart_grain, bins=chart_bins, top_n=chart_top_n,
            orientation=chart_orientation, highlight_quarter=highlight_q4,
        )
        if chart_note:
            st.info(chart_note) if chart_fig is not None else st.warning(chart_note)
        if chart_fig is not None:
            st.caption(f"**Visualization type: {custom_kind}** · Data level: {custom_view} · {len(chart_table):,} plotted rows")
            st.plotly_chart(chart_fig, use_container_width=True, key="custom_figure")
            with st.expander("View plotted data and download"):
                st.dataframe(chart_table.head(1000), use_container_width=True, hide_index=True)
                st.caption("Preview shows at most 1,000 rows; CSV contains the full plotted dataset.")
                csv_button, html_button = st.columns(2)
                with csv_button:
                    st.download_button(
                        "⬇ Download plotted data (.csv)",
                        data=report_csv(chart_table),
                        file_name="custom_chart_data.csv", mime="text/csv", key="custom_data_download",
                    )
                with html_button:
                    st.download_button(
                        "⬇ Download interactive chart (.html)",
                        data=chart_fig.to_html(include_plotlyjs=True, full_html=True).encode("utf-8"),
                        file_name="custom_chart.html", mime="text/html", key="custom_html_download",
                    )
    st.caption(
        "Tip: choose 'Order line items' for Sales vs Profit scatter plots; "
        "'Customer summaries' for customer-level comparisons; and "
        "'Monthly by category' for trend analysis. "
        "Regional assignments are shown only when unambiguous."
    )

# ---------------------------- Quality and exports -------------------------- #
with quality_tab:
    st.subheader("Data quality summary")
    quality_table = pd.DataFrame(
        [
            ("Original order line rows", quality["source_order_lines"]),
            ("Unique orders", quality["unique_orders"]),
            ("Unique customers", quality["unique_customers"]),
            ("Customer-location rows", quality["customer_location_rows"]),
            ("Order lines without a customer match", quality["unmatched_order_lines"]),
            ("Unique orders without a customer match", quality["unmatched_unique_orders"]),
            ("Returned Order IDs not found in orders", quality["returns_not_in_orders"]),
            ("Customer IDs with multiple regions", quality["ambiguous_region_customers"]),
            ("Duplicate Row ID occurrences", quality["duplicated_row_ids"]),
            ("Invalid/missing order dates", quality["invalid_order_dates"]),
            ("Invalid/missing ship dates", quality["invalid_ship_dates"]),
        ],
        columns=["Quality metric", "Value"],
    )
    st.dataframe(quality_table, hide_index=True, use_container_width=True)

    if not date_issues.empty:
        st.markdown("**Invalid source dates**")
        st.dataframe(date_issues, hide_index=True, use_container_width=True)

    st.markdown("**Filtered, safely joined order rows**")
    visible_columns = [
        "Row ID", "Order ID", "Order Date", "Customer ID", "Segment", "Country",
        "Category", "Sub-Category", "Sales", "Quantity", "Discount", "Profit",
        "Returned", "Region Display", "Customer Matched",
    ]
    visible_columns = [c for c in visible_columns if c in filtered.columns]
    st.dataframe(filtered[visible_columns].head(1000), use_container_width=True, hide_index=True)
    st.caption("Preview shows up to 1,000 rows; the download below contains all filtered rows.")
    st.download_button(
        "⬇ Download filtered joined orders (CSV)",
        data=report_csv(filtered[visible_columns]),
        file_name="filtered_customer_orders.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.download_button(
        "⬇ Download customer-location reference (CSV)",
        data=report_csv(customer_locations),
        file_name="customer_locations_reference.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.caption(
        "Joining directly to customer-location records would multiply order rows. "
        "The dashboard joins only the stable customer profile and assigns region "
        "when a customer has exactly one region in the source."
    )

st.divider()
st.caption(
    "Data source: CSV files bundled with this project (or uploaded in the sidebar). "
    "All metrics use source order dates, including any future-dated records present in the data. "
    "For public deployment, review privacy before publishing customer-level CSV files."
)
