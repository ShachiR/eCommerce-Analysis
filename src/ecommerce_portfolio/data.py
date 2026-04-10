from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OLIST_DIR = BASE_DIR / "data"
EVENTS_DIR = BASE_DIR / "data"


@dataclass
class OlistData:
    orders: pd.DataFrame
    items: pd.DataFrame
    payments: pd.DataFrame
    reviews: pd.DataFrame
    customers: pd.DataFrame
    products: pd.DataFrame
    sellers: pd.DataFrame
    category_translation: pd.DataFrame
    master: pd.DataFrame = field(default_factory=pd.DataFrame)
    delivered: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


@dataclass
class ConversionData:
    events: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    user_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


@dataclass
class PortfolioData:
    olist: OlistData
    conversion: ConversionData | None


def _parse_olist_dates(orders: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders["purchase_date"] = orders["order_purchase_timestamp"].dt.date
    orders["purchase_hour"] = orders["order_purchase_timestamp"].dt.hour
    orders["purchase_dow"] = orders["order_purchase_timestamp"].dt.day_name()
    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.to_period("M")

    orders["approval_time_hrs"] = (
        (orders["order_approved_at"] - orders["order_purchase_timestamp"])
        .dt.total_seconds()
        / 3600
    )
    orders["shipping_time_days"] = (
        (orders["order_delivered_carrier_date"] - orders["order_approved_at"])
        .dt.total_seconds()
        / 86400
    )
    orders["delivery_time_days"] = (
        (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"])
        .dt.total_seconds()
        / 86400
    )
    orders["estimated_delivery_days"] = (
        (orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"])
        .dt.total_seconds()
        / 86400
    )
    orders["delivery_vs_estimate"] = (
        orders["delivery_time_days"] - orders["estimated_delivery_days"]
    )
    return orders


def _build_olist_master(data: OlistData) -> pd.DataFrame:
    products = data.products.merge(
        data.category_translation, on="product_category_name", how="left"
    )

    order_items_merged = data.items.merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    numeric_agg = (
        order_items_merged.groupby("order_id")
        .agg(
            total_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
        .reset_index()
    )
    # Fast mode: count category occurrences, keep the most frequent per order
    cat_counts = (
        order_items_merged.groupby(["order_id", "product_category_name_english"])
        .size()
        .reset_index(name="_n")
    )
    primary_cat = (
        cat_counts.sort_values("_n", ascending=False)
        .drop_duplicates("order_id", keep="first")[["order_id", "product_category_name_english"]]
        .rename(columns={"product_category_name_english": "primary_category"})
    )
    order_items_agg = numeric_agg.merge(primary_cat, on="order_id", how="left")
    order_items_agg["primary_category"] = order_items_agg["primary_category"].fillna("unknown")

    order_payments_agg = (
        data.payments.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", "first"),
        )
        .reset_index()
    )

    order_reviews_agg = data.reviews.drop_duplicates("order_id")[
        ["order_id", "review_score"]
    ]

    df = (
        data.orders.merge(
            data.customers[
                [
                    "customer_id",
                    "customer_unique_id",
                    "customer_city",
                    "customer_state",
                ]
            ],
            on="customer_id",
            how="left",
        )
        .merge(order_items_agg, on="order_id", how="left")
        .merge(order_payments_agg, on="order_id", how="left")
        .merge(order_reviews_agg, on="order_id", how="left")
    )
    return df


def load_olist_data() -> OlistData:
    notes: list[str] = []
    orders = pd.read_csv(OLIST_DIR / "olist_orders_dataset.csv")
    items = pd.read_csv(OLIST_DIR / "olist_order_items_dataset.csv")
    payments = pd.read_csv(OLIST_DIR / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(OLIST_DIR / "olist_order_reviews_dataset.csv")
    customers = pd.read_csv(OLIST_DIR / "olist_customers_dataset.csv")
    products = pd.read_csv(OLIST_DIR / "olist_products_dataset.csv")
    sellers = pd.read_csv(OLIST_DIR / "olist_sellers_dataset.csv")
    category_translation = pd.read_csv(
        OLIST_DIR / "product_category_name_translation.csv"
    )

    orders = _parse_olist_dates(orders)

    data = OlistData(
        orders=orders,
        items=items,
        payments=payments,
        reviews=reviews,
        customers=customers,
        products=products,
        sellers=sellers,
        category_translation=category_translation,
        notes=notes,
    )
    data.master = _build_olist_master(data)

    delivered = data.master[data.master["order_status"] == "delivered"].copy()
    delivered["is_late"] = delivered["delivery_vs_estimate"] > 0
    data.delivered = delivered

    return data


def load_conversion_data() -> ConversionData | None:
    required = [EVENTS_DIR / "events.csv", EVENTS_DIR / "user_properties.csv", EVENTS_DIR / "item_properties.csv"]
    if not all(f.exists() for f in required):
        return None

    notes: list[str] = []
    events = pd.read_csv(EVENTS_DIR / "events.csv")
    users = pd.read_csv(EVENTS_DIR / "user_properties.csv")
    items = pd.read_csv(EVENTS_DIR / "item_properties.csv")

    events["datetime"] = pd.to_datetime(events["timestamp"], unit="s")
    events["date"] = events["datetime"].dt.date
    events["hour"] = events["datetime"].dt.hour
    events["day_of_week"] = events["datetime"].dt.day_name()
    events["month"] = events["datetime"].dt.month
    events["week"] = events["datetime"].dt.isocalendar().week.astype(int)

    events = events.merge(users, on="visitorid", how="left")

    user_events = (
        events.groupby("visitorid")
        .agg(
            total_views=("event", lambda x: (x == "view").sum()),
            total_carts=("event", lambda x: (x == "addtocart").sum()),
            total_purchases=("event", lambda x: (x == "transaction").sum()),
            unique_items_viewed=("itemid", "nunique"),
            num_sessions_approx=("date", "nunique"),
            device=("device", "first"),
            country=("country", "first"),
            segment=("segment", "first"),
        )
        .reset_index()
    )
    user_events["converted"] = (user_events["total_purchases"] > 0).astype(int)
    user_events["added_to_cart"] = (user_events["total_carts"] > 0).astype(int)

    data = ConversionData(
        events=events,
        users=users,
        items=items,
        user_events=user_events,
        notes=notes,
    )
    return data


def load_portfolio_data() -> PortfolioData:
    return PortfolioData(
        olist=load_olist_data(),
        conversion=load_conversion_data(),
    )
