from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, ttest_ind
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest
from statsmodels.stats.power import NormalIndPower

from ecommerce_portfolio.data import OlistData, ConversionData


# ── Olist overview ────────────────────────────────────────────────────────────

@dataclass
class OlistOverview:
    total_orders: int
    fulfillment_rate: float
    cancel_rate: float
    avg_order_value: float
    repeat_rate: float
    revenue_at_risk: float
    date_min: str
    date_max: str


def build_olist_overview(data: OlistData) -> OlistOverview:
    df = data.master
    total = len(df)
    fulfillment = (df["order_status"] == "delivered").mean()
    cancel = (df["order_status"] == "canceled").mean()
    aov = df["payment_value"].mean()
    repeat = (
        df.groupby("customer_unique_id")["order_id"].nunique().gt(1).mean() * 100
    )
    non_delivered = (df["order_status"] != "delivered").sum()
    revenue_at_risk = non_delivered * aov
    ts = data.orders["order_purchase_timestamp"].dropna()
    return OlistOverview(
        total_orders=total,
        fulfillment_rate=fulfillment,
        cancel_rate=cancel,
        avg_order_value=aov,
        repeat_rate=repeat,
        revenue_at_risk=revenue_at_risk,
        date_min=str(ts.min().date()),
        date_max=str(ts.max().date()),
    )


# ── Olist temporal ────────────────────────────────────────────────────────────

def build_monthly_orders(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("purchase_month")
        .size()
        .reset_index(name="orders")
    )
    monthly["purchase_month"] = monthly["purchase_month"].astype(str)
    return monthly


def build_hourly_orders(df: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        df.groupby("purchase_hour")
        .size()
        .reset_index(name="orders")
    )
    return hourly


# ── Olist status ──────────────────────────────────────────────────────────────

def build_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df["order_status"].value_counts().reset_index()
    dist.columns = ["status", "count"]
    return dist


# ── Olist categories ──────────────────────────────────────────────────────────

def build_top_categories(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        df["primary_category"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"index": "category", "primary_category": "category", "count": "orders"})
    )


def build_top_revenue_categories(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        df.groupby("primary_category")["total_price"]
        .sum()
        .nlargest(n)
        .reset_index()
        .rename(columns={"primary_category": "category", "total_price": "revenue"})
    )


# ── Olist geo / payment ──────────────────────────────────────────────────────

def build_state_orders(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df["customer_state"]
        .value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"customer_state": "state", "count": "orders"})
    )


def build_payment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df["payment_type"].value_counts().reset_index()
    dist.columns = ["payment_type", "count"]
    return dist


# ── Olist funnel ──────────────────────────────────────────────────────────────

def build_olist_funnel(data: OlistData) -> pd.DataFrame:
    df = data.master
    total = len(df)
    approved = df["order_approved_at"].notna().sum()
    shipped = df["order_delivered_carrier_date"].notna().sum()
    delivered = df["order_delivered_customer_date"].notna().sum()

    stages = ["Order Placed", "Approved", "Shipped to Carrier", "Delivered"]
    counts = [total, approved, shipped, delivered]
    funnel = pd.DataFrame({"stage": stages, "orders": counts})
    funnel["pct_of_total"] = (funnel["orders"] / funnel["orders"].iloc[0] * 100).round(2)
    funnel["pct_of_previous"] = (
        funnel["orders"] / funnel["orders"].shift(1) * 100
    ).round(2)
    funnel["drop_off"] = funnel["orders"].shift(1) - funnel["orders"]
    funnel["drop_off_pct"] = (
        funnel["drop_off"] / funnel["orders"].shift(1) * 100
    ).round(2)
    return funnel


# ── Olist delivery ────────────────────────────────────────────────────────────

@dataclass
class DeliveryStats:
    total_delivered: int
    median_days: float
    late_pct: float
    late_count: int
    on_time_avg_review: float
    late_avg_review: float
    review_drop: float
    mann_whitney_p: float


def build_delivery_stats(delivered: pd.DataFrame) -> DeliveryStats:
    with_reviews = delivered.dropna(subset=["review_score"])
    on_time = with_reviews[~with_reviews["is_late"]]["review_score"]
    late = with_reviews[with_reviews["is_late"]]["review_score"]
    _, p_val = mannwhitneyu(on_time, late, alternative="two-sided")
    return DeliveryStats(
        total_delivered=len(delivered),
        median_days=delivered["delivery_time_days"].median(),
        late_pct=delivered["is_late"].mean() * 100,
        late_count=int(delivered["is_late"].sum()),
        on_time_avg_review=on_time.mean(),
        late_avg_review=late.mean(),
        review_drop=on_time.mean() - late.mean(),
        mann_whitney_p=p_val,
    )


def build_state_delivery(delivered: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    top_states = delivered["customer_state"].value_counts().head(n).index
    return (
        delivered[delivered["customer_state"].isin(top_states)]
        .groupby("customer_state")
        .agg(
            avg_delivery_days=("delivery_time_days", "mean"),
            late_pct=("is_late", "mean"),
            avg_review=("review_score", "mean"),
            n_orders=("order_id", "count"),
        )
        .round(3)
        .assign(late_pct=lambda x: (x["late_pct"] * 100).round(1))
        .sort_values("late_pct", ascending=False)
        .reset_index()
        .rename(columns={"customer_state": "state"})
    )


# ── Olist reviews ─────────────────────────────────────────────────────────────

def build_review_distribution(delivered: pd.DataFrame) -> pd.DataFrame:
    with_reviews = delivered.dropna(subset=["review_score"])
    dist = (
        with_reviews["review_score"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    dist.columns = ["score", "count"]
    dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(1)
    return dist


def build_category_reviews(delivered: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    with_reviews = delivered.dropna(subset=["review_score"])
    top_cats = with_reviews["primary_category"].value_counts().head(n).index
    return (
        with_reviews[with_reviews["primary_category"].isin(top_cats)]
        .groupby("primary_category")["review_score"]
        .mean()
        .sort_values()
        .reset_index()
        .rename(columns={"primary_category": "category", "review_score": "avg_score"})
    )


def build_monthly_reviews(delivered: pd.DataFrame) -> pd.DataFrame:
    with_reviews = delivered.dropna(subset=["review_score"])
    monthly = (
        with_reviews.groupby("purchase_month")["review_score"]
        .mean()
        .reset_index()
    )
    monthly["purchase_month"] = monthly["purchase_month"].astype(str)
    return monthly


# ── Olist customer behavior ──────────────────────────────────────────────────

def build_customer_orders(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("customer_unique_id")
        .agg(
            n_orders=("order_id", "nunique"),
            total_spend=("payment_value", "sum"),
            avg_review=("review_score", "mean"),
        )
        .reset_index()
        .assign(is_repeat=lambda x: (x["n_orders"] > 1).astype(int))
    )


# ── Olist A/B test ───────────────────────────────────────────────────────────

@dataclass
class ABTestResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    ci_lower: float
    ci_upper: float
    z_stat: float
    p_value: float
    significant: bool
    n_control: int
    n_treatment: int


def run_olist_ab_test(delivered: pd.DataFrame, seed: int = 42) -> ABTestResult:
    experiment = delivered.dropna(subset=["review_score"]).copy()
    experiment["is_satisfied"] = (experiment["review_score"] >= 4).astype(int)

    np.random.seed(seed)
    experiment["ab_group"] = np.random.choice(
        ["control", "treatment"], size=len(experiment), p=[0.5, 0.5]
    )

    treatment_mask = experiment["ab_group"] == "treatment"
    experiment["is_satisfied_ab"] = experiment["is_satisfied"].copy()

    late_unsat = treatment_mask & experiment["is_late"] & (experiment["is_satisfied"] == 0)
    flip_idx = experiment[late_unsat].sample(frac=0.30, random_state=seed).index
    experiment.loc[flip_idx, "is_satisfied_ab"] = 1

    ontime_unsat = treatment_mask & (~experiment["is_late"]) & (experiment["is_satisfied"] == 0)
    flip_idx2 = experiment[ontime_unsat].sample(frac=0.03, random_state=seed).index
    experiment.loc[flip_idx2, "is_satisfied_ab"] = 1

    ctrl = experiment[experiment["ab_group"] == "control"]
    treat = experiment[experiment["ab_group"] == "treatment"]

    ctrl_rate = ctrl["is_satisfied_ab"].mean()
    treat_rate = treat["is_satisfied_ab"].mean()

    n_c, n_t = len(ctrl), len(treat)
    x_c = ctrl["is_satisfied_ab"].sum()
    x_t = treat["is_satisfied_ab"].sum()

    z_stat, p_value = proportions_ztest([x_t, x_c], [n_t, n_c], alternative="two-sided")
    se_diff = np.sqrt(
        ctrl_rate * (1 - ctrl_rate) / n_c + treat_rate * (1 - treat_rate) / n_t
    )
    diff = treat_rate - ctrl_rate

    return ABTestResult(
        control_rate=ctrl_rate,
        treatment_rate=treat_rate,
        absolute_lift=diff,
        relative_lift=(diff / ctrl_rate * 100) if ctrl_rate > 0 else 0,
        ci_lower=(diff - 1.96 * se_diff) * 100,
        ci_upper=(diff + 1.96 * se_diff) * 100,
        z_stat=float(z_stat),
        p_value=float(p_value),
        significant=float(p_value) < 0.05,
        n_control=n_c,
        n_treatment=n_t,
    )


# ── Olist revenue impact ─────────────────────────────────────────────────────

@dataclass
class OlistRevenueImpact:
    avg_order_value: float
    orders_per_year: float
    current_satisfaction: float
    projected_satisfaction: float
    current_repeat_rate: float
    projected_repeat_rate: float
    additional_repeat_orders: float
    additional_revenue: float
    compensation_cost: float
    net_impact: float


def build_olist_revenue_impact(
    data: OlistData, ab: ABTestResult
) -> OlistRevenueImpact:
    df = data.master
    aov = df["payment_value"].mean()
    orders_year = len(df) / 2
    repeat_rate = (
        df.groupby("customer_unique_id")["order_id"].nunique().gt(1).mean()
    )
    proj_repeat = repeat_rate + (ab.treatment_rate - ab.control_rate) * 0.5
    add_orders = orders_year * (proj_repeat - repeat_rate)
    add_rev = add_orders * aov
    late_orders_yr = orders_year * data.delivered["is_late"].mean()
    comp_cost = late_orders_yr * aov * 0.10
    return OlistRevenueImpact(
        avg_order_value=aov,
        orders_per_year=orders_year,
        current_satisfaction=ab.control_rate,
        projected_satisfaction=ab.treatment_rate,
        current_repeat_rate=repeat_rate * 100,
        projected_repeat_rate=proj_repeat * 100,
        additional_repeat_orders=add_orders,
        additional_revenue=add_rev,
        compensation_cost=comp_cost,
        net_impact=add_rev - comp_cost,
    )


# ── Conversion overview ──────────────────────────────────────────────────────

@dataclass
class ConversionOverview:
    total_events: int
    unique_users: int
    unique_items: int
    unique_transactions: int
    view_count: int
    cart_count: int
    purchase_count: int


def build_conversion_overview(data: ConversionData) -> ConversionOverview:
    ev = data.events
    vc = ev["event"].value_counts()
    return ConversionOverview(
        total_events=len(ev),
        unique_users=ev["visitorid"].nunique(),
        unique_items=ev["itemid"].nunique(),
        unique_transactions=int(ev["transactionid"].dropna().nunique()),
        view_count=int(vc.get("view", 0)),
        cart_count=int(vc.get("addtocart", 0)),
        purchase_count=int(vc.get("transaction", 0)),
    )


# ── Conversion funnel ─────────────────────────────────────────────────────────

def build_conversion_funnel(events: pd.DataFrame) -> pd.DataFrame:
    stages = ["view", "addtocart", "transaction"]
    labels = ["Product View", "Add to Cart", "Purchase"]
    users = [events[events["event"] == s]["visitorid"].nunique() for s in stages]
    funnel = pd.DataFrame({"stage": labels, "users": users})
    funnel["pct_of_total"] = (funnel["users"] / funnel["users"].iloc[0] * 100).round(2)
    funnel["pct_of_previous"] = (
        funnel["users"] / funnel["users"].shift(1) * 100
    ).round(2)
    funnel["drop_off"] = funnel["users"].shift(1) - funnel["users"]
    funnel["drop_off_pct"] = (
        funnel["drop_off"] / funnel["users"].shift(1) * 100
    ).round(2)
    return funnel


def compute_funnel_by_segment(events: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    rows = []
    for seg in events[segment_col].dropna().unique():
        seg_df = events[events[segment_col] == seg]
        viewers = seg_df[seg_df["event"] == "view"]["visitorid"].nunique()
        carters = seg_df[seg_df["event"] == "addtocart"]["visitorid"].nunique()
        buyers = seg_df[seg_df["event"] == "transaction"]["visitorid"].nunique()
        rows.append(
            {
                segment_col: seg,
                "viewers": viewers,
                "carters": carters,
                "buyers": buyers,
                "view_to_cart": carters / viewers * 100 if viewers else 0,
                "cart_to_purchase": buyers / carters * 100 if carters else 0,
                "overall_conversion": buyers / viewers * 100 if viewers else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("overall_conversion", ascending=False)


# ── Conversion temporal ───────────────────────────────────────────────────────

def build_daily_events(events: pd.DataFrame) -> pd.DataFrame:
    daily = events.groupby("date").size().reset_index(name="events")
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def build_hourly_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby("hour").size().reset_index(name="events")


# ── Conversion A/B test ───────────────────────────────────────────────────────

def run_conversion_ab_test(
    user_events: pd.DataFrame, seed: int = 42
) -> ABTestResult:
    ue = user_events.copy()
    baseline_rate = ue["added_to_cart"].mean()

    np.random.seed(seed)
    ue["ab_group"] = np.random.choice(
        ["control", "treatment"], size=len(ue), p=[0.5, 0.5]
    )

    treatment_nocart = (ue["ab_group"] == "treatment") & (ue["added_to_cart"] == 0)
    n_convert = int(treatment_nocart.sum() * baseline_rate * 0.18 / (1 - baseline_rate))
    convert_idx = ue[treatment_nocart].sample(n=n_convert, random_state=seed).index
    ue["added_to_cart_ab"] = ue["added_to_cart"].copy()
    ue.loc[convert_idx, "added_to_cart_ab"] = 1

    ctrl = ue[ue["ab_group"] == "control"]
    treat = ue[ue["ab_group"] == "treatment"]

    ctrl_rate = ctrl["added_to_cart_ab"].mean()
    treat_rate = treat["added_to_cart_ab"].mean()

    n_c, n_t = len(ctrl), len(treat)
    x_c = int(ctrl["added_to_cart_ab"].sum())
    x_t = int(treat["added_to_cart_ab"].sum())

    p_pool = (x_c + x_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z_stat = (treat_rate - ctrl_rate) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    se_diff = np.sqrt(
        ctrl_rate * (1 - ctrl_rate) / n_c + treat_rate * (1 - treat_rate) / n_t
    )
    diff = treat_rate - ctrl_rate

    return ABTestResult(
        control_rate=ctrl_rate,
        treatment_rate=treat_rate,
        absolute_lift=diff,
        relative_lift=(diff / ctrl_rate * 100) if ctrl_rate > 0 else 0,
        ci_lower=(diff - 1.96 * se_diff) * 100,
        ci_upper=(diff + 1.96 * se_diff) * 100,
        z_stat=float(z_stat),
        p_value=float(p_value),
        significant=float(p_value) < 0.05,
        n_control=n_c,
        n_treatment=n_t,
    )


# ── Conversion revenue impact ────────────────────────────────────────────────

@dataclass
class ConversionRevenueImpact:
    monthly_users: float
    avg_order_value: float
    current_cart_rate: float
    projected_cart_rate: float
    cart_to_purchase_rate: float
    additional_carts_monthly: float
    additional_purchases_monthly: float
    additional_revenue_monthly: float
    additional_revenue_annual: float


def build_conversion_revenue_impact(
    data: ConversionData, ab: ABTestResult, funnel: pd.DataFrame
) -> ConversionRevenueImpact:
    events = data.events
    tx = events[events["event"] == "transaction"]
    aov = 75.0
    if "price" in tx.columns:
        p = tx["price"].mean()
        if pd.notna(p) and p > 0:
            aov = p

    monthly_users = len(data.user_events) / 12
    cart_to_purchase = funnel["pct_of_previous"].iloc[2] / 100
    add_carts = monthly_users * (ab.treatment_rate - ab.control_rate)
    add_purchases = add_carts * cart_to_purchase
    add_rev_m = add_purchases * aov

    return ConversionRevenueImpact(
        monthly_users=monthly_users,
        avg_order_value=aov,
        current_cart_rate=ab.control_rate,
        projected_cart_rate=ab.treatment_rate,
        cart_to_purchase_rate=cart_to_purchase,
        additional_carts_monthly=add_carts,
        additional_purchases_monthly=add_purchases,
        additional_revenue_monthly=add_rev_m,
        additional_revenue_annual=add_rev_m * 12,
    )


# ── Segment A/B breakdown (shared) ───────────────────────────────────────────

def build_segment_ab_results(
    df: pd.DataFrame,
    ab_col: str,
    metric_col: str,
    segment_col: str,
) -> pd.DataFrame:
    rows = []
    for seg_val in df[segment_col].dropna().unique():
        seg = df[df[segment_col] == seg_val]
        ctrl = seg[seg["ab_group"] == "control"][metric_col]
        treat = seg[seg["ab_group"] == "treatment"][metric_col]
        if len(ctrl) < 2 or len(treat) < 2:
            continue
        ctrl_r = ctrl.mean()
        treat_r = treat.mean()
        lift = (treat_r - ctrl_r) / ctrl_r * 100 if ctrl_r > 0 else 0
        _, p = ttest_ind(ctrl, treat)
        rows.append(
            {
                "segment": str(seg_val),
                "control_rate": round(ctrl_r, 4),
                "treatment_rate": round(treat_r, 4),
                "lift_pct": round(lift, 1),
                "p_value": round(float(p), 4),
                "significant": float(p) < 0.05,
            }
        )
    return pd.DataFrame(rows)
