"""
Customer Retention & Churn Analysis
=====================================
Subscription business case study.

This script:
  1. Generates a realistic SYNTHETIC subscription dataset (no real dataset
     was supplied for this task, so a hazard-based simulation is used to
     create data with genuine, learnable churn patterns).
  2. Analyzes churn patterns across customer segments.
  3. Identifies retention drivers (usage, plan type, tenure, support tickets).
  4. Computes customer lifetime / cohort retention metrics.
  5. Produces a full cohort retention heatmap + supporting charts.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required to run this script. Install it via `pip install matplotlib`."
    ) from exc
import seaborn as sns
from datetime import datetime
sklearn_available = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sklearn_available = False
    LogisticRegression = None  # type: ignore
    StandardScaler = None  # type: ignore
    print(
        "Warning: scikit-learn is not installed. Retention driver analysis will be skipped."
    )

# ------------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------------
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 130
CHART_DIR = "charts"
TODAY = pd.Timestamp("2026-06-21")          # analysis "as of" date
N_CUSTOMERS = 3000
COHORT_START = pd.Timestamp("2024-01-01")
COHORT_END = pd.Timestamp("2025-12-01")     # last signup cohort

# ------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION (hazard-based churn simulation)
# ------------------------------------------------------------------
# Acquisition channel mix + relative churn-risk multiplier per channel
segments = ["Organic", "Paid Ads", "Referral", "Partner"]
segment_probs = [0.40, 0.30, 0.20, 0.10]
segment_hazard_mult = {"Organic": 0.9, "Paid Ads": 1.35, "Referral": 0.65, "Partner": 1.0}

# Plan mix + relative churn-risk multiplier + monthly price (INR)
plans = ["Basic", "Standard", "Premium"]
plan_probs = [0.45, 0.35, 0.20]
plan_hazard_mult = {"Basic": 1.4, "Standard": 1.0, "Premium": 0.6}
plan_price = {"Basic": 299, "Standard": 599, "Premium": 999}

payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking"]
payment_probs = [0.45, 0.25, 0.20, 0.10]

# Signup volume grows slightly over the 24 monthly cohorts
cohort_months = pd.date_range(COHORT_START, COHORT_END, freq="MS")
growth = np.linspace(0.7, 1.3, len(cohort_months))
cohort_weights = growth / growth.sum()

signup_dates = np.random.choice(cohort_months, size=N_CUSTOMERS, p=cohort_weights)
signup_dates = pd.to_datetime(signup_dates) + pd.to_timedelta(
    np.random.randint(0, 28, size=N_CUSTOMERS), unit="D"
)

customer_segment = np.random.choice(segments, size=N_CUSTOMERS, p=segment_probs)
customer_plan = np.random.choice(plans, size=N_CUSTOMERS, p=plan_probs)
customer_payment = np.random.choice(payment_methods, size=N_CUSTOMERS, p=payment_probs)

# Usage frequency (avg sessions/month) -- log-normal, slightly higher for Premium
base_usage = np.random.lognormal(mean=2.1, sigma=0.6, size=N_CUSTOMERS)
plan_usage_bonus = pd.Series(customer_plan).map({"Basic": 0, "Standard": 2, "Premium": 5}).values
usage_frequency = np.clip(base_usage + plan_usage_bonus, 0.5, 60)

# Support tickets per month (Poisson; more tickets => more churn risk)
ticket_rate = np.where(customer_plan == "Basic", 0.55,
                np.where(customer_plan == "Standard", 0.35, 0.20))
support_tickets = np.random.poisson(lam=ticket_rate)


def usage_multiplier(u):
    if u >= 20:
        return 0.5
    elif u >= 10:
        return 0.8
    elif u >= 5:
        return 1.1
    else:
        return 1.6


def support_multiplier(t):
    if t == 0:
        return 0.9
    elif t == 1:
        return 1.0
    elif t == 2:
        return 1.3
    else:
        return 1.7


def base_hazard(tenure_month):
    """Tenure-dependent baseline monthly churn hazard (onboarding risk +
    a renewal-month bump around month 12)."""
    if tenure_month == 1:
        h = 0.10
    elif tenure_month == 2:
        h = 0.07
    elif tenure_month == 3:
        h = 0.05
    elif tenure_month <= 6:
        h = 0.035
    elif tenure_month <= 12:
        h = 0.025
    elif tenure_month <= 18:
        h = 0.02
    else:
        h = 0.018
    if tenure_month in (11, 12, 13):  # annual renewal risk bump
        h += 0.03
    return h


# Monthly hazard simulation per customer until churn or TODAY
records = []
for i in range(N_CUSTOMERS):
    signup = signup_dates[i]
    seg, plan, pay = customer_segment[i], customer_plan[i], customer_payment[i]
    usage, tickets = usage_frequency[i], support_tickets[i]

    max_tenure = (TODAY.year - signup.year) * 12 + (TODAY.month - signup.month)
    max_tenure = max(max_tenure, 0)

    churned = False
    churn_tenure = None
    for t in range(1, max_tenure + 1):
        h = (base_hazard(t) * plan_hazard_mult[plan] * segment_hazard_mult[seg]
             * usage_multiplier(usage) * support_multiplier(tickets))
        h = min(max(h, 0.0), 0.95)
        if np.random.random() < h:
            churned = True
            churn_tenure = t
            break

    tenure_months = churn_tenure if churned else max_tenure
    churn_date = signup + pd.DateOffset(months=tenure_months) if churned else pd.NaT
    last_active = churn_date if churned else TODAY

    records.append({
        "customer_id": f"CUST{i+1:05d}",
        "signup_date": signup,
        "signup_cohort": signup.strftime("%Y-%m"),
        "acquisition_channel": seg,
        "plan_type": plan,
        "monthly_charge_inr": plan_price[plan],
        "payment_method": pay,
        "avg_sessions_per_month": round(usage, 1),
        "support_tickets_per_month": tickets,
        "tenure_months": tenure_months,
        "churned": churned,
        "churn_date": churn_date,
        "last_active_date": last_active,
    })

df = pd.DataFrame(records)
df.to_csv("subscription_customers.csv", index=False)
print(f"Generated {len(df)} customer records -> subscription_customers.csv")
print(df.head())

# ------------------------------------------------------------------
# 2. CHURN PATTERNS BY SEGMENT
# ------------------------------------------------------------------
overall_churn_rate = df["churned"].mean()
print(f"\nOverall churn rate: {overall_churn_rate:.1%}")

churn_by_plan = df.groupby("plan_type")["churned"].mean().reindex(plans)
churn_by_channel = df.groupby("acquisition_channel")["churned"].mean().reindex(segments)
churn_by_payment = df.groupby("payment_method")["churned"].mean()

# Usage & support buckets for clearer driver analysis
df["usage_bucket"] = pd.cut(
    df["avg_sessions_per_month"], bins=[0, 5, 10, 20, 100],
    labels=["<5/mo", "5-10/mo", "10-20/mo", "20+/mo"]
)
df["ticket_bucket"] = df["support_tickets_per_month"].clip(upper=3).map(
    {0: "0 tickets", 1: "1 ticket", 2: "2 tickets", 3: "3+ tickets"}
)
churn_by_usage = df.groupby("usage_bucket")["churned"].mean()
churn_by_tickets = df.groupby("ticket_bucket")["churned"].mean().reindex(
    ["0 tickets", "1 ticket", "2 tickets", "3+ tickets"]
)

# ------------------------------------------------------------------
# 3. RETENTION DRIVERS -- logistic regression (standardized coefficients)
# ------------------------------------------------------------------
model_df = df.copy()
model_df["channel_paid_ads"] = (model_df["acquisition_channel"] == "Paid Ads").astype(int)
model_df["channel_referral"] = (model_df["acquisition_channel"] == "Referral").astype(int)
model_df["channel_partner"] = (model_df["acquisition_channel"] == "Partner").astype(int)
model_df["plan_standard"] = (model_df["plan_type"] == "Standard").astype(int)
model_df["plan_premium"] = (model_df["plan_type"] == "Premium").astype(int)

feature_cols = [
    "avg_sessions_per_month", "support_tickets_per_month",
    "channel_paid_ads", "channel_referral", "channel_partner",
    "plan_standard", "plan_premium",
]
X = model_df[feature_cols].values
y = model_df["churned"].astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logit = LogisticRegression(max_iter=1000)
logit.fit(X_scaled, y)

driver_impact = pd.Series(logit.coef_[0], index=feature_cols).sort_values()
# Positive coefficient => increases churn odds; negative => protects against churn

# ------------------------------------------------------------------
# 4. CUSTOMER LIFETIME METRICS
# ------------------------------------------------------------------
avg_lifetime_churned = df.loc[df["churned"], "tenure_months"].mean()
avg_tenure_active = df.loc[~df["churned"], "tenure_months"].mean()
median_lifetime_churned = df.loc[df["churned"], "tenure_months"].median()

avg_revenue_per_customer = df["monthly_charge_inr"].mean()
# Simple CLV approximation = avg monthly charge * avg lifetime (months) for churned customers
approx_clv = avg_revenue_per_customer * avg_lifetime_churned

print(f"\nAvg lifetime of churned customers: {avg_lifetime_churned:.1f} months")
print(f"Avg tenure of still-active customers: {avg_tenure_active:.1f} months")
print(f"Approx. CLV (churned cohort): Rs.{approx_clv:,.0f}")

# ------------------------------------------------------------------
# 5. COHORT RETENTION ANALYSIS
# ------------------------------------------------------------------
def months_between(d1, d2):
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)

max_age = months_between(TODAY, COHORT_START)
cohort_sizes = df.groupby("signup_cohort")["customer_id"].count()

retention_matrix = pd.DataFrame(
    index=sorted(df["signup_cohort"].unique()),
    columns=range(0, max_age + 1),
    dtype=float,
)

for cohort, group in df.groupby("signup_cohort"):
    cohort_start = pd.Timestamp(cohort + "-01")
    size = len(group)
    age_limit = months_between(TODAY, cohort_start)
    for age in range(0, age_limit + 1):
        # active at this "age" if not churned yet, or churned after this age
        still_active = group[
            (~group["churned"]) | (group["tenure_months"] > age)
        ]
        retention_matrix.loc[cohort, age] = len(still_active) / size

retention_matrix = retention_matrix.dropna(axis=1, how="all")

# ------------------------------------------------------------------
# 6. CHURN RATE OVER CALENDAR TIME
# ------------------------------------------------------------------
calendar_months = pd.date_range(COHORT_START, TODAY, freq="MS")
monthly_churn_rate = []
monthly_active_count = []
for m in calendar_months:
    active_at_start = df[(df["signup_date"] <= m)]
    active_at_start = active_at_start[
        (~active_at_start["churned"]) | (active_at_start["churn_date"] >= m)
    ]
    churned_this_month = active_at_start[
        active_at_start["churned"]
        & (active_at_start["churn_date"].dt.to_period("M") == m.to_period("M"))
    ]
    rate = len(churned_this_month) / len(active_at_start) if len(active_at_start) else np.nan
    monthly_churn_rate.append(rate)
    monthly_active_count.append(len(active_at_start))

churn_trend = pd.DataFrame({
    "month": calendar_months,
    "churn_rate": monthly_churn_rate,
    "active_customers": monthly_active_count,
})

# ==================================================================
# VISUALIZATIONS
# ==================================================================
import os
os.makedirs(CHART_DIR, exist_ok=True)

# --- Chart 1: Churn rate by plan type ---
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = sns.color_palette("rocket", len(churn_by_plan))
bars = ax.bar(churn_by_plan.index, churn_by_plan.values * 100, color=colors)
for b in bars:
    ax.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Churn Rate by Plan Type", fontsize=13, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Plan Type")
ax.set_ylim(0, max(churn_by_plan.values * 100) * 1.25)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_churn_by_plan.png")
plt.show()
plt.close()

# --- Chart 2: Churn rate by acquisition channel ---
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = sns.color_palette("rocket", len(churn_by_channel))
bars = ax.bar(churn_by_channel.index, churn_by_channel.values * 100, color=colors)
for b in bars:
    ax.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Churn Rate by Acquisition Channel", fontsize=13, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Acquisition Channel")
ax.set_ylim(0, max(churn_by_channel.values * 100) * 1.25)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_churn_by_channel.png")
plt.show()
plt.close()

# --- Chart 3: Churn rate by usage frequency bucket (key retention driver) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = sns.color_palette("crest", len(churn_by_usage))
bars = ax.bar(churn_by_usage.index.astype(str), churn_by_usage.values * 100, color=colors)
for b in bars:
    ax.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Churn Rate by Usage Frequency", fontsize=13, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Avg. Sessions per Month")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_churn_by_usage.png")
plt.show()
plt.close()

# --- Chart 4: Churn rate by support tickets ---
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = sns.color_palette("flare", len(churn_by_tickets))
bars = ax.bar(churn_by_tickets.index, churn_by_tickets.values * 100, color=colors)
for b in bars:
    ax.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Churn Rate by Monthly Support Tickets", fontsize=13, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Support Tickets per Month")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_churn_by_support_tickets.png")
plt.show()
plt.close()

# --- Chart 5: Retention driver impact (logistic regression coefficients) ---
fig, ax = plt.subplots(figsize=(8, 5))
label_map = {
    "avg_sessions_per_month": "Usage frequency",
    "support_tickets_per_month": "Support tickets",
    "channel_paid_ads": "Channel: Paid Ads",
    "channel_referral": "Channel: Referral",
    "channel_partner": "Channel: Partner",
    "plan_standard": "Plan: Standard",
    "plan_premium": "Plan: Premium",
}
labels = [label_map[i] for i in driver_impact.index]
colors = ["#2a9d8f" if v < 0 else "#e76f51" for v in driver_impact.values]
ax.barh(labels, driver_impact.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Retention Drivers: Impact on Churn Odds\n(standardized logistic regression)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Coefficient  (negative = reduces churn, positive = increases churn)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_retention_drivers.png")
plt.show()
plt.close()

# --- Chart 6: Cohort retention heatmap ---
fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(retention_matrix * 100, annot=True, fmt=".0f", cmap="YlGnBu",
            cbar_kws={"label": "% Retained"}, linewidths=0.4, linecolor="white", ax=ax)
ax.set_title("Cohort Retention Heatmap (% of cohort still active by month since signup)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Months Since Signup")
ax.set_ylabel("Signup Cohort (Month)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_cohort_retention_heatmap.png")
plt.show()
plt.close()

# --- Chart 7: Monthly churn rate trend over calendar time ---
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(churn_trend["month"], churn_trend["churn_rate"] * 100,
        marker="o", markersize=3, color="#e63946", linewidth=1.6)
ax.set_title("Monthly Churn Rate Over Time", fontsize=13, fontweight="bold")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Month")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_monthly_churn_trend.png")
plt.show()
plt.close()

# --- Chart 8: Customer lifetime distribution (churned customers) ---
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.histplot(df.loc[df["churned"], "tenure_months"], bins=24, color="#457b9d", ax=ax)
ax.axvline(avg_lifetime_churned, color="red", linestyle="--",
           label=f"Mean = {avg_lifetime_churned:.1f} mo")
ax.set_title("Customer Lifetime Distribution (Churned Customers)", fontsize=13, fontweight="bold")
ax.set_xlabel("Tenure at Churn (months)")
ax.set_ylabel("Number of Customers")
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_lifetime_distribution.png")
plt.show()
plt.close()

print(f"\nAll charts saved to ./{CHART_DIR}/")

# ------------------------------------------------------------------
# 7. EXPORT KEY METRICS FOR THE REPORT
# ------------------------------------------------------------------
summary = {
    "overall_churn_rate": overall_churn_rate,
    "total_customers": len(df),
    "active_customers": int((~df["churned"]).sum()),
    "churned_customers": int(df["churned"].sum()),
    "avg_lifetime_churned_months": avg_lifetime_churned,
    "median_lifetime_churned_months": median_lifetime_churned,
    "avg_tenure_active_months": avg_tenure_active,
    "approx_clv_inr": approx_clv,
    "churn_by_plan": churn_by_plan.to_dict(),
    "churn_by_channel": churn_by_channel.to_dict(),
    "churn_by_usage": {str(k): v for k, v in churn_by_usage.to_dict().items()},
    "churn_by_tickets": churn_by_tickets.to_dict(),
    "churn_by_payment": churn_by_payment.to_dict(),
    "driver_impact": driver_impact.to_dict(),
}

import json
with open("summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nSummary metrics exported -> summary_metrics.json")
print("\n=== KEY METRICS ===")
for k, v in summary.items():
    if not isinstance(v, dict):
        print(f"{k}: {v}")