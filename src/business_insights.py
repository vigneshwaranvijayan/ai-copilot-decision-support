"""Domain-aware business insight and recommendation engine.

The functions here are intentionally transparent and data-grounded. They do not
use an unrestricted LLM and do not fetch outside market/HR/customer knowledge.
Every recommendation is derived from columns and values in the uploaded active
DataFrame, with a safety warning for human review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px


@dataclass
class BusinessInsightResult:
    domain: str
    answer: str
    table: pd.DataFrame
    chart: Any = None
    warning: str = "These are decision-support suggestions based only on the uploaded dataset. A human reviewer must check context before action."
    context: Dict[str, Any] | None = None


def _norm(s: Any) -> str:
    import re
    text = "" if s is None else str(s).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact(s: Any) -> str:
    return _norm(s).replace(" ", "")


def _find_cols(df: pd.DataFrame, keywords: Sequence[str], *, numeric: Optional[bool] = None) -> List[str]:
    out: List[str] = []
    keys = [_compact(k) for k in keywords]
    for col in df.columns:
        c = _compact(col)
        if any(k in c for k in keys):
            if numeric is True and not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if numeric is False and pd.api.types.is_numeric_dtype(df[col]):
                continue
            out.append(col)
    return out


def _first_col(df: pd.DataFrame, keywords: Sequence[str], *, numeric: Optional[bool] = None) -> Optional[str]:
    cols = _find_cols(df, keywords, numeric=numeric)
    return cols[0] if cols else None


def _date_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        cname = _compact(col)
        if any(k in cname for k in ["date", "invoice", "orderdate", "transactiondate", "created", "month", "year"]):
            cols.append(col)
    # include dtype datetime if user already loaded it that way
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        if col not in cols:
            cols.append(col)
    return cols


def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _safe_sum_numeric(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def detect_business_domain(df: pd.DataFrame, question: str = "") -> str:
    """Return likely domain for suggestion logic.

    Domains: sales_retail, employee_hr, customer_feedback, customer_churn,
    bank_marketing, operations_service, general.
    """
    q = _compact(question)
    cols = " ".join(_compact(c) for c in df.columns)

    # Question terms get priority because one dataset may contain several fields.
    if any(k in q for k in ["employee", "employe", "salary", "promotion", "resign", "attrition", "hr", "overtime"]):
        return "employee_hr"
    if any(k in q for k in ["sale", "sales", "product", "purchase", "stock", "revenue", "profit", "forecast", "nextmonth", "sell"]):
        return "sales_retail"
    if any(k in q for k in ["feedback", "review", "comment", "complaint", "customerfeedback"]):
        return "customer_feedback"
    if any(k in q for k in ["marketing", "campaign", "subscribe", "subscription", "bank"]):
        return "bank_marketing"
    if any(k in q for k in ["ticket", "service", "complaint", "operation", "delay", "resolution"]):
        return "operations_service"

    if any(k in cols for k in ["quantity", "unitprice", "sales", "revenue", "invoice", "stockcode", "product", "description", "order"]):
        return "sales_retail"
    if any(k in cols for k in ["employee", "monthlyincome", "salary", "jobsatisfaction", "performance", "overtime", "attrition"]):
        return "employee_hr"
    if any(k in cols for k in ["customerfeedback", "feedback", "review", "comment"]):
        return "customer_feedback"
    if any(k in cols for k in ["churn", "tenure", "monthlycharges", "contract"]):
        return "customer_churn"
    if any(k in cols for k in ["campaign", "pdays", "poutcome", "subscription", "deposit"]) or " y " in f" {cols} ":
        return "bank_marketing"
    if any(k in cols for k in ["ticket", "complaint", "status", "resolution", "priority", "sla", "delay"]):
        return "operations_service"
    return "general"


def _date_product_value_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    date_col = None
    for col in _date_cols(df):
        parsed = _to_datetime_series(df[col])
        if parsed.notna().sum() >= max(5, int(len(df) * 0.2)):
            date_col = col
            break
    product_col = _first_col(df, ["product", "item", "description", "stockcode", "sku", "article", "category"], numeric=False)
    quantity_col = _first_col(df, ["quantity", "qty", "units", "volume"], numeric=True)
    value_col = _first_col(df, ["sales", "revenue", "amount", "total", "price", "unitprice", "profit"], numeric=True)
    # For retail invoice data, compute revenue later if Quantity and UnitPrice exist.
    return date_col, product_col, quantity_col, value_col


def sales_retail_insights(df: pd.DataFrame, dataset_name: str = "active dataset") -> BusinessInsightResult:
    date_col, product_col, quantity_col, value_col = _date_product_value_columns(df)
    work = df.copy()
    if date_col:
        work["__date"] = _to_datetime_series(work[date_col])
        work = work[work["__date"].notna()].copy()
    else:
        work["__date"] = pd.NaT

    # Create value metric. If both quantity and unit price are present, prefer sales_value.
    unit_price = _first_col(work, ["unitprice", "unit price", "price"], numeric=True)
    if quantity_col and unit_price and quantity_col != unit_price:
        work["__sales_value"] = pd.to_numeric(work[quantity_col], errors="coerce").fillna(0) * pd.to_numeric(work[unit_price], errors="coerce").fillna(0)
        metric_col = "__sales_value"
        metric_name = "estimated_sales_value"
    elif value_col:
        work["__sales_value"] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
        metric_col = "__sales_value"
        metric_name = str(value_col)
    elif quantity_col:
        work["__sales_value"] = pd.to_numeric(work[quantity_col], errors="coerce").fillna(0)
        metric_col = "__sales_value"
        metric_name = str(quantity_col)
    else:
        nums = work.select_dtypes(include=np.number).columns.tolist()
        if nums:
            metric_col = nums[0]
            metric_name = str(nums[0])
        else:
            table = pd.DataFrame({"issue": ["No numeric sales/quantity/value column found"], "suggested_action": ["Upload sales data with quantity, revenue, amount, unit price, or profit columns."]})
            return BusinessInsightResult("sales_retail", f"I detected a sales/retail scenario, but I need a numeric sales/quantity/value column to generate stock or sales suggestions. Source used: `{dataset_name}`.", table)

    if not product_col:
        # monthly trend only, no product recommendation.
        if date_col and not work.empty:
            work["month"] = work["__date"].dt.to_period("M").astype(str)
            monthly = work.groupby("month", as_index=False)[metric_col].sum().rename(columns={metric_col: metric_name})
            if len(monthly) >= 2:
                last = monthly.iloc[-1][metric_name]
                prev = monthly.iloc[-2][metric_name]
                change = ((last - prev) / abs(prev) * 100) if prev else np.nan
                table = pd.DataFrame([{ "finding": "Latest monthly trend", "evidence": f"{monthly.iloc[-2]['month']}={prev:,.2f}; {monthly.iloc[-1]['month']}={last:,.2f}", "suggested_action": "Use the latest trend for stock planning, but check seasonality and promotions before purchasing.", "change_percent": round(change, 2) if not np.isnan(change) else None }])
                chart = px.line(monthly, x="month", y=metric_name, title=f"Monthly {metric_name}")
                return BusinessInsightResult("sales_retail", f"I found a sales time trend but no product/category column, so suggestions are at overall sales level only. Source used: `{dataset_name}`.", table, chart)
        table = pd.DataFrame({"suggested_action": ["Add a product/category column to generate product-specific purchase and improvement suggestions."]})
        return BusinessInsightResult("sales_retail", f"I detected sales data, but no product/category field was found. Source used: `{dataset_name}`.", table)

    # Product-level insights.
    work[product_col] = work[product_col].astype(str).str.slice(0, 80)
    rows: List[Dict[str, Any]] = []

    top_products = work.groupby(product_col, dropna=False, as_index=False)[metric_col].sum().sort_values(metric_col, ascending=False).head(8)
    for _, r in top_products.iterrows():
        rows.append({
            "priority": "High" if r[metric_col] > top_products[metric_col].median() else "Medium",
            "insight_type": "Strong seller",
            "product_or_segment": r[product_col],
            "evidence": f"Top total {metric_name}: {r[metric_col]:,.2f}",
            "business_suggestion": "Protect stock availability, monitor supplier lead time, and consider promotion/bundling if margin is acceptable.",
            "human_review_warning": "Do not over-purchase without checking inventory, margin, seasonality and returns.",
        })

    if date_col and not work.empty:
        work["month"] = work["__date"].dt.to_period("M").astype(str)
        months = sorted(work["month"].dropna().unique())
        if len(months) >= 2:
            latest = months[-1]
            previous = months[-2]
            piv = work[work["month"].isin([previous, latest])].groupby([product_col, "month"])[metric_col].sum().unstack(fill_value=0)
            if previous in piv.columns and latest in piv.columns:
                piv["change"] = piv[latest] - piv[previous]
                piv["change_percent"] = np.where(piv[previous] != 0, piv["change"] / piv[previous].abs() * 100, np.nan)
                improving = piv.sort_values("change", ascending=False).head(5).reset_index()
                declining = piv[piv[previous] > 0].sort_values("change", ascending=True).head(5).reset_index()
                for _, r in improving.iterrows():
                    rows.append({
                        "priority": "Medium",
                        "insight_type": "Recent growth",
                        "product_or_segment": r[product_col],
                        "evidence": f"{previous}: {r[previous]:,.2f}; {latest}: {r[latest]:,.2f}; change {r['change']:,.2f}",
                        "business_suggestion": "Consider increasing near-term stock or visibility because recent demand improved.",
                        "human_review_warning": "Check whether the growth was caused by a one-off promotion or unusual order.",
                    })
                for _, r in declining.iterrows():
                    rows.append({
                        "priority": "Medium",
                        "insight_type": "Declining product",
                        "product_or_segment": r[product_col],
                        "evidence": f"{previous}: {r[previous]:,.2f}; {latest}: {r[latest]:,.2f}; change {r['change']:,.2f}",
                        "business_suggestion": "Review price, product listing, promotion, availability, and customer feedback before reordering at the same level.",
                        "human_review_warning": "Decline can be seasonal; compare with the same month last year if available.",
                    })

        # Same-month last-year evidence if possible.
        work["year"] = work["__date"].dt.year
        work["month_num"] = work["__date"].dt.month
        years = sorted(work["year"].dropna().unique())
        if len(years) >= 2:
            latest_year = years[-1]
            latest_month = int(work.loc[work["year"] == latest_year, "month_num"].max())
            last_year = latest_year - 1
            same_month = work[(work["month_num"] == latest_month) & (work["year"].isin([last_year, latest_year]))]
            if not same_month.empty and same_month["year"].nunique() >= 2:
                sy = same_month.groupby([product_col, "year"])[metric_col].sum().unstack(fill_value=0)
                if last_year in sy.columns and latest_year in sy.columns:
                    sy["year_change"] = sy[latest_year] - sy[last_year]
                    moved_last_year_bad_now = sy[(sy[last_year] > 0) & (sy[latest_year] < sy[last_year])].sort_values("year_change").head(5).reset_index()
                    for _, r in moved_last_year_bad_now.iterrows():
                        rows.append({
                            "priority": "Medium",
                            "insight_type": "Good last year, weaker now",
                            "product_or_segment": r[product_col],
                            "evidence": f"Same month {last_year}: {r[last_year]:,.2f}; {latest_year}: {r[latest_year]:,.2f}",
                            "business_suggestion": "Investigate why this product is weaker than last year: pricing, availability, competitor alternatives, listing quality or promotion timing.",
                            "human_review_warning": "This uses uploaded historical data only, not external market demand.",
                        })

    table = pd.DataFrame(rows).drop_duplicates(subset=["insight_type", "product_or_segment"]).head(25)
    if table.empty:
        table = pd.DataFrame({"business_suggestion": ["No product-level pattern found. Try a longer date range or clearer product/quantity/revenue columns."]})
    chart = None
    if not top_products.empty:
        chart_df = top_products.rename(columns={metric_col: metric_name})
        chart = px.bar(chart_df, x=product_col, y=metric_name, title=f"Top products by {metric_name}")
    answer = (
        f"I generated sales/retail business suggestions from `{product_col}` and `{metric_name}`. "
        "The suggestions cover strong sellers, recent growth, declining products and stock/purchase review signals where dates are available. "
        f"Source used: `{dataset_name}`."
    )
    return BusinessInsightResult("sales_retail", answer, table, chart, context={"domain": "sales_retail", "product_col": product_col, "metric_col": metric_name})


def employee_hr_insights(df: pd.DataFrame, dataset_name: str = "active dataset", target_column: Optional[str] = None, positive_label: Optional[Any] = None) -> BusinessInsightResult:
    id_col = _first_col(df, ["employeeid", "employee id", "employee", "id"])
    salary_col = _first_col(df, ["monthlyincome", "salary", "income", "pay", "rate"], numeric=True)
    perf_col = _first_col(df, ["performancerating", "performance", "rating", "score"], numeric=True)
    overtime_col = _first_col(df, ["overtime", "over time"], numeric=False)
    satisfaction_col = _first_col(df, ["jobsatisfaction", "satisfaction", "environment", "worklife"], numeric=True)
    years_col = _first_col(df, ["yearsatcompany", "tenure", "years", "experience"], numeric=True)
    attrition_col = target_column if target_column and _compact(target_column) in ["attrition", "resign", "resignation", "left", "leaver"] else _first_col(df, ["attrition", "resign", "left", "leaver"])

    work = df.copy()
    rows: List[Dict[str, Any]] = []
    if id_col is None:
        work["row_index"] = work.index
        id_col = "row_index"

    # Department/group risks.
    dept_col = _first_col(df, ["department", "jobrole", "role", "job", "team"], numeric=False)
    if attrition_col and dept_col:
        pos = positive_label if positive_label is not None else "Yes"
        rates = work.groupby(dept_col)[attrition_col].agg(total="count", attrition_rate=lambda s: (s.astype(str).str.lower() == str(pos).lower()).mean() * 100).reset_index()
        rates = rates.sort_values("attrition_rate", ascending=False).head(8)
        for _, r in rates.iterrows():
            rows.append({
                "priority": "High" if r["attrition_rate"] >= 25 and r["total"] >= 5 else "Medium",
                "insight_type": "Attrition / resignation risk segment",
                "employee_or_segment": r[dept_col],
                "evidence": f"{dept_col} group count={int(r['total'])}; attrition/resignation positive rate={r['attrition_rate']:.1f}%",
                "business_suggestion": "Review workload, manager support, career progression, and retention conversations for this group.",
                "human_review_warning": "Do not make individual HR decisions automatically; use this only for management review and support planning.",
            })

    # High salary with weaker performance = review, not dismissal.
    if salary_col and perf_col:
        tmp = work[[id_col, salary_col, perf_col] + ([dept_col] if dept_col and dept_col in work.columns else [])].copy()
        tmp[salary_col] = pd.to_numeric(tmp[salary_col], errors="coerce")
        tmp[perf_col] = pd.to_numeric(tmp[perf_col], errors="coerce")
        high_salary = tmp[salary_col].quantile(0.75)
        low_perf = tmp[perf_col].quantile(0.35)
        cases = tmp[(tmp[salary_col] >= high_salary) & (tmp[perf_col] <= low_perf)].sort_values([salary_col, perf_col], ascending=[False, True]).head(5)
        for _, r in cases.iterrows():
            rows.append({
                "priority": "Medium",
                "insight_type": "Compensation/performance review signal",
                "employee_or_segment": r[id_col],
                "evidence": f"{salary_col}={r[salary_col]:,.2f}; {perf_col}={r[perf_col]:,.2f}",
                "business_suggestion": "Review role expectations, recent performance evidence, workload, training needs and compensation context with HR/manager input.",
                "human_review_warning": "This must not be used to label a person as useless or not needed; final HR decisions require policy, evidence and human review.",
            })

    # Strong performer/support/promotion review signal.
    if perf_col:
        cols = [id_col, perf_col]
        if salary_col: cols.append(salary_col)
        if years_col: cols.append(years_col)
        if dept_col and dept_col not in cols: cols.append(dept_col)
        tmp = work[cols].copy()
        tmp[perf_col] = pd.to_numeric(tmp[perf_col], errors="coerce")
        perf_hi = tmp[perf_col].quantile(0.85)
        if salary_col:
            tmp[salary_col] = pd.to_numeric(tmp[salary_col], errors="coerce")
            salary_med = tmp[salary_col].median()
            cases = tmp[(tmp[perf_col] >= perf_hi) & (tmp[salary_col] <= salary_med)].sort_values(perf_col, ascending=False).head(5)
            suggestion = "Consider recognition, career-development discussion, pay review eligibility or promotion-readiness review."
        else:
            cases = tmp[tmp[perf_col] >= perf_hi].sort_values(perf_col, ascending=False).head(5)
            suggestion = "Consider recognition, development opportunities or promotion-readiness review."
        for _, r in cases.iterrows():
            evidence = f"{perf_col}={r[perf_col]:,.2f}"
            if salary_col and salary_col in r:
                evidence += f"; {salary_col}={r[salary_col]:,.2f}"
            rows.append({
                "priority": "Medium",
                "insight_type": "High performer support/reward review",
                "employee_or_segment": r[id_col],
                "evidence": evidence,
                "business_suggestion": suggestion,
                "human_review_warning": "Promotion or salary decisions require manager/HR review, role criteria and fairness checks.",
            })

    # Burnout risk.
    if overtime_col or satisfaction_col:
        cols = [id_col]
        for c in [overtime_col, satisfaction_col, attrition_col, dept_col]:
            if c and c in work.columns and c not in cols:
                cols.append(c)
        tmp = work[cols].copy()
        mask = pd.Series(False, index=tmp.index)
        if overtime_col:
            mask = mask | tmp[overtime_col].astype(str).str.lower().isin(["yes", "true", "1", "y"])
        if satisfaction_col:
            sat = pd.to_numeric(tmp[satisfaction_col], errors="coerce")
            mask = mask | (sat <= sat.quantile(0.25))
        cases = tmp[mask].head(8)
        for _, r in cases.iterrows():
            rows.append({
                "priority": "High" if satisfaction_col and pd.to_numeric(pd.Series([r.get(satisfaction_col)]), errors="coerce").iloc[0] <= pd.to_numeric(work[satisfaction_col], errors="coerce").quantile(0.25) else "Medium",
                "insight_type": "Burnout/support risk signal",
                "employee_or_segment": r[id_col],
                "evidence": "; ".join(f"{c}={r[c]}" for c in cols if c != id_col),
                "business_suggestion": "Schedule supportive manager check-in, review workload/overtime, and consider wellbeing or training support.",
                "human_review_warning": "This is not a disciplinary decision; it is a support and retention review signal.",
            })

    table = pd.DataFrame(rows).head(30)
    if table.empty:
        table = pd.DataFrame({"business_suggestion": ["I could not find enough HR-specific columns. Useful columns include employee ID, salary, performance rating, overtime, job satisfaction, department and attrition."]})
    chart = None
    if attrition_col and dept_col and not table.empty:
        rates = work.groupby(dept_col)[attrition_col].agg(count="count", positive_rate=lambda s: (s.astype(str).str.lower() == str(positive_label if positive_label is not None else 'yes').lower()).mean() * 100).reset_index().sort_values("positive_rate", ascending=False).head(10)
        chart = px.bar(rates, x=dept_col, y="positive_rate", hover_data=["count"], title=f"{attrition_col} rate by {dept_col}")
    answer = (
        "I generated HR/employee decision-support suggestions from the uploaded data. "
        "The output focuses on support, retention, review and development actions, not automatic hiring/firing decisions. "
        f"Source used: `{dataset_name}`."
    )
    return BusinessInsightResult("employee_hr", answer, table, chart, warning="HR outputs are high-risk decision support only. Do not use this system for dismissal, demotion, promotion or pay decisions without human HR review, fairness checks and proper evidence.", context={"domain": "employee_hr"})


def marketing_insights(df: pd.DataFrame, dataset_name: str = "active dataset", target_column: Optional[str] = None, positive_label: Optional[Any] = None) -> BusinessInsightResult:
    target = target_column or _first_col(df, ["y", "subscribed", "subscription", "deposit", "response", "converted"])
    if not target or target not in df.columns:
        table = pd.DataFrame({"business_suggestion": ["Select a yes/no target such as subscription response (`y`) to generate marketing targeting suggestions."]})
        return BusinessInsightResult("bank_marketing", f"I detected marketing/banking data but no binary response target is selected. Source used: `{dataset_name}`.", table)
    pos = positive_label if positive_label is not None else "yes"
    rows: List[Dict[str, Any]] = []
    categorical = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    for col in categorical[:8]:
        grp = df.groupby(col)[target].agg(total="count", positive_rate=lambda s: (s.astype(str).str.lower() == str(pos).lower()).mean() * 100).reset_index()
        grp = grp[grp["total"] >= max(5, int(len(df) * 0.005))].sort_values("positive_rate", ascending=False).head(3)
        for _, r in grp.iterrows():
            rows.append({
                "priority": "High" if r["positive_rate"] >= 20 else "Medium",
                "insight_type": "High response segment",
                "customer_segment": f"{col} = {r[col]}",
                "evidence": f"{int(r['total'])} records; positive response rate={r['positive_rate']:.1f}%",
                "business_suggestion": "Prioritise this segment for campaign testing or personalised follow-up, subject to consent and contact policy.",
                "human_review_warning": "Avoid unfair targeting or over-contacting. Follow marketing consent and governance rules.",
            })
    for col in numeric[:5]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() < 10:
            continue
        high = vals >= vals.quantile(0.75)
        low = vals <= vals.quantile(0.25)
        for label, mask in [("high", high), ("low", low)]:
            if mask.sum() < 5:
                continue
            rate = (df.loc[mask, target].astype(str).str.lower() == str(pos).lower()).mean() * 100
            rows.append({
                "priority": "Medium",
                "insight_type": "Numeric response pattern",
                "customer_segment": f"{label} {col}",
                "evidence": f"{int(mask.sum())} records; positive response rate={rate:.1f}%",
                "business_suggestion": "Compare this segment with campaign cost and customer suitability before prioritising outreach.",
                "human_review_warning": "Correlation is not causation; check whether this variable is appropriate for campaign targeting.",
            })
    table = pd.DataFrame(rows).sort_values("evidence", ascending=False).head(25) if rows else pd.DataFrame({"business_suggestion": ["No clear marketing segments found. Try selecting the target and checking campaign/segment columns."]})
    chart = None
    if rows:
        temp = pd.DataFrame(rows).head(10).copy()
        temp["rate"] = temp["evidence"].str.extract(r"rate=([0-9.]+)%").astype(float)
        chart = px.bar(temp, x="customer_segment", y="rate", color="priority", title="Marketing response suggestions")
    return BusinessInsightResult("bank_marketing", f"I generated campaign/marketing suggestions from response target `{target}`. Source used: `{dataset_name}`.", table, chart, warning="Marketing suggestions are decision support only. Check consent, fairness and campaign policy before acting.", context={"domain": "bank_marketing", "target_column": target})


def operations_insights(df: pd.DataFrame, dataset_name: str = "active dataset") -> BusinessInsightResult:
    category_col = _first_col(df, ["category", "complaint", "type", "issue", "service", "request", "ticket"], numeric=False)
    area_col = _first_col(df, ["area", "region", "location", "city", "borough", "branch"], numeric=False)
    status_col = _first_col(df, ["status", "resolution", "closed", "resolved"], numeric=False)
    date_col = None
    for c in _date_cols(df):
        if _to_datetime_series(df[c]).notna().sum() >= 5:
            date_col = c
            break
    rows: List[Dict[str, Any]] = []
    if category_col:
        counts = df[category_col].astype(str).value_counts().head(8)
        for cat, count in counts.items():
            rows.append({
                "priority": "High" if count >= counts.median() else "Medium",
                "insight_type": "High-volume service issue",
                "area_or_issue": cat,
                "evidence": f"{int(count)} records in `{category_col}`",
                "business_suggestion": "Investigate root cause, allocate support capacity, prepare standard response guidance, and monitor resolution trend.",
                "human_review_warning": "Operational context and severity should be reviewed before changing resources.",
            })
    if area_col and category_col:
        combo = df.groupby([area_col, category_col]).size().reset_index(name="count").sort_values("count", ascending=False).head(8)
        for _, r in combo.iterrows():
            rows.append({
                "priority": "Medium",
                "insight_type": "Location/issue hotspot",
                "area_or_issue": f"{area_col}={r[area_col]} | {category_col}={r[category_col]}",
                "evidence": f"{int(r['count'])} records",
                "business_suggestion": "Check local process, staffing, supplier or service-quality factors for this hotspot.",
                "human_review_warning": "Avoid blaming individuals or locations without operational investigation.",
            })
    if status_col:
        status_counts = df[status_col].astype(str).value_counts(normalize=False).head(8)
        for stat, count in status_counts.items():
            if any(k in str(stat).lower() for k in ["open", "pending", "delayed", "unresolved"]):
                rows.append({
                    "priority": "High",
                    "insight_type": "Unresolved/pending work",
                    "area_or_issue": stat,
                    "evidence": f"{int(count)} records with status `{stat}`",
                    "business_suggestion": "Prioritise backlog review, escalation rules and staffing coverage for unresolved work.",
                    "human_review_warning": "Check record age and severity before escalation.",
                })
    table = pd.DataFrame(rows).head(25) if rows else pd.DataFrame({"business_suggestion": ["No clear operations columns found. Useful columns include category, issue type, area/location, status, date and resolution time."]})
    chart = None
    if category_col:
        counts = df[category_col].astype(str).value_counts().head(10).reset_index()
        counts.columns = [category_col, "count"]
        chart = px.bar(counts, x=category_col, y="count", title=f"Top {category_col} values")
    return BusinessInsightResult("operations_service", f"I generated operational improvement suggestions from the uploaded service/operations data. Source used: `{dataset_name}`.", table, chart, context={"domain": "operations_service"})


def general_business_insights(df: pd.DataFrame, dataset_name: str = "active dataset") -> BusinessInsightResult:
    rows: List[Dict[str, Any]] = []
    # Missing data improvement.
    missing = df.isna().sum().sort_values(ascending=False)
    for col, count in missing[missing > 0].head(5).items():
        rows.append({
            "priority": "Medium",
            "insight_type": "Data quality improvement",
            "area": col,
            "evidence": f"{int(count)} missing cells",
            "business_suggestion": "Improve data capture/validation for this field before relying on analysis or model results.",
            "human_review_warning": "Missing values can bias conclusions; check source systems.",
        })
    # High-cardinality categoricals.
    for col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])][:8]:
        top = df[col].astype(str).value_counts(dropna=False).head(1)
        if not top.empty:
            rows.append({
                "priority": "Low",
                "insight_type": "Dominant category check",
                "area": col,
                "evidence": f"Most common value `{top.index[0]}` appears {int(top.iloc[0])} times",
                "business_suggestion": "Review whether this concentration is expected or indicates operational/data-capture imbalance.",
                "human_review_warning": "This is exploratory only, not a final business decision.",
            })
    if not rows:
        rows.append({
            "priority": "Low",
            "insight_type": "General analysis",
            "area": "dataset",
            "evidence": f"{len(df):,} rows and {df.shape[1]} columns",
            "business_suggestion": "Ask a more specific question such as highest numeric value, target by segment, missing values, or model performance.",
            "human_review_warning": "Use domain context before acting.",
        })
    table = pd.DataFrame(rows).head(20)
    return BusinessInsightResult("general", f"I generated general business/data-quality suggestions from the active dataset. Source used: `{dataset_name}`.", table, None, context={"domain": "general"})


def business_insight_engine(
    df: pd.DataFrame,
    question: str = "",
    dataset_name: str = "active dataset",
    target_column: Optional[str] = None,
    positive_label: Optional[Any] = None,
) -> BusinessInsightResult:
    domain = detect_business_domain(df, question)
    if domain == "sales_retail":
        return sales_retail_insights(df, dataset_name)
    if domain == "employee_hr":
        return employee_hr_insights(df, dataset_name, target_column=target_column, positive_label=positive_label)
    if domain in {"bank_marketing"}:
        return marketing_insights(df, dataset_name, target_column=target_column, positive_label=positive_label)
    if domain == "operations_service":
        return operations_insights(df, dataset_name)
    # Feedback/churn is handled in copilot.py for feedback-specific themes, but
    # general business suggestions can still go to feedback domain via caller.
    return general_business_insights(df, dataset_name)
