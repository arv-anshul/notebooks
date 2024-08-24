from collections.abc import Iterable
from typing import Any, Literal, TypedDict

from scipy import stats

from eda.utils import raise_for_nan

Num_VS_Num_Hypothesis = Literal[
    "kendalltau",
    "pearsonr",
    "spearmanr",
]

Num_VS_Cat_Hypothesis = Literal[
    "f_oneway",
    "kruskal",
]

Cat_VS_Cat_Hypothesis = Literal["chi2_contingency"]


class HypothesisResult(TypedDict):
    p_value: float
    statistic: float
    hypothesis: str
    reject_null_hypothesis: bool


def num_vs_num(
    data1,
    data2,
    hypothesis: Num_VS_Num_Hypothesis = "pearsonr",
    *,
    alpha: float = 0.05,
    **hypothesis_kw,
) -> HypothesisResult:
    raise_for_nan(data1)
    raise_for_nan(data2)

    if not hasattr(stats, hypothesis):
        raise AttributeError(f"No {hypothesis=} in scipy.stats module.")

    result = getattr(stats, hypothesis)(data1, data2, **hypothesis_kw)
    p_value = result.pvalue
    return {
        "p_value": p_value,
        "statistic": result.statistic,
        "hypothesis": hypothesis,
        "reject_null_hypothesis": p_value < alpha,
    }


def num_vs_cat(
    groups: Iterable[Any],
    hypothesis: Num_VS_Cat_Hypothesis = "f_oneway",
    *,
    alpha: float = 0.05,
    **hypothesis_kw,
) -> HypothesisResult:
    if not hasattr(stats, hypothesis):
        raise AttributeError(f"No {hypothesis=} in scipy.stats module.")

    result = getattr(stats, hypothesis)(*groups, **hypothesis_kw)
    p_value = result.pvalue
    return {
        "p_value": p_value,
        "statistic": result.statistic,
        "hypothesis": hypothesis,
        "reject_null_hypothesis": p_value < alpha,
    }


def cat_vs_cat(
    contingency,
    hypothesis: Cat_VS_Cat_Hypothesis = "chi2_contingency",
    *,
    alpha: float = 0.05,
    **hypothesis_kw,
) -> HypothesisResult:
    if not hasattr(stats, hypothesis):
        raise AttributeError(f"No {hypothesis=} in scipy.stats module.")

    result = getattr(stats, hypothesis)(contingency, **hypothesis_kw)
    p_value = result.pvalue
    return {
        "p_value": p_value,
        "statistic": result.statistic,
        "hypothesis": hypothesis,
        "reject_null_hypothesis": p_value < alpha,
    }
