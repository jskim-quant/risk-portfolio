# Visual review

Reviewed on 2026-10-01. Scope: 32 embedded PNG figures across notebooks 01–11. The review covered rendered labels, legends, units, scales and the plotting code. Revised figures were inspected again after execution.

## Corrections

| Location | Finding | Change |
|---|---|---|
| All notebooks | Inconsistent text sizes | Matplotlib default fonts and sizes; wider figures for long labels |
| 01 | Obsolete section reference and imprecise GPD boundary labels | Removed the reference; clarified that moment failure includes the boundary |
| 02 | Legend covered part of an ES error bar | Moved the legend below the plot |
| 05 | Bar lengths on a logarithmic ES axis could mislead | Used points; retained the labelled logarithmic scale |
| 01, 05 | Model curves relied on colour alone | Added different line styles |
| 05 | Small multiples used different axis ranges | Disclosed independent ranges in the title |
| 02, 06, 07, 09 | Fractional shares and ratios required mental conversion | Used percentage tick labels |
| 08, 09 | Angled or omitted year labels | Used horizontal year labels and included every plotted year |
| 11 | Capital decomposition touched the upper axis boundary | Set explicit headroom and kept signed step labels |
| 11 | Small sensitivity effects were hard to read | Added numerical labels |
| 11 | Finite simulated ES could be mistaken for theoretical ES | Added an explicit warning for GPD shape at or above 1 |
| 11 | AY2001 was labelled as a confirmed data artifact | Changed the label to AY2001 inclusion; the cause remains unresolved |

## Checks

- All 11 notebooks executed in fresh kernels: 141 code cells, no error outputs.
- Existing test suite: 55 passed.
- All registered result files passed source and content freshness checks.
- Compared 1,419 saved numerical values with commit `57b2a1d3bca85a4f24d3b4e4e554d68b6c50d69d`. All agreed within relative tolerance `1e-7` and absolute tolerance `1e-8`. The largest scaled difference was `4.50e-8`, in a motor bootstrap standard-deviation estimate.
- Read plotting inputs against their tables and result fields, including EUR versus USD units, percentage scaling, simulated ES, reserve runoff and capital-step reconciliation.
- Replicate standard deviations remain labelled as spread, not confidence intervals. Lognormal reserve intervals remain identified as an additional assumption.

Execution details are in [validation.json](validation.json). Numerical checks and the figure inventory are in [visual_checks.json](visual_checks.json); cell indices are zero-based.

## Scope limits

The review covers saved notebook images. IDE zoom levels, mobile layouts, PDF/HTML exports and formal colour-vision simulations were not tested. Plot corrections do not independently validate the statistical models or source data.
