---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Julia 1.3.0
    language: julia
    name: julia-1.3
---

```julia
using DataFrames
using DataFramesMeta
using Distributions
using VegaLite
using Base.Threads

include("./impact_modeling.jl")
IM = ImpactModeling
```

```julia
MEAN_NORM = get(ENV, "MEAN_NORM", 20)::Real;
MEAN_LOGNORM = get(ENV, "MEAN_LOGNORM", 1)::Real;
```

## Repeat basic analysis


TODO: model fraction of the top and the second applicant lost

```julia
n_applicants = [2, 5, 10, 25, 50, 100, 200, 300, 400, 500];
c_scales = range(0.1, 3.0; step=0.1);
imp_dfs = Array{DataFrame, 1}(undef, 2 * length(c_scales));
dens_dfs = Array{DataFrame, 1}(undef, 2 * length(c_scales));

@time @threads for (i, s) in collect(enumerate(c_scales))
    for (j, m, d, lq) in zip(1:2, [MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal], [0.01, 0.001])
        c_i = (i - 1) * 2 + j
        c_dist = d(m, s)
        imp_dfs[c_i] = IM.estimate_applicant_impact_summary(5000, n_applicants, c_dist)
        imp_dfs[c_i][!, :scale] .= s
        imp_dfs[c_i][!, :distribution] .= "$d"

        c_x = range(quantile.(c_dist, [lq, 0.99])..., length=100);
        dens_dfs[c_i] = DataFrame(:x => c_x, :d => pdf.(c_dist, c_x), :scale => s, :distribution => "$d")
    end
end

imp_df = vcat(imp_dfs...);
dens_df = vcat(dens_dfs...);
```

```julia
plt = @vlplot(config={axisY={minExtent=30, maxExtent=30}}) + hcat(
    imp_df |>
    @vlplot(
        x={:n_applicants, title="#Applicants", type="quantitative"},
        y={:median, title="Impact", type="quantitative"},
        transform = [{filter = {selection = :ScaleDist}}],
        width=500, height=300
    ) +
    @vlplot(
        mark={:line, point=true},
        selection = {
            ScaleDist = {
                type = "single", fields = ["scale", "distribution"],
                init = {scale = c_scales[1], distribution="Normal"},
                bind = {scale = {input = "range", min=minimum(c_scales), max=maximum(c_scales), step=diff(c_scales)[1], name="Scale"}, 
                        distribution = {input = "select", options=["Normal", "LogNormal"], name="Distribution"}}
            }
        }
    ) +
    @vlplot(
        mark={:errorbar, ticks=true},
        y={:UQ, type="quantitative", title=""},
        y2={:LQ}
    ),
    dens_df |> @vlplot(
        width=300, height=300,
        transform = [{filter = {selection = :ScaleDist}}],
        x={:x, title="Impact", type="quantitative"},
        y={:d, title="Density", type="quantitative"},
        mark=:line
    )
)
```

## Fraction of impact of top organizations


Particularly, this section can be considered as answer on the question "How many impact we'd loose if every applicant give up if not accepted to the one of top organization instead of applying to lower-rank ones".


**TODO:** it's not size-normaized, so to understand where to focus we need to use "potentials" instead of absolute impacts

```julia
org_impact_scales = range(0.25, 3.0; step=0.25);
org_impact_dfs = DataFrame[];

c_lock = SpinLock();
@time @threads for s in org_impact_scales
    for (m, d) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
        df = vcat(IM.organization_impact_fraction.(5000, [2, 5, 10, 50, 100, 1000], d(m, s))...)
        df[!, :scale] .= s
        df[!, :distribution] .= "$d"

        lock(c_lock)
        push!(org_impact_dfs, df)
        unlock(c_lock)
    end
end

org_impact_df = vcat(org_impact_dfs...);
```

```julia
vln_base = @vlplot(
    x={:organization_id, title="Organizations id", type="quantitative"},
    color={:n_organizations, scale={scheme = "category20b"}, type="nominal"},
    transform = [{filter = {selection = :ScaleDist}}]
)

plt = @where(org_impact_df, :organization_id .<= 20) |> 
@vlplot(config={axisY={minExtent=30, maxExtent=30}}) + hcat(
    vln_base + @vlplot(
        y={:cum_total_median, title="Cumulative fraction of total impact", type="quantitative"},
        mark={:line, point=true},
        selection = {
            ScaleDist = {
                type = "single", fields = ["scale", "distribution"],
                init = {scale = 2.0, distribution="LogNormal"},
                bind = {scale = {input = "range", min=minimum(org_impact_scales), max=maximum(org_impact_scales), step=diff(org_impact_scales)[1], name="Scale"}, 
                        distribution = {input = "select", options=["Normal", "LogNormal"], name="Distribution"}}
            }
        },
        width=350, height=300
    ) +
    @vlplot(
        mark={:errorbar, ticks=true},
        y={:cum_total_UQ, typ="quantitative", title=""},
        y2={:cum_total_LQ}
    ),
    vln_base + @vlplot(
        y={:top_median, title="Fraction of impact of the top organization", type="quantitative"},
        mark={:line, point=true},
        width=350, height=300
    ) +
    @vlplot(
        mark={:errorbar, ticks=true},
        y={:top_UQ, type="quantitative", title=""},
        y2={:top_LQ}
    )
)
```
