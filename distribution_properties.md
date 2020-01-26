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
import Gadfly

GF = Gadfly

include("./impact_modeling.jl")
IM = ImpactModeling
```

```julia
MEAN_NORM = get(ENV, "MEAN_NORM", 20)::Real;
MEAN_LOGNORM = get(ENV, "MEAN_LOGNORM", 1)::Real;
```

## Distribution properties

```julia
n_objects = 100;
```

### Log-normal

```julia

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
```

```julia


```

```julia
IM.summarize_sample(LogNormal(1, 2), n_objects)
```

```julia
n_applicants = [2, 5, 10, 25, 50, 100, 200, 300, 400, 500];
c_scales = range(0.1, 3.0; step=0.1);
dens_dfs, quant_dfs, ratio_dfs = DataFrame[], DataFrame[], DataFrame[];
c_lock = SpinLock();

@time @threads for (i, s) in collect(enumerate(c_scales))
    for (j, m, d, lq) in zip(1:2, [MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal], [0.01, 0.001])
        c_dist = d(m, s)
        c_x = range(quantile.(c_dist, [lq, 0.99])..., length=100);

        c_quant_df = IM.rotate_df(IM.summarize_sample(c_dist, n_objects), scale=s, distribution="$d")
        
        sample_sum = IM.summarize_sample(c_dist, n_objects)
        c_ratio_df = DataFrame(:med => sample_sum.max[1] / sample_sum.median[1], :UQ => sample_sum.max[1] / sample_sum.UQ[1], 
            :scale => s, :distribution => "$d")

        lock(c_lock)
        push!(dens_dfs, DataFrame(:x => c_x, :d => pdf.(c_dist, c_x), :scale => s, :distribution => "$d"))
        push!(quant_dfs, c_quant_df)
        push!(ratio_dfs, c_ratio_df)
        unlock(c_lock)
    end
end

dens_df = vcat(dens_dfs...);
quant_df = vcat(quant_dfs...);
ratio_df = vcat(ratio_dfs...);
```

```julia
plt = @vlplot(config={axisY={minExtent=30, maxExtent=30}}) + hcat(
#     imp_df |>
#     @vlplot(
#         x={:n_applicants, title="#Applicants", type="quantitative"},
#         y={:median, title="Impact", type="quantitative"},
#         transform = [{filter = {selection = :ScaleDist}}],
#         width=500, height=300
#     ) +
#     @vlplot(
#         mark={:line, point=true},
#         selection = {
#             ScaleDist = {
#                 type = "single", fields = ["scale", "distribution"],
#                 init = {scale = c_scales[1], distribution="Normal"},
#                 bind = {scale = {input = "range", min=minimum(c_scales), max=maximum(c_scales), step=diff(c_scales)[1], name="Scale"}, 
#                         distribution = {input = "select", options=["Normal", "LogNormal"], name="Distribution"}}
#             }
#         }
#     ) +
#     @vlplot(
#         mark={:errorbar, ticks=true},
#         y={:UQ, type="quantitative", title=""},
#         y2={:LQ}
#     ),
    dens_df |> @vlplot(
        width=300, height=300,
        transform = [{filter = {selection = :ScaleDist}}],
        x={:x, title="Impact", type="quantitative"},
        y={:d, title="Density", type="quantitative"}
    ) +
    @vlplot(
        mark=:line,
        selection = {
            ScaleDist = {
                type = "single", fields = ["scale", "distribution"],
                init = {scale = c_scales[1], distribution="Normal"},
                bind = {scale = {input = "range", min=minimum(c_scales), max=maximum(c_scales), step=diff(c_scales)[1], name="Scale"}, 
                        distribution = {input = "select", options=["Normal", "LogNormal"], name="Distribution"}}
            }
        }
    )
)
```

TODO: multiple scales per plot, join only ratio and quant plots (without dist)

```julia
# quant_df |> @vlplot(
#         width=300, height=300,
#         transform = [{filter = {selection = :ScaleDist}}],
#         x={:scale, title="Impact", type="quantitative"},
#         y={:values, title="Density", type="quantitative"},
#     ) +
#     @vlplot(
#         mark=:point,
#         selection = {
#             ScaleDist = {
#                 type = "single", fields = ["scale", "distribution"],
#                 init = {scale = c_scales[1], distribution="Normal"},
#                 bind = {scale = {input = "range", min=minimum(c_scales), max=maximum(c_scales), step=diff(c_scales)[1], name="Scale"}, 
#                         distribution = {input = "select", options=["Normal", "LogNormal"], name="Distribution"}}
#             },
#             legend = {type = "multi", fields = [:rid], bind = "legend"},
#         }
#     )
```

```julia
GF.plot(p_df, x=:scale, y=:values, color=:type, GF.Geom.PointGeometry, GF.Scale.y_log10)
```

```julia
GF.plot(x=rand(LogNormal(1, 1), 10000), GF.Geom.histogram)
```

```julia
GF.plot(x=rand(LogNormal(1, 2), 10000), GF.Geom.histogram)
```

```julia
IM.rotate_df(IM.summarize_sample(LogNormal(1, 2), n_objects), scale=2, d=9)
```

```julia
p_df = vcat([IM.rotate_df(IM.summarize_sample(LogNormal(1, s), n_objects), scale=s) for s in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]]...);
GF.plot(p_df, x=:scale, y=:values, color=:type, GF.Geom.PointGeometry, GF.Scale.y_log10)
```

```julia
p_df = vcat([IM.rotate_df(IM.summarize_sample(LogNormal(10, s), n_objects), scale=s) for s in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]]...);
GF.plot(p_df, x=:scale, y=:values, color=:type, GF.Geom.PointGeometry, GF.Scale.y_log10)
```

```julia
scales = range(0.01, 3.0; length=10)
samples = IM.summarize_sample.(LogNormal.(1, scales), n_objects)
p_df = DataFrame(:med => [d.max[1] / d.median[1] for d in samples], :UQ => [d.max[1] / d.UQ[1] for d in samples], :scale => scales)

GF.plot(p_df, GF.layer(x=:scale, y=:med, GF.Geom.PointGeometry, color=[:median]), 
    GF.layer(x=:scale, y=:UQ, GF.Geom.PointGeometry, color=[:UQ]), GF.Scale.y_log10, GF.Guide.ylabel("Ratio of the max value to median or UQ"))
```

```julia
samples = IM.summarize_sample.(LogNormal.(-10, scales), n_objects)
p_df = DataFrame(:med => [d.max[1] / d.median[1] for d in samples], :UQ => [d.max[1] / d.UQ[1] for d in samples], :scale => scales)

GF.plot(p_df, GF.layer(x=:scale, y=:med, GF.Geom.PointGeometry, color=[:median]), 
    GF.layer(x=:scale, y=:UQ, GF.Geom.PointGeometry, color=[:UQ]), GF.Scale.y_log10, GF.Guide.ylabel("Ratio of the max value to median / UQ"))
```

### Normal

```julia
p_df = vcat([IM.rotate_df(IM.summarize_sample(Normal(10, s), n_objects), scale=s) for s in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]]...);
GF.plot(p_df, x=:scale, y=:values, color=:type, GF.Geom.PointGeometry)
```

```julia
scales = range(0.01, 3.0; length=10)
samples = IM.summarize_sample.(Normal.(1, scales), n_objects)
p_df = DataFrame(:med => [d.max[1] / d.median[1] for d in samples], :UQ => [d.max[1] / d.UQ[1] for d in samples], :scale => scales)

GF.plot(p_df, GF.layer(x=:scale, y=:med, GF.Geom.PointGeometry, color=[:median]), 
    GF.layer(x=:scale, y=:UQ, GF.Geom.PointGeometry, color=[:UQ]), GF.Guide.ylabel("Ratio of the max value to median / UQ"))
```

### Exponential

```julia
p_df = vcat([IM.rotate_df(IM.summarize_sample(Exponential(s), n_objects), scale=s) for s in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]]...);
GF.plot(p_df, x=:scale, y=:values, color=:type, GF.Geom.PointGeometry)
```

```julia
scales = range(0.01, 3.0; length=10)
samples = IM.summarize_sample.(Exponential.(scales), n_objects)
p_df = DataFrame(:med => [d.max[1] / d.median[1] for d in samples], :UQ => [d.max[1] / d.UQ[1] for d in samples], :scale => scales)

GF.plot(p_df, GF.layer(x=:scale, y=:med, GF.Geom.PointGeometry, color=[:median]), 
    GF.layer(x=:scale, y=:UQ, GF.Geom.PointGeometry, color=[:UQ]), GF.Guide.ylabel("Ratio of the max value to median / UQ"))
```
