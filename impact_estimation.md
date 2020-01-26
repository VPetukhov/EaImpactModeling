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
using Plots
import Gadfly
using VegaLite
using Base.Threads

GF = Gadfly

include("./impact_modeling.jl")
IM = ImpactModeling
```

```julia
MEAN_NORM = get(ENV, "MEAN_NORM", 20)::Real;
MEAN_LOGNORM = get(ENV, "MEAN_LOGNORM", 1)::Real;
AGGREGATION = Symbol(get(ENV, "AGGREGATION", :prod));
```

## Total impact model


Here we estimate weighted sum impact of candidates. Weights are proportional to total "potential" of companies. At the beginning we assume perfect matching: the top-*k* company select the top-*k* candidate (it's property of the metric and it fits to the real world)


### Overview

```julia
n_organizations = [2, 3, 5, 10, 15, 20, 30];
n_applicants_total = [30, 50, 75, 100, 150, 200, 300];
```

```julia
total_imp_scales = range(0.25, 3.0; step=0.25);
total_imp_dfs = DataFrame[];

c_lock = SpinLock();
@time @threads for s in total_imp_scales
    for (ma, ad) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
        for (mo, od) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
            df = vcat([IM.total_impact_per_company(3000, n_a, n_o, ad(ma, s), od(mo, s), aggregation=AGGREGATION) 
                    for n_a in n_applicants_total for n_o in n_organizations]...);
            df[!, :scale] .= s
            df[!, :app_distribution] .= "$ad"
            df[!, :org_distribution] .= "$od"

            lock(c_lock)
            push!(total_imp_dfs, df)
            unlock(c_lock)
        end
    end
end

total_imp_df = vcat(total_imp_dfs...);
```

```julia
p_df = IM.optimize_df(
    total_imp_df, 
    [:LQ => :LQ, :UQ => :UQ, :median => :m],
    [:n_organizations => :no, :n_applicants => :na, :scale => :sc, :app_distribution => :ad, :org_distribution => :od]);

p_df |>
IM.vl_plot_base(:no => "#Organizations", :m => "Total impact", :na => "#Applicants") +
@vlplot(
    mark={:line, point=true},
    selection = {
        Selectors = {
            type = "single", fields = ["sc", "od", "ad"],
            init = {sc = 1.0, od="LogNormal", ad="LogNormal", ne=0},
            bind = {sc = {input = "range", min=minimum(total_imp_scales), max=maximum(total_imp_scales), step=diff(total_imp_scales)[1], name="Scale"},
                    ad = {input = "select", options=["Normal", "LogNormal"], name="Applicant distribution"},
                    od = {input = "select", options=["Normal", "LogNormal"], name="Organization distribution"}},
            clear = "false"
        },
        legend = {type = "multi", fields = [:na], bind = "legend"}
    }
) +
IM.vl_errorbar()
```

Impact increases as more impactful organizations come to the market


### Relative impact if one of top candidates choose not top organization and will be replaced with not so top candidate


Do this plot for fraction of total number instead of absolute

```julia
include("./impact_modeling.jl")
IM = ImpactModeling
```

```julia
impact_loss_scales = range(1, 3.0; step=1);
impact_loss_dfs = DataFrame[]
impact_loss_n_orgs = 50:150:350
impact_loss_n_applicants = 50:250:1000

c_lock = SpinLock();
@time @threads for s in impact_loss_scales
    @threads for n_orgs in impact_loss_n_orgs
        t_dfs = DataFrame[]
        for n_apps in impact_loss_n_applicants
            for (ma, ad) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
                for (mo, od) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
                    df = IM.wrong_choice_impact_loss(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
                        real_ids=vcat(1:5, 7, 10, 15, 20, 30, 50), shifts=vcat(1:5, 7, 10, 15, 20), aggregation=AGGREGATION);
#                     df = IM.wrong_choice_impact_loss_frac(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
#                         real_quants=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.45], shift_quants=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.45], 
#                         aggregation=AGGREGATION);
#                     df = IM.wrong_choice_impact_loss_frac(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
#                         real_quants=range(0.0, 0.5, step=0.05), shift_quants=range(0.0, 1.0, step=0.05), aggregation=AGGREGATION);
#                     df = IM.wrong_choice_impact_loss(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
#                         real_ids=vcat(1:2:50), shifts=vcat(1:5, 7, 10, 15, 20), aggregation=AGGREGATION);
                    df[!, :scale] .= s
                    df[!, :app_distribution] .= "$ad"
                    df[!, :org_distribution] .= "$od"
                    df[!, :n_applicants] .= n_apps
                    df[!, :n_organizations] .= n_orgs

                    push!(t_dfs, df)
                end
            end
        end
        lock(c_lock)
        append!(impact_loss_dfs, t_dfs)
        unlock(c_lock)
    end
end

impact_loss_df = vcat(impact_loss_dfs...);
```

```julia
p_df = IM.optimize_df(
    impact_loss_df, 
    [:personal_LQ => :pl, :personal_median => :pm, :personal_UQ => :pu, 
        :total_LQ => :tl, :total_median => :tm, :total_UQ => :tu],
    [:real_id => :rid, :shift => :sh, :scale => :sc, :app_distribution => :ad, :org_distribution => :od, 
        :n_applicants => :na, :n_organizations => :no]);

vln_base = @vlplot(
    x={:sh, title="Shift", type="quantitative"},
    color={:rid, scale={scheme = "category20b"}, type="nominal", title="Real id"},
    transform = [{filter = {selection = :ScaleDistNs}}],
    opacity = {condition = {selection = "legend", value = 1}, value = 0.1},
    width=350, height=300
)

plt = p_df |> @vlplot(config={axisY={minExtent=30, maxExtent=30}}) + hcat(
    vln_base + @vlplot(
        y={:pm, title="Fraction of personal impact", type="quantitative"},
        mark={:line, point=true},
        selection = {
            ScaleDistNs = {
                type = "single", fields = ["sc", "od", "ad", "na", "no"],
                init = {sc = 1.0, od="LogNormal", ad="LogNormal", 
                        na=impact_loss_n_applicants[1], no=impact_loss_n_orgs[1]},
                bind = {sc={input="range", min=minimum(impact_loss_scales), max=maximum(impact_loss_scales), step=diff(impact_loss_scales)[1], name="Scale"},
                        na={input="range", min=minimum(impact_loss_n_applicants), max=maximum(impact_loss_n_applicants), step=diff(impact_loss_n_applicants)[1], name="#Applicants"},
                        no={input="range", min=minimum(impact_loss_n_orgs), max=maximum(impact_loss_n_orgs), step=diff(impact_loss_n_orgs)[1], name="#Organizations"},
                        ad = {input = "select", options=["Normal", "LogNormal"], name="Applicant distribution"},
                        od = {input = "select", options=["Normal", "LogNormal"], name="Organization distribution"}},
                clear = "false"
            },
            legend = {type = "multi", fields = [:rid], bind = "legend"}
        }
    ) + @vlplot(
        mark={:errorbar, ticks=true},
        y={:pu, typ="quantitative", title="", type="quantitative"},
        y2={:pl, type="quantitative"}
    ),
    vln_base + @vlplot(
        y={:tm, title="Fraction of total impact", type="quantitative"}, 
        mark={:line, point=true}
    ) + @vlplot(
        mark={:errorbar, ticks=true},
        y={:tu, typ="quantitative", title=""},
        y2={:tl, type="quantitative"}
    )
);
                                    
IM.savehtml("plots/impact_loss.html", plt);
```

```julia
p_df = IM.optimize_df(
    impact_loss_df, 
    [:personal_median => :pm],
    [:real_id => :rid, :shift => :sh, :scale => :sc, :app_distribution => :ad, :org_distribution => :od, 
        :n_applicants => :na, :n_organizations => :no]);

plt = p_df |> @vlplot(
    config={axisY={minExtent=30, maxExtent=30}},
    x={:rid, title="Real applicant rank", type="quantitative", scale = {domain = {selection = "brush"}}},
    y={:pm, title="Fraction of personal impact", type="quantitative"},
    color={:sh, title="Shift", scale={scheme = "category20b"}, type="nominal"},
    transform = [{filter = {selection = :ScaleDistNs}}],
    opacity = {condition = {selection = "legend", value = 1}, value = 0.1},
    width=350, height=300,
    mark={:line, point=true},
    selection = {
        ScaleDistNs = {
            type = "single", fields = ["sc", "od", "ad", "na", "no"],
            init = {sc = 1.0, od="LogNormal", ad="LogNormal", 
                    na=impact_loss_n_applicants[1], no=impact_loss_n_orgs[1]},
            bind = {sc={input="range", min=minimum(impact_loss_scales), max=maximum(impact_loss_scales), step=diff(impact_loss_scales)[1], name="Scale"},
                    na={input="range", min=minimum(impact_loss_n_applicants), max=maximum(impact_loss_n_applicants), step=diff(impact_loss_n_applicants)[1], name="#Applicants"},
                    no={input="range", min=minimum(impact_loss_n_orgs), max=maximum(impact_loss_n_orgs), step=diff(impact_loss_n_orgs)[1], name="#Organizations"},
                    ad = {input = "select", options=["Normal", "LogNormal"], name="Applicant distribution"},
                    od = {input = "select", options=["Normal", "LogNormal"], name="Organization distribution"}},
            clear = "false"
        },
        legend = {type = "multi", fields = [:rid], bind = "legend"},
        brush = {type = "interval", encodings = ["x"]}
    }
);
                        
IM.savehtml("plots/impact_loss_personal.html", plt);
```

Conclusions:
- It's alright to apply to many organizations and pick not the top one
- Looks like we really need to increase number of talented candidates. But in the community we do it with Funnel model. So the question is how to filter low-fit candidate earlier and how to develop community to reach more applicant without large dissatisfaction

Here size of "target audience" is approximately equal to the total number of position in all companies in the field


- Because of ranking, product of two samples is probably worse than log-normal. Need to try normal model for applicants


TODO: plots for HR to show that even if impact for the top organization is lost, total impact is less affected


## More complex modifications


### Imperfect matching

```julia
n_organizations = [2, 3, 5, 10, 15, 20, 30];
n_applicants_total = [30, 50, 75, 100, 150, 200, 300];
```

```julia
total_imp_scales = range(0.25, 3.0; step=0.25);
total_imp_weight_dfs = DataFrame[];

c_lock = SpinLock();
@time @threads for s in total_imp_scales
    for (ma, ad) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
        for (mo, od) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
            df = vcat([IM.total_impact_per_company(3000, n_a, n_o, ad(ma, s), od(mo, s); matching=:weighted) 
                    for n_a in n_applicants_total for n_o in n_organizations]...);
            df[!, :scale] .= s
            df[!, :app_distribution] .= "$ad"
            df[!, :org_distribution] .= "$od"

            lock(c_lock)
            push!(total_imp_weight_dfs, df)
            unlock(c_lock)
        end
    end
end

total_imp_weight_df = vcat(total_imp_weight_dfs...);
```

```julia
total_imp_weight_df |>
@vlplot(
    config={axisY={minExtent=60, maxExtent=60}},
    x={:n_organizations, title="#Organizations"},
    y={:median, title="Total impact"},
    color={:n_applicants, scale={scheme = "category20b"}, type="nominal", title="#Applicants"},
    opacity = {condition = {selection = "legend", value = 1}, value = 0.1},
    transform = [{filter = {selection = :ScaleDist}}],
    width=500, height=300
) +
@vlplot(
    mark={:line, point=true},
    selection = {
        ScaleDist = {
            type = "single", fields = ["scale", "org_distribution", "app_distribution"],
            init = {scale = 1.0, org_distribution="LogNormal", app_distribution="LogNormal"},
            bind = {scale = {input = "range", min=minimum(total_imp_scales), max=maximum(total_imp_scales), step=diff(total_imp_scales)[1], name="Scale"}, 
                    app_distribution = {input = "select", options=["Normal", "LogNormal"], name="Applicant distribution"},
                    org_distribution = {input = "select", options=["Normal", "LogNormal"], name="Organization distribution"}},
            clear = "false"
        },
        legend = {type = "multi", fields = [:n_applicants], bind = "legend"}
    }
) +
@vlplot(
    mark={:errorbar, ticks=true},
    y={:UQ, typ="quantitative", title=""},
    y2={:LQ}
)
```

Impact loss in case of HR focus on top:

```julia
n_organizations = [2, 3, 5, 10, 15, 20, 30, 50, 75];
n_applicants_total = [50, 75, 100, 150, 200, 300, 500];

hr_imp_scales = range(0.5, 3.0; step=0.5);
n_exact_vals = range(0, 30, step=5)
hr_imp_loss_dfs = DataFrame[];

c_lock = SpinLock();
@time @threads for s in hr_imp_scales
    @threads for (ma, ad) in collect(zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal]))
        t_dfs = DataFrame[]
        for (mo, od) in zip([MEAN_NORM, MEAN_LOGNORM], [Normal, LogNormal])
            for n_exact in n_exact_vals
                for matching in [:weighted, :random]
                    df = vcat([IM.impact_loss_from_matching(1000, n_a, n_o, ad(ma, s), od(mo, s); matching2=matching, n_exact=n_exact) 
                            for n_a in n_applicants_total for n_o in n_organizations]...);
                    df[!, :scale] .= s
                    df[!, :app_distribution] .= "$ad"
                    df[!, :org_distribution] .= "$od"
                    df[!, :n_exact] .= n_exact
                    df[!, :matching] .= "$matching"

                    push!(t_dfs, df)
                end
            end
        end
        lock(c_lock)
        append!(hr_imp_loss_dfs, t_dfs)
        unlock(c_lock)
    end
end

hr_imp_loss_df = vcat(hr_imp_loss_dfs...);
```

TODO: add ylims [0, 1], add plot titles, use plot repeating

```julia
p_df = IM.optimize_df(
    hr_imp_loss_df, 
    [:LQ => :LQ, :UQ => :UQ, :median => :m],
    [:n_organizations => :no, :n_applicants => :na, :n_exact => :ne, :scale => :sc, :app_distribution => :ad, :org_distribution => :od, :matching => :match]);
```

```julia
include("./impact_modeling.jl")
IM = ImpactModeling
```

```julia
vl_plot_base = IM.vl_plot_base(:no => "#Organizations", :m => "Fraction of the impact lost", :na => "#Applicants") +
@vlplot(
    mark={:line, point=true},
    selection = {
        Selectors = {
            type = "single", fields = ["sc", "ne", "od", "ad"],
            init = {sc = 1.0, od="LogNormal", ad="LogNormal", ne=0},
            bind = {sc = {input = "range", min=minimum(hr_imp_scales), max=maximum(hr_imp_scales), step=diff(hr_imp_scales)[1], name="Scale"},
                    ne = {input = "range", min=minimum(n_exact_vals), max=maximum(n_exact_vals), step=diff(n_exact_vals)[1], name="#Perfectly matched"},
                    ad = {input = "select", options=["Normal", "LogNormal"], name="Applicant distribution"},
                    od = {input = "select", options=["Normal", "LogNormal"], name="Organization distribution"}},
            clear = "false"
        },
        legend = {type = "multi", fields = [:na], bind = "legend"}
    }
) +
IM.vl_errorbar();
                        
# @where(p_df, :match .== "random") |> vl_plot_base
```

```julia
@vlplot(config={axisY={minExtent=40, maxExtent=40}}) + 
hcat(vl_plot_base(@where(p_df, :match .== "weighted")), vl_plot_base(@where(p_df, :match .== "random")))
```

### Personal fit

```julia
pers_fit_dist = TruncatedNormal(1, 0.2, 0, 1000)
@time total_impact_df_pers_fit = vcat([IM.total_impact_per_company(3000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); 
            personal_fit_dist=pers_fit_dist) 
        for n_a in n_applicants_total for n_o in n_organizations]...);

t_colors = IM.ints_to_colors(total_impact_df_pers_fit.n_applicants; min_val=20)
GF.plot(total_impact_df_pers_fit, x=:n_organizations, y=:median, ymax=:UQ, ymin=:LQ, color=:n_applicants,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...))
```

### Number of applicants vs Number of organizations


How much applicants we need to attract to match increase of impact of one additional organization?

**WARNING:** this section is mostly misleading, as it assumes that new organization can utilize only one applicant, while most probably with large pool of applicants it could take much more. So this part should be improved before making any conclusions.

```julia
n_organizations = [2, 3, 5, 10, 15, 20, 30];
n_applicants_total = [30, 50, 75, 100, 150, 200, 300];

@time n_additional_applicants_per_org = vcat([
    IM.bin_search_n_apps_per_new_org(2000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); quartile_frac=0.01, max_iters=200)
        for n_a in n_applicants_total for n_o in n_organizations]...);

GF.plot(n_additional_applicants_per_org, x=:n_applicants, y=:frac_additional_applicants, color=:n_organizations,
    GF.Geom.line, GF.Guide.ylabel("Fraction of applicants per organization"))
```
