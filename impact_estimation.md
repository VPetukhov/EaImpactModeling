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

Relevant articles:
- [One approach to comparing global problems in terms of expected impact](https://80000hours.org/articles/problem-framework/): scale, neglectedness, solvability and personal fit
- [**How replaceable are the top candidates in large hiring rounds? Why the answer flips depending on the distribution of applicant ability**](https://80000hours.org/2019/08/how-replaceable-are-top-candidates-in-large-hiring-rounds/): modeling of best candidates. Includes parametric distributions
- [Recap: why do some organisations say their recent hires are worth so much?](https://80000hours.org/2019/05/why-do-organisations-say-recent-hires-are-worth-so-much/): survey of cost of loosing the best candidates.
- [Think twice before talking about ‘talent gaps’ – clarifying nine misconceptions](https://80000hours.org/2018/11/clarifying-talent-gaps/)
- [**Comparative advantage**](https://80000hours.org/articles/comparative-advantage/): about allocation of people with different qualities on different jobs. Includes part on "complementarity" where some people earn to give so other people can do work directly
- [Part 5: The world’s biggest problems and why they’re not what first comes to mind](https://80000hours.org/career-guide/world-problems/): classical examples of heavy-tailed distributions in impact

Other articles:
- [Have a particular strength? Already an expert in a field? Here are the socially impactful careers 80,000 Hours suggests you consider first.](https://80000hours.org/articles/advice-by-expertise/): example of very high bar
- [Why you should consider applying for grad school right now](https://80000hours.org/2017/11/consider-applying-for-a-phd-program-now/):
  - "We are concerned about untrained amateurs going directly into trying to solve very difficult and pressing global problems. They can then cause harm overall, by lowering the average quality of analysis or launching ill-considered projects due to a lack of experience or understanding. A PhD reduces the risk you’ll accidentally do this."
- [Part 3: No matter your job, here’s 3 evidence-based ways anyone can have a real impact](https://80000hours.org/career-guide/anyone-make-a-difference/): Baseline on donations

Outside:
- [After one year of applying for EA jobs: It is really, really hard to get hired by an EA organisation](https://forum.effectivealtruism.org/posts/jmbP9rwXncfa32seH/after-one-year-of-applying-for-ea-jobs-it-is-really-really): That is opinion on applying to EA top organizations + supporting comments
- [A Framework for Thinking about the EA Labor Market](https://forum.effectivealtruism.org/posts/CkYq5vRaJqPkpfQEt/a-framework-for-thinking-about-the-ea-labor-market): That is a a supply-demand view on the talent gap within EA


Parameters:
- Personal fit
- Impact of the area
- Impact within the area

Other parameters:
- Competition

```julia
mean_norm, mean_log = 20, 1;
```

## Repeat basic analysis


TODO: model fraction of the top and the second applicant lost

```julia
n_applicants = [2, 5, 10, 25, 50, 100, 200, 300, 400, 500];
c_scales = range(0.1, 3.0; step=0.1);
imp_dfs = Array{DataFrame, 1}(undef, 2 * length(c_scales));
dens_dfs = Array{DataFrame, 1}(undef, 2 * length(c_scales));

@time @threads for (i, s) in collect(enumerate(c_scales))
    for (j, m, d, lq) in zip(1:2, [mean_norm, mean_log], [Normal, LogNormal], [0.01, 0.001])
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

## Total impact model


Here we estimate weighted sum impact of candidates. Weights are proportional to total "potential" of companies. At the beginning we assume perfect matching: the top-*k* company select the top-*k* candidate (it's property of the metric and it fits to the real world)


### Fraction of impact of top organizations


Particularly, this section can be considered as answer on the question "How many impact we'd loose if every applicant give up if not accepted to the one of top organization instead of applying to lower-rank ones".


**TODO:** it's not size-normaized, so to understand where to focus we need to use "potentials" instead of absolute impacts

```julia
org_impact_scales = range(0.25, 3.0; step=0.25);
org_impact_dfs = DataFrame[];

c_lock = SpinLock();
@time @threads for s in org_impact_scales
    for (m, d) in zip([mean_norm, mean_log], [Normal, LogNormal])
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
    for (ma, ad) in zip([mean_norm, mean_log], [Normal, LogNormal])
        for (mo, od) in zip([mean_norm, mean_log], [Normal, LogNormal])
            df = vcat([IM.total_impact_per_company(3000, n_a, n_o, ad(ma, s), od(mo, s)) 
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
total_imp_df |>
    @vlplot(
        config={axisY={minExtent=60, maxExtent=60}},
        x={:n_organizations, title="#Organizations"},
        y={:median, title="Total impact"},
        color={:n_applicants, scale={scheme = "category20b"}, type="nominal"},
        transform = [{filter = {selection = :ScaleDist}}],
        width=700, height=300
    ) +
    @vlplot(
        mark={:line, point=true},
        selection = {
            ScaleDist = {
                type = "single", fields = ["scale", "org_distribution", "app_distribution"],
                init = {scale = 1.0, org_distribution="LogNormal", app_distribution="LogNormal"},
                bind = {scale = {input = "range", min=minimum(total_imp_scales), max=maximum(total_imp_scales), step=diff(total_imp_scales)[1], name="Scale"}, 
                        app_distribution = {input = "select", options=["Normal", "LogNormal"], name="Applicant distribution"},
                        org_distribution = {input = "select", options=["Normal", "LogNormal"], name="Organization distribution"}}
            }
        }
    ) +
    @vlplot(
        mark={:errorbar, ticks=true},
        y={:UQ, typ="quantitative", title=""},
        y2={:LQ}
    )
```

Impact increases as more impactful organizations come to the market


### Relative impact if one of top candidates choose not top organization and will be replaced with not so top candidate


Do this plot for fraction of total number instead of absolute

```julia
include("./impact_modeling.jl")
IM = ImpactModeling
```

```julia
# impact_loss_df_abs = deepcopy(impact_loss_df);
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
            for (ma, ad) in zip([mean_norm, mean_log], [Normal, LogNormal])
                for (mo, od) in zip([mean_norm, mean_log], [Normal, LogNormal])
#                     df = IM.wrong_choice_impact_loss(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
#                         real_ids=vcat(1:5, 7, 10, 15, 20, 30), shifts=vcat(1:5, 7, 10, 15, 20));
#                     df = IM.wrong_choice_impact_loss_frac(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
#                         real_quants=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.45], shift_quants=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.45]);
#                     df = IM.wrong_choice_impact_loss_frac(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
#                         real_quants=range(0.0, 0.5, step=0.05), shift_quants=range(0.0, 1.0, step=0.05));
                    df = IM.wrong_choice_impact_loss(1000, n_apps, n_orgs, ad(ma, s), od(mo, s); 
                        real_ids=vcat(1:2:50), shifts=vcat(1:5, 7, 10, 15, 20));
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

p_df = IM.optimize_df(
    impact_loss_df, 
    [:personal_LQ => :pl, :personal_median => :pm, :personal_UQ => :pu, 
        :total_LQ => :tl, :total_median => :tm, :total_UQ => :tu],
    [:real_id => :rid, :shift => :sh, :scale => :sc, :app_distribution => :ad, :org_distribution => :od, 
        :n_applicants => :na, :n_organizations => :no]);
```

```julia
plt = p_df |> @vlplot(
    config={axisY={minExtent=30, maxExtent=30}},
    x={:rid, title="Real applicant rank", type="quantitative", scale = {domain = {selection = "brush"}}},
    color={:sh, scale={scheme = "category20b"}, type="nominal", name="Shift"},
    transform = [{filter = {selection = :ScaleDistNs}}],
    opacity = {condition = {selection = "legend", value = 1}, value = 0.1},
    width=350, height=300,
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
        legend = {type = "multi", fields = [:rid], bind = "legend"},
        brush = {type = "interval", encodings = ["x"]}
    }
)
```

```julia
impact_loss_scales = range(1, 3.0; step=1);
impact_loss_dfs = DataFrame[]
org_dists = []
app_dists = []

n_apps, n_orgs = 50, 50
c_lock = SpinLock();
@time @threads for s in impact_loss_scales
    for (ma, ad) in zip([mean_norm, mean_log], [Normal, LogNormal])
        for (mo, od) in zip([mean_norm, mean_log], [Normal, LogNormal])
            a_dist = ad(ma, s)
            o_dist = od(mo, s)
            df = IM.wrong_choice_impact_loss(5000, n_apps, n_orgs, a_dist, o_dist; 
                real_ids=vcat(1:2:50), shifts=vcat(1:5, 7, 10, 15, 20));
            df[!, :scale] .= s
            df[!, :app_distribution] .= "$ad"
            df[!, :org_distribution] .= "$od"

            c_x = range(quantile.(a_dist, [0.05, 0.995])..., length=500);
            ad_df = DataFrame(:x => c_x, :d => pdf.(a_dist, c_x), :scale => s, :distribution => "$ad")
            c_x = range(quantile.(o_dist, [0.005, 0.995])..., length=500);
            od_df = DataFrame(:x => c_x, :d => pdf.(o_dist, c_x), :scale => s, :distribution => "$od")

            lock(c_lock)
            push!(impact_loss_dfs, t_dfs)
            push!(org_dists, od_df)
            push!(app_dists, ad_df)
            unlock(c_lock)
        end
    end
end

impact_loss_df = vcat(impact_loss_dfs...);
org_dist_df = vcat(org_dists...)
app_dist_df = vcat(app_dists...)

p_df = IM.optimize_df(
    impact_loss_df, 
    [:personal_LQ => :pl, :personal_median => :pm, :personal_UQ => :pu, 
        :total_LQ => :tl, :total_median => :tm, :total_UQ => :tu],
    [:real_id => :rid, :shift => :sh, :scale => :sc, :app_distribution => :ad, :org_distribution => :od]);
```

```julia
plt = p_df |> @vlplot(
    config={axisY={minExtent=30, maxExtent=30}},
    x={:rid, title="Real applicant rank", type="quantitative"},
    color={:sh, scale={scheme = "category20b"}, type="nominal", name="Shift"},
    transform = [{filter = {selection = :ScaleDistNs}}],
    opacity = {condition = {selection = "legend", value = 1}, value = 0.1},
    width=350, height=300,
    y={:pm, title="Fraction of personal impact", type="quantitative"},
    mark={:line, point=true},
    selection = {
        ScaleDistNs = {
            type = "single", fields = ["sc", "od", "ad"],
            init = {sc = 1.0, od="LogNormal", ad="LogNormal"},
            bind = {sc={input="range", min=minimum(impact_loss_scales), max=maximum(impact_loss_scales), step=diff(impact_loss_scales)[1], name="Scale"},
                    ad = {input = "select", options=["Normal", "LogNormal"], name="Applicant distribution"},
                    od = {input = "select", options=["Normal", "LogNormal"], name="Organization distribution"}},
            clear = "false"
        },
        legend = {type = "multi", fields = [:rid], bind = "legend"}
    }
)
```

```julia
rank_impact = median(mapslices(x -> sort(x, rev=true), rand(LogNormal(1, 1), 1000, 50), dims=2), dims=1)
```

```julia
vln_base = @vlplot(
    x={:sh, title="Shift", type="quantitative"},
    color={:rid, scale={scheme = "category20b"}, type="nominal"},
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
```

```julia
include("./impact_modeling.jl")
IM = ImpactModeling
```

```julia
IM.savehtml2("test_vl3.html", plt);
```

```julia
IM.savehtml("test_vl2.html", plt);
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
    for (ma, ad) in zip([mean_norm, mean_log], [Normal, LogNormal])
        for (mo, od) in zip([mean_norm, mean_log], [Normal, LogNormal])
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
    @threads for (ma, ad) in collect(zip([mean_norm, mean_log], [Normal, LogNormal]))
        t_dfs = DataFrame[]
        for (mo, od) in zip([mean_norm, mean_log], [Normal, LogNormal])
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

## Distribution properties

```julia
n_objects = 100;
```

### Log-normal

```julia
GF.plot(x=rand(LogNormal(1, 1), 10000), GF.Geom.histogram)
```

```julia
GF.plot(x=rand(LogNormal(1, 2), 10000), GF.Geom.histogram)
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

## Considerations


- Companies aren't sampled randomly: there is no chance we randomly open something better than FHI
  - Should I create complex distribution, which first samples some top companies, and then the rest are from lower-impact distribution?
