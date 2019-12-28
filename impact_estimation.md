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
using VegaLite, VegaDatasets

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

```julia
import PlotlyJS
PLY = PlotlyJS;
```

```julia
n_applicants = [2, 5, 10, 25, 50, 100, 200, 500];
```

TODO: model fraction of the top and the second applicant lost

```julia
c_scales = range(0.1, 3.0; step=0.1);
imp_dfs = Array{DataFrame, 1}(undef, 2 * length(c_scales));
dens_dfs = Array{DataFrame, 1}(undef, 2 * length(c_scales));

@time Threads.@threads for (i, s) in collect(enumerate(c_scales))
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
@vlplot(config={axisY={minExtent=30, maxExtent=30}}) + hcat(
    imp_df |>
    @vlplot(
        x={:n_applicants, title="#Applicants"},
        y={:median, title="Impact"},
        transform = [{filter = {selection = :ScaleDist}}],
        width=700, height=300
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
        y={:UQ, typ="quantitative", title=""},
        y2={:LQ}
    ),
    dens_df |> @vlplot(
        width=300, height=300,
        transform = [{filter = {selection = :ScaleDist}}],
        x={:x, title="Impact"},
        y={:d, title="Density"},
        mark=:line
    )
)
```

## Total impact model


Here we estimate weighted sum impact of candidates. Weights are proportional to total "potential" of companies. At the beginning we assume perfect matching: the top-*k* company select the top-*k* candidate (it's property of the metric and it fits to the real world)


### Fraction of impact of top organizations


Particularly, this section can be considered as answer on the question "How many impact we'd loose if every applicant give up if not accepted to the one of top organization instead of applying to lower-rank ones".


**TODO:** it's not size-normaized, so to understand where to focus we need to use "potentials" instead of absolute impacts


Scale 3:

```julia
org_impact_scales = range(0.25, 3.0; step=0.25);
org_impact_dfs = Array{DataFrame, 1}(undef, 2 * length(org_impact_scales));

@time Threads.@threads for (i, s) in collect(enumerate(org_impact_scales))
    for (j, m, d) in zip(1:2, [mean_norm, mean_log], [Normal, LogNormal])
        c_i = (i - 1) * 2 + j
        df = vcat(IM.organization_impact_fraction.(5000, [2, 5, 10, 50, 100, 1000], d(m, s))...)
        df[!, :scale] .= s
        df[!, :distribution] .= "$d"
        org_impact_dfs[c_i] = df
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

@where(org_impact_df, :organization_id .<= 20) |> 
@vlplot(config={axisY={minExtent=30, maxExtent=30}}) + hcat(
    vln_base + @vlplot(
        y={:cum_total_median, title="Cumulative fraction of total impact", type="quantitative"},
        mark={:line, point=true},
        selection = {
            ScaleDist = {
                type = "single", fields = ["scale", "distribution"],
                init = {scale = 2.0, distribution="LogNormal"},
                bind = {scale = {input = "range", min=minimum(total_imp_scales), max=maximum(total_imp_scales), step=diff(total_imp_scales)[1], name="Scale"}, 
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
        y={:top_UQ, typ="quantitative", title=""},
        y2={:top_LQ}
    )
)
```

### Overview

```julia
total_imp_scales = range(0.25, 3.0; step=0.25);
total_imp_dfs = Array{DataFrame, 1}(undef, 4 * length(total_imp_scales));

@time Threads.@threads for (i, s) in collect(enumerate(total_imp_scales))
    for (j, ma, ad) in zip(1:2, [mean_norm, mean_log], [Normal, LogNormal])
        for (k, mo, od) in zip([0, 2], [mean_norm, mean_log], [Normal, LogNormal])
            c_i = (i - 1) * 4 + j + k
            df = vcat([IM.total_impact_per_company(3000, n_a, n_o, ad(ma, s), od(mo, s)) 
                    for n_a in n_applicants_total for n_o in n_organizations]...);
            df[!, :scale] .= s
            df[!, :app_distribution] .= "$ad"
            df[!, :org_distribution] .= "$od"
            total_imp_dfs[c_i] = df
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

```julia
@time impact_loss_df = vcat([vcat([IM.wrong_choice_impact_loss(1000, 500, 50, LogNormal(1, 1), LogNormal(1, 1); 
            real_id=ri, shift=s) for s in vcat(1:5, 7, 10, 15, 20)]...) for ri in vcat(1:5, 7, 10, 15, 20, 30)]...);

t_colors = IM.ints_to_colors(impact_loss_df.real_id; min_val=20)
GF.plot(impact_loss_df, x=:shift, y=:personal_median, ymax=:personal_UQ, ymin=:personal_LQ, color=:real_id,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...),
    GF.Guide.ylabel("Fraction of personal impact"), GF.Guide.title("500 applicants, 50 organizations"))
```

```julia
GF.plot(impact_loss_df, x=:shift, y=:total_median, ymax=:total_UQ, ymin=:total_LQ, color=:real_id,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...),
    GF.Guide.ylabel("Fraction of total impact"), GF.Guide.title("500 applicants, 50 organizations"))
```

```julia
@time impact_loss_df = vcat([vcat([IM.wrong_choice_impact_loss(1000, 100, 50, LogNormal(1, 1), LogNormal(1, 1); 
            real_id=ri, shift=s) for s in vcat(1:5, 7, 10, 15, 20)]...) for ri in vcat(1:5, 7, 10, 15, 20, 30)]...);

t_colors = IM.ints_to_colors(impact_loss_df.real_id; min_val=20)
GF.plot(impact_loss_df, x=:shift, y=:personal_median, ymax=:personal_UQ, ymin=:personal_LQ, color=:real_id,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...),
    GF.Guide.ylabel("Fraction of personal impact"), GF.Guide.title("100 applicants, 50 organizations"))
```

```julia
@time impact_loss_df = vcat([vcat([IM.wrong_choice_impact_loss(1000, 30, 20, LogNormal(1, 1), LogNormal(1, 1); 
            real_id=ri, shift=s) for s in 1:10]...) for ri in 1:10]...);

t_colors = IM.ints_to_colors(impact_loss_df.real_id; min_val=20)
GF.plot(impact_loss_df, x=:shift, y=:personal_median, ymax=:personal_UQ, ymin=:personal_LQ, color=:real_id,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...),
    GF.Guide.ylabel("Fraction of personal impact"), GF.Guide.title("30 applicants, 20 organizations"))
```

```julia
@time impact_loss_df = vcat([vcat([IM.wrong_choice_impact_loss(1000, 1000, 100, LogNormal(1, 1), LogNormal(1, 1); 
            real_id=ri, shift=s) for s in vcat(1:5, 7, 10, 15, 20)]...) for ri in vcat(1:5, 7, 10, 15, 20, 30)]...);

t_colors = IM.ints_to_colors(impact_loss_df.real_id; min_val=20)
GF.plot(impact_loss_df, x=:shift, y=:personal_median, ymax=:personal_UQ, ymin=:personal_LQ, color=:real_id,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...),
    GF.Guide.ylabel("Fraction of personal impact"), GF.Guide.title("1000 applicants, 100 organizations"))
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
total_impact_df_rand = vcat([IM.total_impact_per_company(5000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); matching=:weighted) 
        for n_a in n_applicants_total for n_o in n_organizations]...);

t_colors = IM.ints_to_colors(total_impact_df_rand.n_applicants; min_val=20)
GF.plot(total_impact_df_rand, x=:n_organizations, y=:median, ymax=:UQ, ymin=:LQ, color=:n_applicants,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...))
```

Impact loss in case of HR focus on top:

```julia
n_organizations = [2, 3, 5, 10, 15, 20, 30, 50];
n_applicants_total = [50, 75, 100, 150, 200, 300];

hr_impact_loss_df = vcat([IM.impact_loss_from_matching(5000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); matching2=:weighted) 
        for n_a in n_applicants_total for n_o in n_organizations]...);

t_colors = IM.ints_to_colors(total_impact_df_rand.n_applicants; min_val=20)
GF.plot(hr_impact_loss_df, x=:n_organizations, y=:median, ymax=:UQ, ymin=:LQ, color=:n_applicants,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...))
```

```julia
n_organizations = [10, 15, 20, 30, 50, 100, 150];
n_applicants_total = [150, 200, 300, 500];

hr_impact_loss_df = vcat([IM.impact_loss_from_matching(5000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); matching2=:weighted, n_exact=10) 
        for n_a in n_applicants_total for n_o in n_organizations]...);

t_colors = IM.ints_to_colors(total_impact_df_rand.n_applicants; min_val=20)
GF.plot(hr_impact_loss_df, x=:n_organizations, y=:median, ymax=:UQ, ymin=:LQ, color=:n_applicants,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...))
```

```julia
hr_impact_loss_df = vcat([IM.impact_loss_from_matching(5000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); matching2=:random, n_exact=10) 
        for n_a in n_applicants_total for n_o in n_organizations]...);

t_colors = IM.ints_to_colors(total_impact_df_rand.n_applicants; min_val=20)
GF.plot(hr_impact_loss_df, x=:n_organizations, y=:median, ymax=:UQ, ymin=:LQ, color=:n_applicants,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...))
```

```julia
hr_impact_loss_df = vcat([IM.impact_loss_from_matching(5000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); matching1=:weighted, matching2=:random, n_exact=10) 
        for n_a in n_applicants_total for n_o in n_organizations]...);

t_colors = IM.ints_to_colors(total_impact_df_rand.n_applicants; min_val=20)
GF.plot(hr_impact_loss_df, x=:n_organizations, y=:median, ymax=:UQ, ymin=:LQ, color=:n_applicants,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...))
```

```julia
hr_impact_loss_df = vcat([IM.impact_loss_from_matching(5000, n_a, n_o, LogNormal(1, 1), LogNormal(1, 1); matching1=:weighted, matching2=:random) 
        for n_a in n_applicants_total for n_o in n_organizations]...);

t_colors = IM.ints_to_colors(total_impact_df_rand.n_applicants; min_val=20)
GF.plot(hr_impact_loss_df, x=:n_organizations, y=:median, ymax=:UQ, ymin=:LQ, color=:n_applicants,
    GF.Geom.line, GF.Geom.errorbar, GF.Scale.color_discrete_manual(t_colors...))
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
