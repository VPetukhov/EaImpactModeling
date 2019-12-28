module ImpactModeling

import Colors

using DataFrames
using Distributions
using ProgressMeter
using Random: shuffle
using StatsBase

function summarize_vec(vec::Vector{T} where T <: Real)
    res = DataFrame(quantile(vec, [0.0, 0.25, 0.5, 0.75, 1.0])', [:min, :LQ, :median, :UQ, :max])
    res[!, :mean] .= mean(vec)
    return res
end

function summarize_sample(distr::UnivariateDistribution, n_objects::Int, n_samples::Int=10000)
    sample = vcat(mapslices(summarize_vec, rand(distr, (n_samples, n_objects)), dims=2)...)
    res = aggregate(sample, mean)
    names!(res, names(sample))
    return res
end

function best_additional_value(n_samples::Int, n_applicants::Int, distr::UnivariateDistribution)::Vector{Float64}
    samples = mapslices(sort, rand(distr, (n_samples, n_applicants)), dims=2)
    return samples[:, end] - samples[:, end - 1]
end

function estimate_applicant_impact_summary(n_samples::Int, n_applicants::Vector{Int}, distr::UnivariateDistribution)
    best_values = best_additional_value.(n_samples, n_applicants, distr);
    applicant_impact_df = vcat(summarize_vec.(best_values)...)
    applicant_impact_df[!, :n_applicants] = n_applicants;
    return applicant_impact_df
end

function organization_impact_fraction(n_samples::Int, n_organizations::Int, organization_dist::UnivariateDistribution)
    samples = mapslices(x -> sort(x, rev=true), rand(organization_dist, (n_samples, n_organizations)), dims=2)
    frac_of_top = samples ./ samples[:, 1]
    frac_of_total = samples ./ sum(samples, dims=2)
    cum_frac_of_total = mapslices(cumsum, frac_of_total, dims=2)

    res = vcat(mapslices(summarize_vec, frac_of_top, dims=1)...)
    names!(res, [Symbol("top_$s") for s in names(res)])
    
    res_total = vcat(mapslices(summarize_vec, frac_of_total, dims=1)...)
    names!(res_total, [Symbol("total_$s") for s in names(res_total)])
    
    res_cum = vcat(mapslices(summarize_vec, cum_frac_of_total, dims=1)...)
    names!(res_cum, [Symbol("cum_total_$s") for s in names(res_cum)])
    
    res = hcat(res, res_total, res_cum)
    res[!, :n_organizations] .= n_organizations
    res[!, :organization_id] .= 1:size(res, 1)

    return res
end

# total_impact_per_company(n_samples::Int, n_applicants::Vector{Int}, n_organizations::Vector{Int}, 
#     applicant_dist::UnivariateDistribution, organization_dist::UnivariateDistribution) =
#         vcat(total_impact_per_company.(n_samples, n_applicants, n_organizations, applicant_dist, organization_dist)...)

function total_impact_per_company(personal_impacts::Matrix{Float64}, company_impacts::Matrix{Float64},
        personal_fit_dist::UnivariateDistribution; matching::Symbol=:exact)
    n_applicants = size(personal_impacts, 2)
    total_impacts = zeros(size(personal_impacts, 1))
    company_impacts = mapslices(x -> sort(x, rev=true), company_impacts, dims=2)
    for si in 1:size(personal_impacts, 1)
        personal_fits = rand(personal_fit_dist, (n_applicants, size(company_impacts, 2)));
        pers_impacts_adj = personal_fits .* personal_impacts[si,:]
        for ci in 1:size(company_impacts, 2)
            id = 0
            if matching == :exact
                ai = findmax(pers_impacts_adj[:, 1])[2]
            elseif matching == :weighted
                ai = sample(1:n_applicants, FrequencyWeights(pers_impacts_adj[:, 1]), 1)
            elseif matching == :random
                ai = sample(1:n_applicants, 1)
            else
                error("Unknown matching type: $matching")
            end

            total_impacts[si] += pers_impacts_adj[ai, 1] * company_impacts[si, ci]
            pers_impacts_adj = pers_impacts_adj[:, 2:end]
            pers_impacts_adj[ai, :] .= 0
        end
    end

    return total_impacts
end

function arrange_personal_impacts(personal_impacts::Matrix{Float64}; matching::Symbol=:exact, n_exact::Int=0)
    if n_exact > 0 && matching != :exact
        personal_impacts = mapslices(x -> sort(x, rev=true), personal_impacts, dims=2)
    end

    if matching == :exact
        personal_impacts = mapslices(x -> sort(x, rev=true), personal_impacts, dims=2)
    elseif matching == :weighted
        personal_impacts[:, (n_exact + 1):end] = mapslices(row -> sample(row, FrequencyWeights(row), length(row); replace=false), 
            personal_impacts[:, (n_exact + 1):end], dims=2)
    elseif matching == :random
        personal_impacts[:, (n_exact + 1):end] = mapslices(shuffle, personal_impacts[:, (n_exact + 1):end], dims=2)
    else
        error("Unknown matching type: $matching")
    end
    
    return personal_impacts
end

function total_impact_per_company(personal_impacts::Matrix{Float64}, company_impacts::Matrix{Float64};
        matching::Symbol=:exact, n_exact::Int=0)
    n_min = min(size(personal_impacts, 2), size(company_impacts, 2))
    company_impacts = mapslices(x -> sort(x, rev=true), company_impacts, dims=2)[:, 1:n_min]
    personal_impacts = arrange_personal_impacts(personal_impacts, matching=matching, n_exact=n_exact)[:, 1:n_min]
    return vec(sum(personal_impacts .* company_impacts, dims=2))
end

function total_impact_per_company(n_samples::Int, n_applicants::Int, n_organizations::Int, 
        applicant_dist::UnivariateDistribution, organization_dist::UnivariateDistribution;
        personal_fit_dist::Union{UnivariateDistribution, Nothing}=nothing, matching::Symbol=:exact)
    personal_impacts = rand(applicant_dist, (n_samples, n_applicants))
    company_impacts = rand(organization_dist, (n_samples, n_organizations))

    total_impact = personal_fit_dist === nothing ? 
        total_impact_per_company(personal_impacts, company_impacts; matching=matching) :
        total_impact_per_company(personal_impacts, company_impacts, personal_fit_dist; matching=matching)

    res = summarize_vec(total_impact)
    res[!, :n_applicants] .= n_applicants
    res[!, :n_organizations] .= n_organizations
    return res
end

function impact_loss_from_matching(n_samples::Int, n_applicants::Int, n_organizations::Int, 
        applicant_dist::UnivariateDistribution, organization_dist::UnivariateDistribution;
        matching1::Symbol=:exact, matching2::Symbol=:weighted, n_exact::Int=0)

    personal_impacts = rand(applicant_dist, (n_samples, n_applicants))
    company_impacts = rand(organization_dist, (n_samples, n_organizations))

    total_impact = total_impact_per_company(personal_impacts, company_impacts; matching=matching1, n_exact=n_exact)
    total_impact2 = total_impact_per_company(personal_impacts, company_impacts; matching=matching2, n_exact=n_exact)

    res = summarize_vec((total_impact .- total_impact2) ./ total_impact)
    res[!, :n_applicants] .= n_applicants
    res[!, :n_organizations] .= n_organizations
    
    return res
end

function wrong_choice_impact_loss(n_samples::Int, n_applicants::Int, n_organizations::Int, 
        applicant_dist::UnivariateDistribution, organization_dist::UnivariateDistribution;
        real_id::Int, shift::Int)
    personal_impacts = rand(applicant_dist, (n_samples, n_applicants));
    company_impacts = rand(organization_dist, (n_samples, n_organizations));

    applied_id = real_id + shift
    @assert applied_id <= min(n_applicants, n_organizations)

    personal_impacts = arrange_personal_impacts(personal_impacts, matching=:exact)[:, 1:applied_id]

    n_min = min(size(personal_impacts, 2), size(company_impacts, 2))
    company_impacts = mapslices(x -> sort(x, rev=true), company_impacts, dims=2)[:, 1:n_min];

    original_impacts = personal_impacts[:, real_id] .* company_impacts[:, real_id];
    original_impacts_total = sum(personal_impacts .* company_impacts, dims=2);

    personal_impacts[:, [real_id, applied_id]] .= personal_impacts[:, [applied_id, real_id]];
    final_impacts = personal_impacts[:, applied_id] .* company_impacts[:, applied_id];
    final_impacts_total = sum(personal_impacts .* company_impacts, dims=2);

    res = summarize_vec(vec((original_impacts_total .- final_impacts_total) ./ original_impacts))
    names!(res, [Symbol("personal_$s") for s in names(res)])

    res_total = summarize_vec(vec((original_impacts_total .- final_impacts_total) ./ original_impacts_total))
    names!(res_total, [Symbol("total_$s") for s in names(res_total)])

    res = hcat(res, res_total)

    res[!, :real_id] .= real_id
    res[!, :shift] .= shift

    return res
end

function bin_search_n_apps_per_new_org(n_samples::Int, n_applicants::Int, n_organizations::Int, 
        applicant_dist::UnivariateDistribution, organization_dist::UnivariateDistribution; 
        max_iters=100, quartile_frac=0.05)
    n_a_upper = n_applicants
    impact_target = total_impact_per_company(n_samples, n_applicants, n_organizations + 1, applicant_dist, organization_dist);

    converged = false
    for i in 1:max_iters
        n_a_upper *= 2
        impact_cur = total_impact_per_company(n_samples, n_a_upper, n_organizations, applicant_dist, organization_dist);
        if impact_cur.median > impact_target.median
            converged = true
            break
        end
    end

    if !converged
        error("First phase didn't converge ($n_applicants, $n_organizations)")
    end

    converged = false
    n_a_mean = n_a_lower = n_applicants + 1
    for i in 1:max_iters
        n_a_mean = round(Int, (n_a_lower + n_a_upper) / 2)
        impact_cur = total_impact_per_company(n_samples, n_a_mean, n_organizations, applicant_dist, organization_dist);
        if impact_cur.median > ((1 - quartile_frac) * impact_target.median + quartile_frac * impact_target.UQ)
            n_a_upper = n_a_mean
        elseif impact_cur.median < ((1 - quartile_frac) * impact_target.median + quartile_frac * impact_target.LQ)
            n_a_lower = n_a_mean
        else
            converged = true
            break
        end
    end

    if !converged
        @warn "Second phase didn't converge ($n_applicants, $n_organizations)"
    end

    n_a_mean -= n_applicants
    return DataFrame(:n_applicants => n_applicants, :n_organizations => n_organizations, 
        :n_additional_applicants => n_a_mean, :frac_additional_applicants => n_a_mean / n_applicants)
end

## Utils

function ints_to_colors(vals::Vector{Int}, colormap::String="Oranges"; min_val=1, rev=true)
    res = Colors.colormap(colormap)[round.(Int, range(min_val, 100; length=length(unique(vals))))]
    if rev
        return reverse(res)
    else
        return res
    end
end

rotate_df(df_row::DataFrame; other_cols...) =
    DataFrame(:values => collect(df_row[1,:]), :type => names(df_row), other_cols...)

end