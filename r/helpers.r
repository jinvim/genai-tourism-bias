# helper functions used in the paper
library(dplyr)
library(arrow)

# this is for manual indentation of table rows in gt
indent_row <- function(string, n = 2) {
    paste0(strrep("\u00A0", n), " ", string)
}

# read simulation data
read_sim <- function(file.name) {
    read_parquet(
        file.name,
        col_select = c(
            "dst",
            "org",
            "month",
            "iter",
            "flow_sim",
            "any_sim",
            "out",
            "pe",
            "distkm",
            "border",
            "self_loop"
        )
    ) |>
    rename_with(~ gsub("_", ".", .x)) |>
    rename(
        flow = flow.sim,
        any = any.sim
    ) |>
    group_by(iter) |>
    mutate(p = flow / sum(flow)) |>
    ungroup()
}

# read empirical data
read_emp <- function(file.name) {
    read_parquet(
        file.name,
        col_select = c(
            "dst",
            "org",
            "month",
            "iter",
            "flow_emp",
            "any_emp",
            "p_emp",
            "pe",
            "d",
            "m",
            "dm",
            "od",
            "distkm",
            "border",
            "self_loop"
        )
    ) |>
    # because empirical data is same across iters, just use iter 0
    filter(iter == 0) |>
    select(-iter) |>
    rename_with(~ gsub("_", ".", .x)) |>
    rename(
        flow = flow.emp,
        any = any.emp,
        p = p.emp
    )
}

# calculate demographic proportions
# (for descriptive table)
calc_demo_prop <- function(df, bycol, strat) {
    df |>
    group_by({{bycol}}) |>
    summarize(pop = sum(pop), .groups = "drop") |>
    mutate(
        p = pop / sum(pop),
        strat = strat,
        category = as.character({{bycol}})
    ) |>
    mutate(
        category = factor(
            category,
            levels = c(
                "female",
                "male",
                "18 to 24 years old",
                "25 to 34 years old",
                "35 to 44 years old",
                "45 to 54 years old",
                "55 to 64 years old",
                "65 years old or older",
                "less than $25,000",
                "$25,000 to $49,999",
                "$50,000 to $74,999",
                "$75,000 to $99,999",
                "$100,000 to $149,999",
                "$150,000 or more"
            ),
            labels = c(
                "Female",
                "Male",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65+",
                "< $25k",
                "$25k-$49k",
                "$50k-$74k",
                "$75k-$99k",
                "$100k-$149k",
                "≥ $150k"
            )
        )
    ) |>
    select(strat, category, p)
}

# calcuate proportion of cells with any flows
# (median over 1,000 iterations)
calc_any_prop <- function(df, data.name) {
    df |>
        group_by(iter) |>
        summarize(
            any.p = sum(any) / n(),
        ) |>
        ungroup() |>
        summarize(
            median.any.p = median(any.p),
            lower.ci = quantile(any.p, 0.025),
            upper.ci = quantile(any.p, 0.975)
        ) |>
        mutate(data.name = data.name) |>
        relocate(data.name, .before = median.any.p)
}
    

# for simulation data, aggregate all 1,000 iteration
# and calculate sum of flows & cells with any flows
# (used for visualizing DV; Flow_{(i,j,m)})
calc_dv_desc <- function(df, data.name) {
    df |>
        group_by(dst, org, month) |>
        summarize(
            flow = sum(flow),
            pe = first(pe),
            .groups = "drop"
        ) |>
        mutate(
            any = if_else(flow > 0, 1, 0),
            data.name = data.name
        ) |>
        group_by(org) |>
        mutate(
            out = sum(flow),
            offset = pe * out
        )
}

# summary statistics for DV and IV
# (min, max, median, mean, sd)
calc_summary <- function(df, data.name, ...) {
    stat_funs <- list(
        min    = ~ min(.x, na.rm = TRUE),
        max    = ~ max(.x, na.rm = TRUE),
        median = ~ median(.x, na.rm = TRUE),
        mean   = ~ mean(.x, na.rm = TRUE),
        sd     = ~ sd(.x, na.rm = TRUE)
    )

    df |>
        summarise(
            across(
                c(...),
                stat_funs
            )
        ) |>
        pivot_longer(
            cols = everything(),
            names_to = c("variable", "stat"),
            names_sep = "_",
            values_to = "value"
        ) |>
        pivot_wider(
            names_from = stat,
            values_from = value
        ) |>
        mutate(data.name = data.name)
}

# calculate proportion of tourist flow by month
# also, calculate gini index for a given scenario
# (median over 1,000 iterations)
calc_month_prop <- function(df, data.name) {
    df |>
        group_by(iter, month) |>
        summarize(
            sum.p = sum(p),
            .groups = "drop"
        ) |>
        group_by(iter) |>
        mutate(
            gini = ineq(sum.p, type = "Gini"),
        ) |>
        group_by(month) |>
        summarize(
            median.sum.p = median(sum.p),
            lower.ci = quantile(sum.p, 0.025),
            upper.ci = quantile(sum.p, 0.975),
            median.gini = median(gini),
            .groups = "drop"
        ) |>
        mutate(
            data.name = data.name,
            month = factor(
                month,
                levels = seq(1, 12),
                labels = month.abb,
                ordered = TRUE
            )
        )
}


# calculate proportion of tourist flow by destination
# also, calculate gini index for a given scenario
# (median over 1,000 iterations)
calc_dst_prop <- function(df, data.name) {
    df |>
        group_by(iter, dst) |>
        summarize(
            sum.p = sum(p),
            .groups = "drop"
        ) |>
        group_by(iter) |>
        mutate(
            gini = ineq(sum.p, type = "Gini"),
        ) |>
        group_by(dst) |>
        summarize(
            median.sum.p = median(sum.p),
            lower.ci = quantile(sum.p, 0.025),
            upper.ci = quantile(sum.p, 0.975),
            median.gini = median(gini),
            .groups = "drop"
        ) |>
        mutate(data.name = data.name)
}

# calculate gini index for destination-bycol combination
# (median over 1,000 iterations)
calc_gini_by <- function(df, bycol, data.name) {
    df |>
        group_by(iter, dst, {{ bycol }}) |>
        summarize(
            sum.p = sum(p),
            .groups = "drop"
        ) |>
        group_by(iter, dst) |>
        summarize(
            gini = ineq(sum.p, type = "Gini", na.rm = TRUE),
            .groups = "drop"
        ) |>
        group_by(dst) |>
        summarize(
            median.gini = median(gini, na.rm = TRUE),
            lower.ci = quantile(gini, 0.025, na.rm = TRUE),
            upper.ci = quantile(gini, 0.975, na.rm = TRUE),
            .groups = "drop"
        ) |>
        mutate(data.name = data.name)
}

# calculate entropy index for destination-bycol combination
# (median over 1,000 iterations)
calc_entropy_by <- function(df, bycol, data.name) {
    df |>
        group_by(iter, dst, {{ bycol }}) |>
        summarize(
            sum.p = sum(p),
            .groups = "drop"
        ) |>
        group_by(iter, dst) |>
        # recalculate proportions within destination
        mutate(
            p = sum.p / sum(sum.p)
        ) |>
        filter(p > 0) |>
        summarize(
            entropy = -sum(p * log(p)),
            .groups = "drop"
        ) |>
        group_by(dst) |>
        summarize(
            median.entropy = median(entropy),
            lower.ci = quantile(entropy, 0.025),
            upper.ci = quantile(entropy, 0.975),
            .groups = "drop"
        ) |>
        mutate(data.name = data.name)
}

# function to calculate graph-level statistics
# calculate graph-level weighted reciprocity
# Sequartini et al. (2013)
calc_reciprocity <- function(df) {
    # aggregate flows for destination-origin pairs
    df.edges <- df |>
        group_by(iter, dst, org) |>
        summarize(
            flow = sum(flow),
            .groups = "drop"
        )
    # calculate flows in the reverse direction
    df.edges <- df.edges |>
        left_join(
            df.edges |>
                rename(flow.rev = flow, org = dst, dst = org),
            by = c("iter", "dst", "org")
        ) |>
        # exclude self-loops (not needed for reciprocity index)
        filter(dst != org) |>
        mutate(
            # calculate minimum of edges between two nodes
            min.flow = pmin(flow, flow.rev, na.rm = TRUE),
        ) |>
        group_by(iter) |>
        summarize(
            # since we double counted the reciprocated edges, divide by 2
            recip = sum(min.flow) / sum(flow) / 2,
        )
    df.edges
}

# calculate weighted median travel distance across iterations
# excluding self-loops
calc_median_dist <- function(df) {
    df |>
        filter(dst != org) |>
        group_by(iter) |>
        summarize(
            median.dist = weighted.median(distkm, w = flow),
        )
}

# calculate proportion of travel to bordering states & self-loops
calc_border_self <- function(df) {
    df |>
        group_by(iter) |>
        summarize(
            prop.border = sum(p * border),
            prop.self = sum(p * self.loop),
        )
}

# calculate all graph-level statistics
calc_graph_stats <- function(df, data.name) {
    df.recip <- calc_reciprocity(df)
    df.dist <- calc_median_dist(df)
    df.bs <- calc_border_self(df)

    df.stats <- df.recip |>
        left_join(df.dist, by = "iter") |>
        left_join(df.bs, by = "iter") |>
        mutate(data.name = data.name)
    df.stats |>
        pivot_longer(
            cols = -c(iter, data.name),
            names_to = "stat",
            values_to = "value"
        ) |>
        mutate(
            stat = factor(
                stat,
                levels = c("recip", "median.dist", "prop.border", "prop.self"),
                labels = c(
                    "Reciprocity",
                    "Median travel distance (km; excluding in-state travel)",
                    "Ratio of travel to bordering states",
                    "Ratio of in-state travel"
                ),
                ordered = TRUE
            ),
        )
}

# fit an ppml model using data aggregated over 1,000 iterations
fit_agg_ppml <- function(file.name, sim.name, emp.name) {
    read_parquet(file.name) |>
        rename_with(~ gsub("_", ".", .x)) |>
        group_by(dst, org, month) |>
        summarize(
            flow.sim = sum(flow.sim),
            pe = first(pe),
            ln.d = first(ln.d),
            ln.m = first(ln.m),
            ln.asdm = first(ln.asdm),
            ln.asod = first(ln.asod),
            .groups = "drop"
        ) |>
        group_by(org) |>
        mutate(
            out = sum(flow.sim),
            baseline = out * pe,
            ln.baseline = log(baseline + 1e-6)
        ) |>
        ungroup() %>%
        glm(
            flow.sim ~ ln.d + ln.m + ln.asdm + ln.asod + offset(ln.baseline) + 0,
            data = .,
            family = poisson(link = "log")
        ) %>%
        lmtest::coeftest(
            .,
            vcov = sandwich::vcovHC(., type = "HC3")
        ) |>
        tidy() |>
        mutate(
            sim = sim.name,
            emp = emp.name,
        )
}

# style p-values
pval_stars <- function(x) {
  dplyr::case_when(
    x < 0.001 ~ "***",
    x < 0.01 ~ "**",
    x < 0.05 ~ "*",
    TRUE ~ ""
  )
}

pval_format <- function(x) {
  dplyr::case_when(
    x < 0.001 ~ "<0.001",
    TRUE ~ sprintf("%4.3f", x)
  )
}