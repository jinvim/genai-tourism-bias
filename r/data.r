# if rda file exists, load it
# if not, create it

# setup
library(tidyverse)
library(arrow)
library(sf)
library(gt)
library(glue)

source("r/helpers.r")

# load simulation and empirical data if needed
lazy_load_data <- function() {
    if(!exists("sim.rand")) {
        sim.rand <<- read_sim("data/reg/rand-advan.parquet")
    }
    
    if(!exists("sim.advan")) {
        sim.advan <<- read_sim("data/reg/advan-advan.parquet")
    }

    if(!exists("sim.nhts")) {
        sim.nhts <<- read_sim("data/reg/nhts-nhts.parquet")
    }

    if(!exists("sim.gemini")) {
        sim.gemini <<- read_sim("data/reg/gemini-advan.parquet")
    }

    if(!exists("sim.gpt")) {
        sim.gpt <<- read_sim("data/reg/gpt-advan.parquet")
    }
    
    if(!exists("emp.advan")) {
        emp.advan <<- read_emp("data/reg/advan-advan.parquet")
    }

    if(!exists("emp.nhts")) {
        emp.nhts <<- read_emp("data/reg/nhts-nhts.parquet")
    }
}

# states simlple features for visualization
if (file.exists("rda/sf.states.rda")) {
    load("rda/sf.states.rda")
} else {
    sf.states <- st_read("data/states.geojson")|>
        rename_with(~ gsub("_", ".", .x)) |>
        mutate(state.fips = as.integer(state.fips))
    save(sf.states, file = "rda/sf.states.rda")
}

# example data for illustraing how beta_2 works
if (file.exists("rda/df.beta.demo.rda")) {
    load("rda/df.beta.demo.rda")
} else {
    df.beta.demo <- read_sim("data/reg/nhts-nhts.parquet") |>
        calc_month_prop("nhts") |>
        mutate(
            p = median.sum.p / sum(median.sum.p),
            p.pos1 = p ** (1+1),
            p.pos2 = p ** (1+2),
            p.neg1 = p ** (1-1),
            p.neg2 = p ** (1-2),
        ) |>
        mutate(
            p.pos1 = p.pos1 / sum(p.pos1),
            p.pos2 = p.pos2 / sum(p.pos2),
            p.neg1 = p.neg1 / sum(p.neg1),
            p.neg2 = p.neg2 / sum(p.neg2)
        ) |>
        select(month, p, p.pos1, p.pos2, p.neg1, p.neg2) |>
        pivot_longer(cols = -month, names_to = "case", values_to = "p")
    save(df.beta.demo, file = "rda/df.beta.demo.rda")
}

# data for descriptive analysis & visualizations
## for descriptive statistics table
if (file.exists("rda/df.sum.rda")) {
    load("rda/df.sum.rda")
} else {
    lazy_load_data()
    df.sum <- bind_rows(
        calc_summary(sim.advan, "sim.advan", flow, out),
        calc_summary(sim.nhts, "sim.nhts", flow, out),
        calc_summary(sim.gemini, "sim.gemini", flow, out),
        calc_summary(sim.gpt, "sim.gpt", flow, out),
        calc_summary(emp.advan, "emp.advan", pe, d, m, dm, od),
        calc_summary(emp.nhts, "emp.nhts", pe, d, m, dm, od),
        ) |>
        mutate(
            notation = factor(
                variable,
                levels = c("flow", "out", "pe", "d", "m", "dm", "od"),
                labels = c(
                    "$Flow_{(i,j,m)}$",
                    "$\\sum_{j,m} Flow_{(i,j,m)}$",
                    "$S_{(j,m|i)}$",
                    "$D_{j}$",
                    "$M_{m}$",
                    "$DM_{j,m}$",
                    "$OD_{i,j}$"
                ),
            ),
            variable = factor(
                variable,
                levels = c("flow", "out", "pe", "d", "m", "dm", "od"),
                labels = md(
                    c(
                        "Tourist flow",
                        "Origin total outflow",
                        "Empirical share",
                        "Destination",
                        "Month",
                        "Destination-month",
                        "Origin-destination"
                    )
                ),
                ordered = TRUE
            ),
            data.name = factor(
                data.name,
                levels = c(
                    "sim.advan",
                    "sim.nhts",
                    "sim.gemini",
                    "sim.gpt",
                    "emp.advan",
                    "emp.nhts"
                ),
                labels = c(
                    "Simulation: ADVAN Mobility Data",
                    "Simulation: National Household Travel Survey",
                    "Simulation: Gemini 2.5 Flash Lite",
                    "Simulation: GPT-4.1 Nano",
                    "Empirical: ADVAN Mobility Data",
                    "Empirical: National Household Travel Survey"
                ),
                ordered = TRUE
            )
        ) |>
        relocate(notation, .after = variable)
    save(df.sum, file = "rda/df.sum.rda")
}

## difference in precense of any tourist flow betwween simulations
if (file.exists("rda/df.any.rda")) {
    load("rda/df.any.rda")
} else {
    lazy_load_data()

    df.any <- bind_rows(
        calc_dv_desc(sim.advan, "advan"),
        calc_dv_desc(sim.nhts, "nhts"),
        calc_dv_desc(sim.gemini, "gemini"),
        calc_dv_desc(sim.gpt, "gpt")
        ) |>
        mutate(
            any = factor(
                any,
                levels = c(0, 1),
                labels = c("No flow", "Any flow"),
                ordered = TRUE
            ),
        ) |>
        pivot_wider(
            id_cols = c(dst, org, month),
            names_from = data.name,
            values_from = any
        ) |>
        ungroup()
    save(df.any, file = "rda/df.any.rda")
}

## destination popularity
if (file.exists("rda/df.dst.rda")) {
    load("rda/df.dst.rda")
} else {
    lazy_load_data()
    df.dst <- bind_rows(
        calc_dst_prop(sim.advan, "advan"),
        calc_dst_prop(sim.nhts, "nhts"),
        calc_dst_prop(sim.gemini, "gemini"),
        calc_dst_prop(sim.gpt, "gpt")
        ) |>
        mutate(
            data.name = factor(
                data.name,
                levels = c("advan", "nhts", "gemini", "gpt"),
                ordered = TRUE
            )
        ) |>
        left_join(
            sf.states |>
            rename(dst = state.fips),
            by = "dst"
        )
    save(df.dst, file = "rda/df.dst.rda")
}

## month popularity
if (file.exists("rda/df.month.rda")) {
    load("rda/df.month.rda")
} else {
    lazy_load_data()
    df.month <- bind_rows(
        calc_month_prop(sim.advan, "advan"),
        calc_month_prop(sim.nhts, "nhts"),
        calc_month_prop(sim.gemini, "gemini"),
        calc_month_prop(sim.gpt, "gpt")
        ) |>
        mutate(
            data.name = factor(
                data.name,
                levels = c("advan", "nhts", "gemini", "gpt"),
                ordered = TRUE
            )
        )
    save(df.month, file = "rda/df.month.rda")
}

## destination-month popularity
if (file.exists("rda/df.month.gini.rda")) {
    load("rda/df.month.gini.rda")
} else {
    lazy_load_data()
    df.month.gini <- bind_rows(
        calc_gini_by(sim.advan, month, "advan"),
        calc_gini_by(sim.nhts, month, "nhts"),
        calc_gini_by(sim.gemini, month, "gemini"),
        calc_gini_by(sim.gpt, month, "gpt")
        ) |>
        mutate(
            data.name = factor(
                data.name,
                levels = c("advan", "nhts", "gemini", "gpt"),
                ordered = TRUE
            )
        ) |>
        left_join(
            sf.states |>
            rename(dst = state.fips),
            by = "dst"
        )
    save(df.month.gini, file = "rda/df.month.gini.rda")
}

## origin-destination popularity
if (file.exists("rda/df.org.entropy.rda")) {
    load("rda/df.org.entropy.rda")
} else {
    lazy_load_data()
    df.org.entropy <- bind_rows(
        calc_entropy_by(sim.advan, org, "advan"),
        calc_entropy_by(sim.nhts, org, "nhts"),
        calc_entropy_by(sim.gemini, org, "gemini"),
        calc_entropy_by(sim.gpt, org, "gpt"),
        ) |>
        mutate(
            data.name = factor(
                data.name,
                levels = c("advan", "nhts", "gemini", "gpt"),
                ordered = TRUE
            )
        ) |>
        left_join(
            sf.states |>
            rename(dst = state.fips),
            by = "dst"
        )
    save(df.org.entropy, file = "rda/df.org.entropy.rda")
}

# network-level metrics
if (
    file.exists("rda/df.graph.rda") &&
    file.exists("rda/df.graph.rand.rda")
) {
    load("rda/df.graph.rda")
    load("rda/df.graph.rand.rda")
} else {
    lazy_load_data()
    df.graph <- bind_rows(
        calc_graph_stats(sim.advan, "advan"),
        calc_graph_stats(sim.nhts, "nhts"),
        calc_graph_stats(sim.gemini, "gemini"),
        calc_graph_stats(sim.gpt, "gpt"),
        ) |>
        mutate(
            data.name = factor(
                data.name,
                levels = c("advan", "nhts", "gemini", "gpt"),
                labels = c(
                    "ADVAN Mobility Data",
                    "National Household Travel Survey",
                    "Gemini 2.5 Flash Lite",
                    "GPT-4.1 Nano"
                ),
                ordered = TRUE
            )
        )
    df.graph.rand <- calc_graph_stats(sim.rand, "rand")

    save(df.graph, file = "rda/df.graph.rda")
    save(df.graph.rand, file = "rda/df.graph.rand.rda")
}

# summary of regression coefficients
if (file.exists("rda/betas.sum.rda")) {
    load("rda/betas.sum.rda")
} else {
    betas.sum <- read_parquet("data/reg/betas.parquet") |>
        mutate(
            param = factor(
                param,
                levels = c("ln_d", "ln_m", "ln_asdm", "ln_asod"),
                labels = c("Destination", "Month", "Destination-Month", "Origin-Destination")
            ),
            sim = factor(
                sim,
                levels = c("rand", "nhts", "advan", "gemini", "gpt"),
                labels = c("Random", "NHTS", "ADVAN", "Gemini 2.5 Flash Lite", "GPT-4.1 Nano")
            ),
            emp = factor(
                emp,
                levels = c("advan", "nhts"),
                labels = c("ADVAN", "NHTS")
            )
        ) |>
        filter(sim %in% c(
            "Gemini 2.5 Flash Lite",
            "GPT-4.1 Nano"
        )) |>
        group_by(sim, emp, param) |>
        summarize(
            median.coeff = median(coeff),
            lower.ci = quantile(coeff, 0.025),
            upper.ci = quantile(coeff, 0.975),
            .groups = "drop"
        )
    save(betas.sum, file = "rda/betas.sum.rda")
}