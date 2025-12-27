# if rda file exists, load it
# if not, create it
# this one is for robustness checks

# setup
library(tidyverse)
library(arrow)
library(glue)

source("r/helpers.r")

# load coefficients data for alternative simulations
if (file.exists("rda/betas.alt.sum.rda")) {
    load("rda/betas.alt.sum.rda")
} else {
    betas.alt.sum <- read_parquet("data/reg/betas-alt.parquet") |>
        mutate(
            param = factor(
                param,
                levels = c("ln_d", "ln_m", "ln_asdm", "ln_asod"),
                labels = c("Destination", "Month", "Destination-Month", "Origin-Destination")
            ),
            emp = factor(
                emp,
                levels = c("advan", "nhts"),
                labels = c("ADVAN", "NHTS")
            )
        ) |>
        group_by(sim, emp, param) |>
        summarize(
            median.coeff = median(coeff),
            lower.ci = quantile(coeff, 0.025),
            upper.ci = quantile(coeff, 0.975),
            .groups = "drop"
        )
    save(betas.alt.sum, file = "rda/betas.alt.sum.rda")
}

# aggregate simulation data and fit PPML models
if (file.exists("rda/agg.betas.rda")) {
    load("rda/agg.betas.rda")
} else {
    agg.betas <- bind_rows(
        fit_agg_ppml("data/reg/gemini-advan.parquet", "gemini", "advan"),
        fit_agg_ppml("data/reg/gemini-nhts.parquet", "gemini", "nhts"),
        fit_agg_ppml("data/reg/gpt-advan.parquet", "gpt", "advan"),
        fit_agg_ppml("data/reg/gpt-nhts.parquet", "gpt", "nhts")
    ) |> 
    rename(param = term) |>
    mutate(
        param = factor(
            param,
            levels = c("ln.d", "ln.m", "ln.asdm", "ln.asod"),
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
    )
    save(agg.betas, file = "rda/agg.betas.rda")
}