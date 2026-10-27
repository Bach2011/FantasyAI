# Club Elo Ratings
library(readr); library(purrr)
dates <- seq(as.Date("2016-08-01"), as.Date("2026-10-26"), by = "week")
elo <- map_dfr(dates, ~ read_csv(paste0("http://api.clubelo.com/", .x),
                                 show_col_types = FALSE) |> dplyr::mutate(snap = .x))

