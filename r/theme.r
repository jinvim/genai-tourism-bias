# this is customized version of kjhealy/myriad
library(ggplot2)
library(showtext)

myriad_font_dir <- "~/Library/Fonts"

font_add(
    "Myriad Pro SemiCondensed",
    regular = paste0(myriad_font_dir, "/", "MyriadPro-SemiCn.otf"),
    bold = paste0(myriad_font_dir, "/", "MyriadPro-SemiboldSemiCn.otf"),
    italic = paste0(myriad_font_dir, "/", "MyriadPro-SemiCnIt.otf"),
    bolditalic = paste0(myriad_font_dir, "/", "MyriadPro-SemiboldCondIt.otf")
)

showtext_opts(dpi = 300)
showtext_auto()

theme_myriad <- function(
  base_size = 12,
  base_family = "Myriad Pro SemiCondensed",
  title_family = "Myriad Pro SemiCondensed",
  base_line_size = base_size / 24,
  base_rect_size = base_size / 24
) {
  half_line <- base_size / 2
  quarter_line <- base_size / 4

  t <- ggplot2::theme_minimal(
    base_family = base_family,
    base_size = base_size
  )

  t <- t +
    theme(
      line = element_line(
        colour = "black",
        linewidth = base_line_size,
        linetype = 1,
        lineend = "butt"
      ),
      rect = element_rect(
        fill = "white",
        colour = "black",
        linewidth = base_rect_size,
        linetype = 1
      ),
      text = element_text(
        family = base_family,
        face = "plain",
        colour = "black",
        size = base_size,
        lineheight = 0.9,
        hjust = 0.5,
        vjust = 0.5,
        angle = 0
      )
    )
  # axis options
  t <- t +
    theme(
      # axis.line = element_line(color = "gray10", linewidth = 0.5),
      axis.text = element_text(color = "black", size = base_size * 1.1),
      axis.text.x = element_text(
        margin = margin(
          t = 0.8 *
            half_line /
            2
        ),
        vjust = 1
      ),
      axis.text.x.top = element_text(
        margin = margin(
          b = 0.8 *
            half_line /
            2
        ),
        vjust = 0
      ),
      axis.text.y = element_text(
        margin = margin(
          r = 0.8 *
            half_line /
            2
        ),
        hjust = 1
      ),
      axis.text.y.right = element_text(
        margin = margin(
          l = 0.8 *
            half_line /
            2
        ),
        hjust = 0
      ),
      axis.line.x = element_line(linewidth = quarter_line / 8),
      axis.ticks = element_line(linewidth = quarter_line / 8),
      axis.ticks.y = element_blank(),
      axis.ticks.length = grid::unit(half_line, "pt"),
      axis.ticks.length.x = NULL,
      axis.ticks.length.x.top = NULL,
      axis.ticks.length.x.bottom = NULL,
      axis.ticks.length.y = NULL,
      axis.ticks.length.y.left = NULL,
      axis.ticks.length.y.right = NULL,
      axis.title = element_text(
        size = base_size * 1.2,
        # face = "italic"
      ),
      axis.title.x = element_text(
        margin = margin(t = half_line / 2),
        vjust = 1
      ),
      axis.title.x.top = element_text(
        margin = margin(b = half_line / 2),
        vjust = 0
      ),
      axis.title.y = element_text(
        angle = 90,
        margin = margin(
          r = half_line /
            2
        ),
        vjust = 1
      ),
      axis.title.y.right = element_text(
        angle = -90,
        margin = margin(
          l = half_line /
            2
        ),
        vjust = 0
      )
    )
  # legend options
  t <- t +
    theme(
      legend.background = element_blank(),
      legend.spacing = grid::unit(base_size, "pt"),
      legend.spacing.x = NULL,
      legend.spacing.y = NULL,
      legend.margin = margin(half_line, half_line, half_line, half_line),
      legend.key = element_rect(fill = "white", colour = NA),
      legend.key.size = grid::unit(1.2, "lines"),
      legend.key.height = NULL,
      legend.key.width = NULL,
      legend.text = element_text(size = base_size * 0.9),
      legend.text.align = NULL,
      legend.title = element_text(hjust = 0),
      legend.title.align = NULL,
      legend.position = "top",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.justification = "center",
      legend.box.margin = margin(0, 0, 0, 0, "cm"),
      legend.box.background = element_blank(),
      legend.box.spacing = grid::unit(base_size, "pt")
    )

  t <- t +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border = element_blank(),
      # panel.grid = element_line(colour = "gray90", linewidth = 0.1),
      # panel.grid.major = element_line(colour = "gray90", linewidth = 0.1),
      # panel.grid.minor = element_line(colour = "gray90", linewidth = 0.1),
      panel.spacing = grid::unit(half_line, "pt"),
      panel.spacing.x = NULL,
      panel.spacing.y = NULL,
      panel.ontop = FALSE
    )

  t <- t +
    theme(
      strip.background = element_blank(),
      strip.clip = "inherit",
      strip.text = element_text(
        colour = "grey10",
        size = base_size * 1.3,
        face = "italic",
        margin = margin(
          0.8 * half_line,
          0.8 * half_line,
          0.8 * half_line,
          0.8 * half_line
        )
      ),
      strip.text.x = NULL,
      strip.text.y = element_text(angle = -90),
      strip.text.y.left = element_text(angle = 90),
      strip.placement = "inside",
      strip.placement.x = NULL,
      strip.placement.y = NULL,
      strip.switch.pad.grid = grid::unit(quarter_line, "pt"),
      strip.switch.pad.wrap = grid::unit(quarter_line, "pt")
    )

  t <- t +
    theme(
      plot.background = element_rect(colour = "white"),
      plot.title = element_text(
        family = title_family,
        face = "bold",
        # size = base_size * 1.4,
        size = base_size * 1.8,
        hjust = 0,
        vjust = 1,
        margin = margin(l = half_line, b = half_line)
      ),
      plot.title.position = "plot",
      # plot.title.position = "panel",
      plot.subtitle = element_text(
        hjust = 0,
        vjust = 1,
        size = base_size * 1.5,
        margin = margin(l = half_line, b = half_line)
      ),
      plot.caption = element_text(
        size = base_size * 0.9,
        hjust = 1,
        vjust = 1,
        margin = margin(t = half_line)
      ),
      plot.caption.position = "panel",
      plot.tag = element_text(
        size = base_size * 1.2,
        hjust = 0.5,
        vjust = 0.5
      ),
      plot.tag.position = "topleft",
      plot.margin = margin(half_line, half_line, half_line, half_line)
    )

  t
}
