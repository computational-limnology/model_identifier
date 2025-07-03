#' Run Multi-Parameter Sensitivity Analysis Using Latin Hypercube Sampling
#'
#' This function performs a multi-parameter sensitivity analysis using
#' Latin Hypercube Sampling (LHS) within a specified relative range for
#' each parameter. It modifies the input files (either `.nml` or `.csv`)
#' for a lake model, runs the model, and extracts defined metrics for each iteration.
#'
#' @param param_names A character vector of parameter names to vary in the sensitivity analysis.
#' @param calib_setup A data frame containing calibration setup information. Must include columns:
#'   \code{pars} (parameter names), \code{x0} (default values), \code{file} (file path relative to model_dir),
#'   and optionally \code{group_name} (for phytoplankton related parameters).
#' @param rel_change A numeric value specifying the relative range (e.g., 0.1 for ±10%) around the default value (\code{x0}) to sample from.
#' @param yaml_file Path to the YAML file used for metric extraction by \code{cal_metrics()}.
#' @param model_dir Path to the directory containing the lake model files and subdirectories. It should include the parameter files (for phytoplankton, )
#' @param n_steps Number of LHS iterations to run (i.e., model realizations).
#' @param model_filter A character string specifying the model filter to pass to \code{cal_metrics()} (default is "GLM").
#'
#' @return A list of length \code{n_steps}, where each element contains:
#'   \describe{
#'     \item{params}{A named list of sampled parameter values for that iteration.}
#'     \item{metrics}{A list or object returned by \code{cal_metrics()}, containing evaluation metrics.}
#'   }
#'
#' @details
#' The function reads model configuration files (which are in the model directory), applies sampled parameter values,
#' runs the lake model using \code{run_glm()}, and collects metrics using \code{cal_metrics()}.
#' It supports modifying both `.nml` and `.csv` files. For `.csv`, the parameter is matched 
#' using a column named \code{p_name}, and optionally updated using \code{group_name} if provided.
#'
#' Parameter sampling is based on Latin Hypercube Sampling (via \code{lhs::randomLHS}),
#' and the sampled values are scaled to a symmetric range around \code{x0}.
#'
#' @importFrom lhs randomLHS
#' @importFrom readr read_csv write_csv
#' @importFrom utils file.remove
#' @importFrom glm3r run_glm
#' @importFrom LakeEnsemblR.WQ cal_metrics
#' @export
#'
#' @examples
#' \dontrun{
#' results <- run_multi_param_sensitivity(
#'   param_names = c("kw", "sed_temp", "DO_sat"),
#'   calib_setup = calib_setup,
#'   rel_change = 0.1,
#'   yaml_file = "config/metrics.yaml",
#'   model_dir = "GLM",
#'   n_steps = 20
#' )
#' }


run_multi_param_sensitivity <- function(param_names, calib_setup, rel_change, yaml_file, model_dir,
                                        n_steps = 10, model_filter = "GLM") {
  results <- list()
  n_params <- length(param_names)

# Generate LHS matrix
  lhs_matrix <- randomLHS(n_steps, n_params)  # values in [0,1]
  
  # Prepare bounds for each parameter
  bounds <- lapply(param_names, function(p) {
    row <- calib_setup[calib_setup$pars == p, ]
    if (nrow(row) == 0) stop("Parameter ", p, " not found in calib_setup.")
    min_val <- row$x0 * (1 - rel_change)
    max_val <- row$x0 * (1 + rel_change)
    return(c(min_val, max_val))
  })
  names(bounds) <- param_names

  for (i in 1:n_steps) {
    cat("\nIteration", i, ":\n")
    
    # Generate random values for each parameter within 10% range
    param_values <- list()

for (j in seq_along(param_names)) {
      p <- param_names[j]
      min_val <- bounds[[p]][1]
      max_val <- bounds[[p]][2]
      
      # Scale LHS value to [min_val, max_val]
      sampled_value <- lhs_matrix[i, j] * (max_val - min_val) + min_val
      param_values[[p]] <- sampled_value
      cat(" -", p, "=", round(sampled_value, 4), "\n")
    }

    # Apply all parameter updates for this iteration
    for (p in names(param_values)) {
      calib_row <- calib_setup[calib_setup$pars == p, ]
      param_path <- file.path(model_dir, calib_row$file[1])

    if(grepl('\\.nml$', calib_row$file[1])){
        nml <- read_nml(param_path)
        nml <- set_nml(nml, p, param_values[[p]])
        write_nml(nml, param_path)
    }
    else if(grepl("\\.csv$", calib_row$file[1])) {
        df<- read_csv(param_path, show_col_types =FALSE)
        names(df)<- gsub("^['\"]|['\"]$", "", names(df))
        pname_col <- "p_name"
        if (length(pname_col) == 0) stop("No 'p_name' column found for ", p)
        df[[pname_col]] <- gsub("^['\"]|['\"]$", "", df[[pname_col]])
        idx <- which(df[[pname_col]] == p)
        if (length(idx) == 0) stop("Parameter ", p, " not found in CSV")

        group_col <- calib_row$group_name[1]

        if(!is.na(group_col) && group_col %in% names(df)) {
            df[idx, group_col] <- param_values[[p]]
         } else {
            df[idx, 2:ncol(df)] <- param_values[[p]]
       }
        write_csv(df, param_path)
      }
    }

    # Run the model and extract metrics
    run_glm(model_dir)
    metrics <- cal_metrics(yaml_file, model_filter = model_filter)
    results[[i]] <- list(params = param_values, metrics = metrics)
    
    # Clean up
    output_nc <- file.path(model_dir, "output", "output.nc")
    if (file.exists(output_nc)) file.remove(output_nc)
  }

  return(results)
}
