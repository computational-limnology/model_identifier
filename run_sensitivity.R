#' Run Sensitivity Analysis for a Model Parameter
#'
#' Performs a one-at-a-time sensitivity analysis for a specified model parameter,
#' optionally targeting a specific group (e.g., a phytoplankton group in a parameter CSV).
#' The parameter value is iteratively changed across a defined range, the model is run,
#' and the selected metrics are calculated at each step.
#'
#' @param param_name Character. Name of the parameter to modify (must match `pars` in `calib_setup`).
#' @param calib_setup Data frame. Calibration setup table containing parameter names, bounds (`lb`, `ub`), 
#' initial values, file names (`file`), and optionally a `group_name` column for group-specific parameters.
#' @param yaml_file Character. Path to the YAML file defining which metrics to extract (output.yaml).
#' @param model_dir Character. Path to the model directory where the input files and outputs are located.
#' @param n_steps Integer. Number of steps (iterations) in the parameter value sequence (default = 10).
#' @param model_filter Character. Model identifier used by `cal_metrics()` to filter the results (default = `"GLM"`).
#' @param group_name Character or `NULL`. Optional if certain pyhtoplankton group should be selected. If provided, only updates the specified group/column
#' in the parameter CSV (e.g., `"cyano"`, `"green"`, `"diatom"`). If `NULL`, all relevant rows in `calib_setup`
#' for `param_name` are used.
#'
#' @return A list with one element per parameter value step. Each element contains:
#' \describe{
#'   \item{param_value}{The value used in this iteration.}
#'   \item{metrics}{The output from `cal_metrics()` for this iteration.}
#' }
#'
#' @details
#' The function supports both `.nml` and `.csv` parameter files. For `.csv` files, it automatically
#' handles quoted column names and parameter names (e.g., `'p_name'`, `'R_growth'`) by stripping quotes.
#' When `group_name` is provided, it only updates that specific group column. Otherwise, all columns
#' (from column 2 onward) are updated for the matching parameter row.
#'
#' @importFrom readr read_csv write_csv
#' @importFrom utils file.remove
#' @importFrom glm3r run_glm
#' @importFrom LakeEnsemblR.WQ cal_metrics

#' @examples
#' \dontrun{
#' results <- run_sensitivity("R_growth", calib_setup, yaml_file = "metrics.yaml",
#'                            model_dir = "model/", n_steps = 10, group_name = "cyano")
#' }
#'
#' @export


run_sensitivity <- function(param_name, calib_setup, yaml_file, model_dir, n_steps = 10, 
                            model_filter = "GLM", group_name = NULL) {
  
  # Filter relevant rows
  if (!param_name %in% calib_setup$pars) {
    stop("Parameter not found in calib_setup dataframe: ", param_name)
  }
  
  param_rows <- calib_setup[calib_setup$pars == param_name, ]
  
  # If group_name is provided, filter further
  if (!is.null(group_name)) {
    param_rows <- param_rows[param_rows$group_name == group_name, ]
    if (nrow(param_rows) == 0) {
      stop("No matching entry for param '", param_name, "' with group_name '", group_name, "'")
    }
  }
  
  param_path <- file.path(model_dir, unique(param_rows$file))
  output_nc <- file.path(model_dir, "output", "output.nc")
  param_values <- seq(param_rows$lb[1], param_rows$ub[1], length.out = n_steps)
  results <- list()
  
  for (i in seq_along(param_values)) {
    cat("\nIteration", i, ": setting", param_name, "to", param_values[i], "\n")
    
    if (grepl("\\.nml$", param_rows$file[1])) {
      nml <- read_nml(param_path)
      nml <- set_nml(nml, param_name, param_values[i])
      write_nml(nml, param_path)
      
    } else if (grepl("\\.csv$", param_rows$file[1])) {
      df <- read_csv(param_path, show_col_types = FALSE)
      # Clean column names by removing quotes
      names(df) <- gsub("^['\"]|['\"]$", "", names(df))
      
      # Identify the parameter name column
      pname_col <- intersect(c("p_name", "pname"), names(df))
      if (length(pname_col) == 0) {
        stop("Could not identify parameter name column ('p_name' or 'pname') in CSV")
      }
      
      # Clean values in the parameter name column by removing quotes
      df[[pname_col]] <- gsub("^['\"]|['\"]$", "", df[[pname_col]])
      
      # Now locate the parameter
      idx <- which(df[[pname_col]] == param_name)
      if (length(idx) == 0) stop("Parameter not found in CSV file: ", param_name)
      
      for (j in seq_len(nrow(param_rows))) {
        group_col <- param_rows$group_name[j]
        idx <- which(df$p_name == param_name)
        if (length(idx) == 0) stop("Parameter not found in CSV file: ", param_name)
        
        if (!is.na(group_col) && group_col %in% names(df)) {
          df[idx, group_col] <- param_values[i]
        } else {
          df[idx, 2:ncol(df)] <- param_values[i]
        }
      }
      write_csv(df, param_path)
    } else {
      stop("Unsupported file type: ", param_rows$file[1])
    }
    
    run_glm(model_dir)
    metrics <- cal_metrics(yaml_file, model_filter = model_filter)
    results[[i]] <- list(param_value = param_values[i], metrics = metrics)
    
    if (file.exists(output_nc)) {
      file.remove(output_nc)
    }
  }
  
  return(results)
}








