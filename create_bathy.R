#' Create Hypsographic Curve from Lake Area and Depth
#'
#' This function generates a hypsographic curve (bathymetric profile) of the lake.
#' The function assumes the lake basin has a circular surface and conical shape, where the radius at the surface is computed from the total surface area. 
#'
#' @param area Numeric. The surface area of the lake in square meters.
#' @param depth Numeric. The maximum depth of the lake in meters.
#' @param interval Numeric. Depth interval in meters for calculating the area at each depth slice.
#'
#' @return A data frame with two columns:
#' \describe{
#'   \item{Depths}{Depth from the surface (0) to the maximum depth in meters.}
#'   \item{Areas}{Estimated lake area (m²) at each depth based on a conical shape assumption.}
#' }
#'
#' @details
#' The function assumes the lake basin has a circular surface and conical shape, where
#' the radius at the surface is computed from the total surface area. 
#'
#' @examples
#' area <- 40000000
#' depth <- 5
#' interval <- 0.1
#' bathy_df <- create_bathy(area, depth, interval)
#' head(bathy_df)
#'
#' @export


create_bathy <- function(area, depth, interval){
  
  r_max <- sqrt(area/pi)
  
  Areas <- numeric()
  Depths <- seq(from= 0, to= depth, by= interval)
  
  
  rel_depth <- Depths / depth
  
  Areas <- pi * (r_max * (1 - rel_depth))^2
  
  return(data.frame(Depths = Depths, Areas = Areas))

  
  }




