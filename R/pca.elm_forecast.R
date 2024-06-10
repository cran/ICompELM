#' Forecasting from PCA based ELM model
#'
#' Forecasts are generated recursively from a trained Extreme Learning Machine
#' built using Principal Component Analysis.
#'
#' @param pca.elm_model A trained PCA based ELM model.
#' @param h Number of periods for forecasting. Defaults to one-step
#'   ahead forecast.
#'
#' @return Vector of point forecasts.
#'
#' @seealso [pca.elm_train()] for training an ICA based ELM model.
#'
#' @examples
#' train_set <- head(price, 12*12)
#' test_set <- tail(price, 12)
#' pca.model <- pca.elm_train(train_data = train_set, lags = 12)
#' y_hat <- pca.elm_forecast(pca.elm_model = pca.model, h = length(test_set))
#' # Evaluation of the forecasts
#' if(require("forecast")) forecast::accuracy(y_hat, test_set)
#' @export

pca.elm_forecast <- function(pca.elm_model, h = 1) {
  train_data <- pca.elm_model$data
  out_weights <- pca.elm_model$out_weights
  lags <- pca.elm_model$lags
  comps <- pca.elm_model$comps
  bias <- pca.elm_model$bias
  actfun <- pca.elm_model$actfun
  center <- pca.elm_model$center
  scale <- pca.elm_model$scale
  dat_df <- as.data.frame(tsutils::lagmatrix(c(train_data),
                                             lag = 0:lags)[-(1:lags), ])
  colnames(dat_df) <- c("y", paste0("y_(t-", 1:lags, ")"))
  for (i in 1:h) {
    dat_df[nrow(dat_df) + 1, -1] = rev(utils::tail(dat_df$y, lags))
    pca <- stats::prcomp(dat_df[, -1], center = center, scale. = scale)
    inp_weights <- pca$rotation[, 1:comps]
    hidden <- scale(dat_df[, -1], center=center, scale=scale) %*% inp_weights
    h.out <- activate(hidden, actfun = actfun)
    colnames(h.out) <- paste0("h", 1:comps)
    if (bias == TRUE) {
      dat_df[nrow(dat_df), 1] <- c(1, unlist(h.out[nrow(h.out), ])) %*%
        out_weights
    } else dat_df[nrow(dat_df), 1] <- h.out[nrow(h.out), ] %*% out_weights
  }
  forecasts <- dat_df$y[(length(train_data)-lags+1) : length(dat_df$y)]
  return(forecasts)
}
