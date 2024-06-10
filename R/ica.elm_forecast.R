#' Forecasting from ICA based ELM model
#'
#' Forecasts are generated recursively from a trained Extreme Learning Machine
#' built using Independent Component Analysis.
#'
#' @param ica.elm_model A trained ICA based ELM model.
#' @param h Number of periods for forecasting. Defaults to one-step
#'   ahead forecast.
#'
#' @return Vector of point forecasts.
#'
#' @seealso [ica.elm_train()] for training an ICA based ELM model.
#'
#' @examples
#' train_set <- head(price, 12*12)
#' test_set <- tail(price, 12)
#' ica.model <- ica.elm_train(train_data = train_set, lags = 12)
#' y_hat <- ica.elm_forecast(ica.elm_model = ica.model, h = length(test_set))
#' # Evaluation of the forecasts
#' if(require("forecast")) forecast::accuracy(y_hat, test_set)
#' @export

ica.elm_forecast <- function(ica.elm_model, h = 1) {
  train_data <- ica.elm_model$data
  out_weights <- ica.elm_model$out_weights
  lags <- ica.elm_model$lags
  comps <- ica.elm_model$comps
  bias <- ica.elm_model$bias
  actfun <- ica.elm_model$actfun
  dat_df <- as.data.frame(tsutils::lagmatrix(c(train_data),
                                             lag = 0:lags)[-(1:lags), ])
  colnames(dat_df) <- c("y", paste0("y_(t-", 1:lags, ")"))
  for (i in 1:h) {
    dat_df[nrow(dat_df) + 1, -1] = rev(utils::tail(dat_df$y, lags))
    ica_fast <- ica::ica(dat_df[, -1], nc = comps)
    inp_weights <- t(ica_fast$W)
    hidden <- scale(dat_df[, -1], scale = F) %*% inp_weights
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
