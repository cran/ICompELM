#' Training of ICA based ELM model for time series forecasting
#'
#' An Extreme Learning Machine is trained by utilizing the concept of
#'   Independent Component Analysis.
#'
#' @param train_data A univariate time series data.
#' @param lags Number of lags to be considered.
#' @param comps Number of independent components to be considered. Corresponds
#'   to number of hidden nodes. Defaults to maximum value, i.e., \code{lags}.
#' @param bias Whether to include bias term while computing output weights.
#'   Defaults to \code{TRUE}.
#' @param actfun Activation function for the hidden layer. Defaults to
#'   \code{sig}. See `Activation functions`.
#'
#' @details
#' An Extreme Learning Machine (ELM) is trained wherein the weights connecting
#'   the input layer and hidden layer are obtained using Independent Component
#'   Analysis (ICA), instead of being chosen randomly. The number of hidden
#'   nodes is determined by the number of independent components.
#'
#' @section Activation functions:
#' The activation function for the hidden layer must be one of the following.
#' \describe{
#'   \item{\code{sig}}{Sigmoid function: \eqn{(1 + e^{-x})^{-1}}}
#'   \item{\code{radbas}}{Radial basis function: \eqn{e^{-x^2}}}
#'   \item{\code{hardlim}}{Hard-limit function: \eqn{\begin{cases} 1, & if\:x
#'         \geq 0 \\ 0, & if\:x<0 \end{cases}}}
#'   \item{\code{hardlims}}{Symmetric hard-limit function: \eqn{\begin{cases}1,
#'               & if\:x \geq 0 \\ -1, & if\:x<0 \end{cases}}}
#'   \item{\code{satlins}}{Symmetric saturating linear function: \eqn{
#'         \begin{cases}1, & if\:x \geq 1 \\ x, & if\:-1<x<1 \\ -1, & if\:x
#'         \leq -1 \end{cases}}}
#'   \item{\code{tansig}}{Tan-sigmoid function: \eqn{2(1 + e^{-2x})^{-1}-1}}
#'   \item{\code{tribas}}{Triangular basis function: \eqn{\begin{cases} 1-|x|,
#'         & if \: -1 \leq x \leq 1 \\ 0, & otherwise \end{cases}}}
#'   \item{\code{poslin}}{Postive linear function: \eqn{\begin{cases} x,
#'         & if\: x \geq 0 \\ 0, & otherwise \end{cases}}}
#'}
#'
#' @return A list containing the trained ICA-ELM model with the following
#'   components.
#' \item{inp_weights}{Weights connecting the input layer to hidden layer,
#'                    obtained from the unmixing matrix \eqn{W} of ICA. The
#'                    columns represent the hidden nodes while rows represent
#'                    input nodes.}
#' \item{out_weights}{Weights connecting the hidden layer to output layer.}
#' \item{fitted.values}{Fitted values of the model.}
#' \item{residuals}{Residuals of the model.}
#' \item{h.out}{A data frame containing the hidden layer outputs (activation
#'              function applied) with columns representing hidden nodes and
#'              rows representing observations.}
#' \item{data}{The univariate \code{ts} data used for training the model.}
#' \item{lags}{Number of lags used during training.}
#' \item{comps}{Number of independent components considered for training. It
#'              determines the number of hidden nodes.}
#' \item{bias}{Whether bias node was included during training.}
#' \item{actfun}{Activation function for the hidden layer.
#'   See `Activation functions`.}
#'
#' @references Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006). Extreme learning
#'   machine: theory and applications. Neurocomputing, 70(1-3), 489-501.
#'   <doi:10.1016/j.neucom.2005.12.126>.
#'
#' Hyvarinen, A. (1999). Fast and robust fixed-point algorithms for independent
#'   component analysis. IEEE transactions on Neural Networks, 10(3), 626-634.
#'   <doi:10.1109/72.761722>.
#'
#' @seealso [ica.elm_forecast()] for forecasting from trained ICA based ELM
#'   model.
#'
#' @examples
#' train_set <- head(price, 12*12)
#' ica.model <- ica.elm_train(train_data = train_set, lags = 12)
#' @export

ica.elm_train <- function(train_data, lags, comps = lags,
                          bias = TRUE, actfun = "sig") {
  if (comps>lags) stop("No. of components must be less than no. of lags.")
  dat_df <- as.data.frame(tsutils::lagmatrix(train_data,
                                             lag = 0:lags)[-(1:lags), ])
  colnames(dat_df) <- c("y", paste0("y_(t-", 1:lags, ")"))
  y <- dat_df$y
  ica_fast <- ica::ica(dat_df[, -1], nc = comps)
  inp_weights <- t(ica_fast$W)
  hidden <- scale(dat_df[, -1], scale = F) %*% inp_weights
  rownames(inp_weights) <-paste0("y_(t-", 1:lags, ")")
  colnames(inp_weights) <- paste0("h", 1:comps)
  h.out <- activate(hidden, actfun = actfun)
  colnames(h.out) <- paste0("h", 1:comps)
  y_h.out <- data.frame("y" = dat_df$y, h.out)
  if (bias == TRUE) {
    X <- stats::model.matrix(y~., data = y_h.out)
    colnames(X)[1] <- "bias"
    out_weights <- as.vector(solve(t(X)%*%X)%*%t(X)%*%y)
    names(out_weights) <- c("bias", paste0("h", 1:comps))
  } else {
    X <- stats::model.matrix(y~.-1, data = y_h.out)
    out_weights <- as.vector(solve(t(X)%*%X)%*%t(X)%*%y)
    names(out_weights) <- paste0("h", 1:comps)
  }
  fitted.values <- X%*%out_weights
  residuals <- y - fitted.values
  out <- list(inp_weights = inp_weights, out_weights = out_weights,
              fitted.values=fitted.values, residuals=residuals, h.out=h.out,
              data = train_data, lags = lags, comps = comps,
              bias = bias, actfun = actfun)
  return(out)
}
