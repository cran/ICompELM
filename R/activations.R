#' Aggregate gram price data
#'
#' National aggregate price of gram from Indian markets, which is a major pulse
#'   in the country. The observations range from January, 2010 upto December,
#'   2023.
#'
#' @name price
#' @docType data
#' @usage price
#' @format A \code{ts} object with 156 observations.
#' @source \url{https://www.agmarknet.gov.in/}
#' @examples
#' plot(price, xlab = "Year", ylab = "Aggregate price of Gram (Rs./Bag)")
#' @keywords data
NULL

sig <- function(x) { # Sigmoid activation function
  return((1 + exp(-1*x))^(-1))
}

radbas <- function(x) { # Radial basis activation function
  return(exp(-1*(x^2)))
}

hardlim <- function(x) { # Hard-limit activation function
  return(ifelse(x >= 0, 1, 0))
}

hardlims <- function(x) { # Symmetric hard-limit activation function
  return(ifelse(x >= 0, 1, -1))
}

satlins <- function(x) { # Symmetric saturating linear activation function
  return(ifelse(x >= 1, 1, ifelse(x <= -1, -1, x)))
}

tansig <- function(x) { # Tan-sigmoid activation function
  return(2/(1+exp(-2*x))-1)
}

tribas <- function(x) { # Triangular basis activation function
  return(ifelse(x >= -1 & x <= 1, 1 - abs(x), 0))
}

poslin <- function(x) { # Sigmoid activation function
  x[x < 0] <- 0
  return(x)
}

activate <- function(x, actfun) {
  if(actfun == "sig") value = as.data.frame(sig(x))
  else if(actfun == "sin") value = as.data.frame(sin(x))
  else if(actfun == "radbas") value = as.data.frame(radbas(x))
  else if(actfun == "hardlim") value = as.data.frame(hardlim(x))
  else if(actfun == "hardlims") value = as.data.frame(hardlims(x))
  else if(actfun == "satlins") value = as.data.frame(satlins(x))
  else if(actfun == "tansig") value = as.data.frame(tansig(x))
  else if(actfun == "tribas") value = as.data.frame(tribas(x))
  else if(actfun == "poslin") value = as.data.frame(poslin(x))
  else if(actfun == "purelin") value = as.data.frame(x)
  return(value)
}
