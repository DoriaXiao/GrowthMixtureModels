data {
  int<lower=1> N;                      // number of observations
  real y[N];                           // y
  real<lower=0> time[N];               //predictor ((age)time)
  // grouping factor
  int<lower=1> J;                      //number of subjects
  int<lower=1,upper=J> Subject[N];     //subject id
  int<lower=1> s[J];                   //subject size
  // class factor
  int<lower=1> K;                      // number of latent classes
  real normal_scale_lnsd;
  real normal_scale_zrho;
}

parameters {
  simplex[K] lambda;                  // mixing proportions
  vector[K] mu_intercept;             // class-specific fixed intercept
  vector[K] mu_slope;                 // class-specific fixed slope
  vector[2] mu_sd;  // mean log standard deviations across classes  
  vector[2] lnsd[K]; // class-specific log standard deviation
  real mu_zrho; // the mean of Fisher's z transformation
  real zrho[K]; // class-specific Fisher's z transformation
  matrix [2, J] z_u[K];  
  real<lower=0> sigma_e;              // fixed scale of mixture components
}

transformed parameters {
  vector[2] sigma_u[K]; // true class-specific standard deviation
  corr_matrix[2] Omega[K]; // true class-specific correlation matrix
  cholesky_factor_corr[2] L_Omega[K];
  matrix[2, J] u[K];
  vector[K] lps[J];
  vector[J] log_liks;
  for(k in 1:K){
    sigma_u[k] = exp(lnsd[k]);
    Omega[k][1,1]= 1;
    Omega[k][2,2]= 1;
    Omega[k][2,1]= tanh(zrho[k]);
    Omega[k][1,2]= tanh(zrho[k]);
    L_Omega[k] = cholesky_decompose(Omega[k]);
    u[k] = diag_pre_multiply(sigma_u[k], L_Omega[k]) * z_u[k];
  }
{
  int pos;
  int target1;
  pos = 1;
  target1 = pos;
  for(j in 1:Subject[N]){
    lps[j] = log(lambda);
    target1 += s[j];
    while (pos < target1){
      for (k in 1:K){
        real mu_k= mu_intercept[k]+u[k][1,j] + (mu_slope[k]+u[k][2,j])* time[pos];
        lps[j][k] += normal_lpdf(y[pos] | mu_k, sigma_e);
      }
      pos = pos+1;
    }
    log_liks[j] = log_sum_exp(lps[j]);
    if (pos>N) break;
    }
}
}



model {
  mu_sd ~ normal (0, 1);
  mu_zrho ~ normal (0, 1);
  sigma_e ~ exponential(1);
  lambda ~ dirichlet(rep_vector(2.0, K));
  for (k in 1:K){
    mu_intercept[k] ~ normal(0, 5);
    mu_slope[k] ~ normal(0, 5);
    lnsd[k] ~ normal (mu_sd, normal_scale_lnsd);
    zrho[k] ~ normal (mu_zrho, normal_scale_zrho);
    to_vector(z_u[k]) ~ std_normal();
  }
   for(j in 1:Subject[N]) target += log_sum_exp(lps[j]);
}

generated quantities {
  int<lower = 1> pred_class_dis[J];       // posterior prediction for respondent j in latent class c
  simplex[K] pred_class[J];               // posterior probabilities of respondent j in latent class c
  for(j in 1:Subject[N]){
    pred_class[j] = softmax(lps[j]);
    pred_class_dis[j] = categorical_rng(pred_class[j]);
  }
}


