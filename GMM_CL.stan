data {
  int<lower=1> N;                                // number of observations
  real y[N];                                     // y
  real<lower=0> time[N];                         // predictor ((age)time)
  // grouping factor
  int<lower=1> J;                                // number of subjects
  int<lower=1,upper=J> Subject[N];               // subject id
  int<lower=1> s[J];                             // subject size
  // class factor
  int<lower=1> K;                                // number of latent classes
}

parameters {
  simplex[K] lambda;                            // mixing proportions
  vector[K] mu_intercept;                       // class-specific fixed intercept
  vector[K] mu_slope;                           // class-specific fixed slope
  vector<lower=0>[2] sigma_k[K];              // class-specific std of random effects 
  cholesky_factor_corr[2] L_k[K];             // the Choleski factor of a 2x2 class-specific correlation matrix
  vector [2] z[K];                         // random effect matrix
  real<lower=0> sigma_e;                        // fixed scale of mixture components
}

transformed parameters {
  vector[2] Sigma[K];
  //vector[K] lps[J];
  vector[K] mu_k;
  vector[J] log_liks;
  
  for (k in 1:K) {
    Sigma[k] = diag_pre_multiply(sigma_k[k], L_k[k]) * z[k];
  }
  {
  int pos;
  int target1;
  pos = 1;
  target1 = pos;
  for(j in 1:Subject[N]){
    vector[K] lps = log(lambda);
    target1 += s[j];
    while (pos < target1){
      for (k in 1:K){
        mu_k[k]= mu_intercept[k]+Sigma[k][1] + (mu_slope[k]+Sigma[k][2])* time[pos];
        lps[k] += normal_lpdf(y[pos] | mu_k[k], sigma_e);
      }
      pos = pos+1;
    }
    log_liks[j] = log_sum_exp(lps);
    if (pos>N) break;
    }
}
}

model {
  sigma_e ~ exponential(1);
  lambda ~ dirichlet(rep_vector(2.0, K));
  for (k in 1:K){
    sigma_k[k] ~  cauchy(0, 5);
    L_k[k] ~ lkj_corr_cholesky(2); // LKJ prior for the correlation matrix
    to_vector(z[k]) ~ std_normal();
    mu_intercept[k] ~ normal(0, 5);
    mu_slope[k] ~ normal(0, 5);
    }
    
  for(j in 1:Subject[N]) target += log_liks[j];
}

generated quantities {
  corr_matrix[2] Omega_k[K];
  vector[J] log_lik;
  int<lower = 1> pred_class_dis[J];     // posterior prediction for respondent j in latent class c
  simplex[K] pred_class[J];
  int pos;
  int target1;
  for (k in 1:K){
    Omega_k[k] =  multiply_lower_tri_self_transpose(L_k[k]);  // so that it returns the correlation matrix, which is equal to L_u * L_u';
  }

  pos = 1;
  target1 = pos;
  for(j in 1:Subject[N]){
    vector[K] lps = log(lambda);
    target1 += s[j];
    while (pos < target1){
      for (k in 1:K){
        lps[k] += normal_lpdf(y[pos] | mu_k[k], sigma_e);
      }
      pos = pos+1;
    }
    log_lik[j] = log_sum_exp(lps);
    for (k in 1: K){
      pred_class[j][k] = exp((lps[k])-log_sum_exp(lps));
    }
    pred_class_dis[j] = categorical_rng(pred_class[j]);
  }
}


