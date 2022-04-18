functions{
  real mmn(real mu_intercept, real mu_slope, vector time_seg, 
          matrix Sigma, real sigma_e, vector y_seg){
    vector[rows(time_seg)] mu_seg;
    matrix[rows(time_seg),2] Z_j;
    matrix[rows(time_seg),rows(time_seg)]Cov_j;
    mu_seg = mu_intercept + (mu_slope)* time_seg;
    Z_j = append_col(rep_vector(1.0, rows(time_seg)), time_seg);
    Cov_j = Z_j * Sigma * Z_j'+diag_matrix(rep_vector(sigma_e^2, rows(time_seg)));
    return multi_normal_lpdf (y_seg| mu_seg, Cov_j);
  }
}

data {
  int<lower=1> N;                      // number of observations
  // grouping factor
  int<lower=1> J;                      //number of subjects
  int<lower=1,upper=J> Subject[N];     //subject id
  int<lower=1> s[J];                   //subject size
  vector[N] y;                           // y
  vector[N] time;               //predictor ((age)time)
  // class factor
  int<lower=1> K;                      // number of latent classes
}

transformed data{
  int pos[J];
  int new_j=1;
  int curr_sub = Subject[1];
  pos[1]=1;
  for (n in 1:N){
    if (Subject[n] != curr_sub){
      new_j+=1;
      pos[new_j]=n;
      curr_sub = Subject[n];
    }
  }
}

parameters {
  simplex[K] lambda;                  // mixing proportions
  vector[K] mu_intercept;             // class-specific fixed intercept
  vector[K] mu_slope;                 // class-specific fixed slope
  vector<lower=0>[2] sigma_u[K];    
  cholesky_factor_corr[2] L_Omega[K];   
  real<lower=0> sigma_e;              // fixed scale of mixture components
}

transformed parameters {
  corr_matrix[2] Omega[K];     // class-specific correlation matrix
  cov_matrix[2] Sigma[K];      // class-specific covariance matrix
  vector[K] lps[J];
  for (k in 1: K){
    Omega[k] =  multiply_lower_tri_self_transpose(L_Omega[k]);
    Sigma[k]= quad_form_diag(Omega[k], sigma_u[k]);
  }
  for(j in 1:Subject[N]){
    lps[j] = log(lambda);
    for (k in 1:K){
      lps[j][k] += mmn(mu_intercept[k], mu_slope[k], segment(time, pos[j],s[j]), 
                      Sigma[k], sigma_e, segment(y, pos[j],s[j]));
    }
  }
}

model {
  for (k in 1:K){
    mu_intercept[k] ~ normal(0, 5);
    mu_slope[k] ~ normal(0, 5);
    sigma_u[k] ~  cauchy(0, 5);
    L_Omega[k] ~ lkj_corr_cholesky(2);  
  }
  sigma_e ~ exponential(1);
  lambda ~ dirichlet(rep_vector(2.0, K));
  for(j in 1:Subject[N]) target += log_sum_exp(lps[j]);
}

generated quantities {
  vector[J] log_lik;
  int<lower = 1> pred_class_dis[J];       // posterior prediction for respondent j in latent class c
  simplex[K] pred_class[J];               // posterior probabilities of respondent j in latent class c
  for(j in 1:Subject[N]){
    log_lik[j] = log_sum_exp(lps[j]);
    pred_class[j] = softmax(lps[j]);
    pred_class_dis[j] = categorical_rng(pred_class[j]);
  }
}
