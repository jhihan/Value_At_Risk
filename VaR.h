#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <functional>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <boost/math/distributions/binomial.hpp>
using boost::math::binomial;
#include <boost/math/distributions/normal.hpp>
using boost::math::normal_distribution;

using namespace std;

//srand (time(NULL));

template <typename T>
vector<int> sort_indexes(const vector<T> &v) {  // By comparing the values of vector-arrat v, soring its index.

	// initialize original index locations
	vector<int> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2];});

	return idx;
}

double GPareto(int N, double beta, double eta, double u, vector <double> v) {
//	boost::math::pareto_distribution <> pareto()

	double sum = 0.0;
	for (int i = 0; i < N;i++) {
		sum += log(1 / beta*pow(1 + eta*(v[i] - u) / beta, -1 / eta - 1));
	}
	return sum;
}

void likelihood(int N, double beta, double eta, double u, vector <double> v) {
	double sum;
	for (int i = 1; i < 10;i++) {
		sum = GPareto(N, beta, eta, u, v);
	}
}

class multi_random {
protected:
	vector<double> mean_array;
	vector<vector<double> > covariance;
	vector<double> rnd_multi;
	//		  Eigen::MatrixXd Cholesky;
public:
	multi_random(int N_size, vector<double> mean_v, vector<vector<double> > covariance_v, base_generator_type & generator);
	//		  inline get_Cholesky(return Cholesky);
	vector<double> get_rnd();

};

multi_random::multi_random(int N_size, vector<double> mean_v, vector<vector<double> > covariance_v, base_generator_type & generator) {
	//	Eigen::VectorXd mean_eigen= VectorXd::Map(mean_v.data(), N_size);
	Eigen::Map<Eigen::VectorXd> mean_eigen(mean_v.data(), N_size);
	Eigen::MatrixXd covariance_eigen(N_size, N_size);
	for (int i = 0; i < N_size; i++) {
		for (int j = 0; j < N_size; j++) {
			covariance_eigen(i, j) = covariance_v[i][j];
		}
	}
	for (int i = 0; i < N_size; i++) {
		mean_eigen(i) = mean_v[i];
	}
	Eigen::MatrixXd Cholesky(N_size, N_size);
	Eigen::LLT<Eigen::MatrixXd> cholSolver(covariance_eigen);
	Cholesky = cholSolver.matrixL();
	boost::normal_distribution<> stan_normal(0.0, 1.0);
	boost::variate_generator<base_generator_type&, boost::normal_distribution<> > uni(generator, stan_normal);
	Eigen::VectorXd rnd_uncorr(N_size);
	Eigen::VectorXd rnd_corr(N_size);
	for (int i = 0; i < N_size; i++) {
		//		rnd_uncorr(i) = quantile(stan_normal, uni());
		rnd_uncorr(i) = uni();
	}
	rnd_corr = Cholesky*rnd_uncorr + mean_eigen;
	rnd_multi.assign(N_size, 0.0);
	for (int i = 0;i < N_size;i++) {
		rnd_multi[i] = rnd_corr(i);
	}
//	for (int i = 0;i < N_size;i++) {
//		cout << rnd_multi[i] << setw(5) << rnd_uncorr(i) << endl;
//	}
}

inline vector<double> multi_random::get_rnd() {
	return rnd_multi;
}


class Porfolio_object
{   
    protected:
        int N_history;
        int N_item;
    public:
        double mean;
        double sigma;
        int getNhistory(){return N_history;}
        int getNitem(){return N_item;}
        vector< vector<double> > portfolio;
        vector< vector<double> > return_daily;
        vector< vector<double> > cov;
		vector<double> mean_v;
        vector< double > value_senario;
        
		Porfolio_object();
		Porfolio_object(int Nhistory, int Nitem);
        void pr_calculation(vector<vector<double> > stock);
};

Porfolio_object::Porfolio_object()
{
    cout <<"set the portforlio and return vector"<<endl;
//    N_history=Nhistory;                                    
//    potforlio.assign(stock.begin(),stock.begin()+N_history+1)
    
}

Porfolio_object::Porfolio_object(int Nhistory, int Nitem)
{
    cout <<"set the portforlio and return vector"<<endl;
    N_history=Nhistory;
    N_item=Nitem;    
                                
//    potforlio.assign(stock.begin(),stock.begin()+N_history+1)
    
}

void Porfolio_object::pr_calculation(vector<vector<double> > stock)
{
    cout <<"Calculate return"<<endl; 
    portfolio.assign(stock.begin(),stock.begin()+N_history);
    vector<double> row;
    
    row.assign(N_item,0);
    return_daily.assign(N_history-1,row);

	mean_v.assign(N_item, 0);
    cov.assign(N_item, row);
    
//    variance.assign(N_item,0);
//    stdev.assign(N_item,0);
    
    value_senario.assign(N_history-1,0);
    cout <<"portfolio is assigned"<<endl; 
	cout << "size=" << portfolio.size() << endl;
	cout << "size=" << portfolio.begin()->size() << endl;
	cout << "size=" << return_daily.size() << endl;
	cout << "size=" << return_daily.begin()->size() << endl;
    for (int i=0;i<N_history-1;i++){
        for (int j=0;j<N_item;j++){
//            return_daily[i][j]=log(portfolio[i+1][j]/portfolio[i][j]);
            return_daily[i][j]=(portfolio[i+1][j]-portfolio[i][j])/portfolio[i][j];
//            variance[j]+=pow(return_daily[i][j],2)
            value_senario[i]=value_senario[i]+portfolio[N_history-1][j]*return_daily[i][j];
//			value_senario[i] = value_senario[i] + return_daily[i][j];
        }
    }

	ofstream dreturn("return.txt");
//	if (!dreturn) {
//		cout << "File can not be opened.\n";
//		return 1;
//	}
	for (int i = 0;i < N_history - 1;i++) {
		for (int j = 0;j < N_item;j++) {
			if (j == (N_item - 1)) {
				dreturn << return_daily[i][j] <<'\n';
			}
			else {
				dreturn << return_daily[i][j] << setw(5);
			}
		}
//		cout << i << "," << return_daily[i][0] << '\n';
	}
	dreturn.close();

	for (int i = 0;i<N_item;i++) {
		for (int j = 0;j<N_history-1;j++) {
			mean_v[i] += (return_daily[j][i] / (N_history - 1));
	    }
	}

    for (int i=0;i<N_item;i++){
        for (int j=0;j<N_item;j++){
            for (int k=0;k<N_history-1;k++){
                cov[i][j]+= (return_daily[k][i]- mean_v[i])*(return_daily[k][j] - mean_v[j])/(N_history-2);
            }
        }
    }
	cout << "mean" << endl;
	for (int i = 0;i < N_item;i++) {
		cout << mean_v[i] << endl;
	}
	cout << "covariance matrix" << endl;
	for (int i = 0;i < N_item;i++) {
		for (int j = 0;j < N_item;j++) {
			cout << cov[i][j] << endl;
		}
	}
//    for (int j=0;j<N_item;j++){
//        variance[j]/=(N_history-1);
//        stdev[j]=sqrt(variance[j]);
//       }
    cout <<"Return is calculated"<<endl; 
}


class VaR_object : public Porfolio_object
{   
    protected:
        double C_level; //confident level
    public:
    int N_day; // the duration for VaR calculation
    int N_trial; //used for the Back_testing
//    int N_history; // the duration of hisory data
    int min_test; // minimal days allowed for back-testing
    int max_test; // maximal days allowed for back-testing
    int N_day_VaR;
    double C_level_back; //confident level for back-tessting
    double C_level_sample; //confident level
    double standard_error;
    double VaR;
    double shortfall;
    string Model_accp;
    double C_interval[2];
	vector <double> weight;
//    vector<double> return_daily; // daily return of the asset    
	VaR_object();
	VaR_object(int Nhistory, int Nitem, int Nday, int Ntrial, double Clevel_back);
    double getClevel(){return C_level;}
    void Quantile_data(int N, double percentile, vector<double> sample);
	void Quantile_data2(int N, double percentile, vector<double> sample);
    void VS_calculation(int N_method, double Clevel); // 0 for normal historical simulation, 1 for bootstrap
    void Historical();
    void Bootstrap();
    void Var_cov();
    void Var_cov2();
    void Monte_Carlo();
    void Binomial_testing();

};

VaR_object::VaR_object()
{
     cout << "Initialize a portfolio" << endl;
     cout << "Parameters are not given" << endl;
     VaR=0.0;
     shortfall=0.0;
     N_day=0;
     N_trial=0;
     C_level_sample=95;
     C_interval[0]=0;
     C_interval[1]=0;
//     N_history=0;
//     return_daily.assign(N_history,0);     
 }

VaR_object::VaR_object(int Nhistory, int Nitem, int Nday, int Ntrial, double Clevel_back) : Porfolio_object(Nhistory,Nitem)
{
     cout << "Initialize a portfolio" << endl;
     cout << "Parameters are given" << endl;
     VaR=0.0;
     shortfall=0.0;
     N_day=Nday;
     N_trial=Ntrial;
     min_test=0;
     max_test=0;
     C_level_sample=95;
     C_level_back=Clevel_back;
     C_interval[0]=0;
     C_interval[1]=0;
     cout << "N_day=" << N_day << endl;
     cout << "N_item=" << N_item << endl;
     cout << "N_history=" << N_history << endl;
     cout << "N_trial=" << N_trial << endl;
//     N_history=Nhistory;
//     return_daily.assign(N_history,0);
 }

void VaR_object::Quantile_data(int N, double percentile,vector<double> sample)
{

    double dindex=percentile*N-0.5;
    int index1,index2;
	int index;
	index = (int)(percentile*N);
    double f1,f2,factor;
    double sum = accumulate(sample.begin(), sample.begin()+N, 0.0);
    double mean = sum / N;
	double sum_weight = 0.0;
//	vector<int> idx;
//	idx.assign(N,0);
//	for (int i = 0; i < N;i++) {
//		idx[i] = i;
//	}
    VaR=0.0;
    shortfall=0.0; 
	vector<int> idx = sort_indexes(sample);
//	for (int i = 0;i < N;i++) {
//		cout << i << "," << idx[i] << endl;
//	}
	int i = 0;
	int i_l, i_r;
	double diff_l, diff_r, diff;
	while(percentile > sum_weight){
		i_l = i-1;
		i_r = i;
		shortfall += weight[idx[i]] * sample[idx[i]];
		diff = percentile - sum_weight;
		sum_weight += weight[idx[i]];
		i = i + 1;
	}
	// VaR is calculated from interpolate the value between the element idx[i_l] and the element  idx[i_r]
//	VaR = (sample[idx[i_l]] * weight[idx[i_l]] * diff_r + sample[idx[i_r]] * weight[idx[i_r]] * diff_l) / (weight[idx[i_l]] * diff_r + weight[idx[i_r]] * diff_l);
	if (i_r == 0) {
		VaR = sample[idx[0]]  -(sample[idx[1]] - sample[idx[0]]) / weight[idx[1]] * (weight[idx[0]] - percentile);
//		shortfall -= (sample[idx[0]] - VaR)*(weight[idx[0]] + (weight[idx[0]] - percentile) / weight[idx[1]]) / 2;  // subtract the part which is overestimated.
		shortfall -= (percentile + weight[idx[0]])*(sample[idx[0]] - VaR) / 2.0;
	}
	else {
		VaR = sample[idx[i_l]] + (sample[idx[i_r]] - sample[idx[i_l]]) / weight[idx[i_r]] * diff;
//		shortfall -= sample[idx[i_r]]  * (sum_weight- percentile) ;  // subtract the part which is overestimated.
		shortfall -= (percentile + sum_weight)*(sample[idx[i_r]] - VaR) / 2.0;
	}
	shortfall = shortfall / percentile;   //expected shortfall


//    cout << "N=" << N << endl;
//    cout << "size=" << sample.size() << endl;
//    cout << "sum=" << sum << endl;
//   cout << sample[0] << setw(5) << sample[1] << endl;
//    cout << "mean=" << mean << endl;
    
//    if (dindex < 0) {
//           index1=0;
//           index2=1;
//            nth_element(sample.begin(), sample.begin() + index1+1, sample.begin()+N);
//            nth_element(sample.begin(), sample.begin() + index2+1, sample.begin()+N);    
//            f1=sample[index1];
//            f2=sample[index2];
//            VaR=f1 + (f1-f2)*dindex;
//            shortfall = -0.5*(f2 -f1)*dindex*dindex - dindex* f1;
//         }
//    else if (dindex > (N-1)) {
//            index1=N-1;
//            index2=N-2;
//            nth_element(sample.begin(), sample.begin() + index1+1, sample.begin()+N);
//            nth_element(sample.begin(), sample.begin() + index2+1, sample.begin()+N); 
//            f1=sample[index1];
//            f2=sample[index2];
//            VaR=f1 + (f1-f2)*(dindex-(N-1));
//            shortfall = accumulate(sample.begin(), sample.begin()+ index1+1, 0.0);
 //        }
 //   else {
//         index1=(int) dindex;
//         index2=((int) dindex)+1;
//         nth_element(sample.begin(), sample.begin() + index1+1, sample.begin()+N);
//         nth_element(sample.begin(),sample.begin() + index2+1, sample.begin()+N); 
//         f1=sample[index1];
//         f2=sample[index2];
//         VaR= f1+ (f2-f1)* (dindex-index1);
//         factor=(dindex-index1-0.5)*f1+0.5*(f2-f1)*(dindex-index1)*(dindex-index1);
//         shortfall = accumulate(sample.begin(), sample.begin()+ index1+1, factor);
//         }     
//     VaR = -(VaR - mean);
//     shortfall = -(shortfall/N - mean);
//	 std::sort(sample.begin(), sample.end());
//	 cout << "VaR test -2" << sample[index-2]  << endl;
//	 cout << "VaR test -1" << sample[index-1] << endl;
//	 cout << "VaR test 0" << sample[index] << endl;
//	 cout << "VaR test +1" << sample[index+1] << endl;
//	 cout << "VaR test +2" << sample[index+2] << endl;
     VaR=-VaR;
     shortfall=-shortfall;
//     cout << "VaR=" <<VaR << endl;
//     cout << "shortfall=" <<shortfall << endl;
 }
void VaR_object::Quantile_data2(int N, double percentile, vector<double> sample) {
	double dindex = percentile*N;
	int index1, index2;
	int index;
	index = (int)(percentile*N);
	double f1, f2;
	VaR = 0.0;
	shortfall = 0.0;
	if (dindex < 1) {
		index1 = 0;
		index2 = 1;
		nth_element(sample.begin(), sample.begin() + index1 + 1, sample.begin() + N);
		f1 = sample[index1];
		nth_element(sample.begin(), sample.begin() + index2 + 1, sample.begin() + N);
		f2 = sample[index2];
		VaR = f1 + (f1 - f2)*(1.0 - dindex);
		shortfall = sample[0] - (percentile + 1.0 / N)*(sample[0] - VaR) / 2.0;
	}
	else {
		index1 = (int)dindex;
		index2 = ((int)dindex) + 1;
		nth_element(sample.begin(), sample.begin() + index1, sample.begin() + N);
		f1 = sample[index1];
		nth_element(sample.begin(), sample.begin() + index2, sample.begin() + N);
		f2 = sample[index2];
		VaR = f1 + (f2 - f1)* (dindex - index1);
		shortfall += accumulate(sample.begin(), sample.begin() + index2, 0.0)* (1.0 / N);
		shortfall -= (percentile + 1.0 / N * (index2 + 1))*(f2 - VaR) / 2.0;
	}
	shortfall = shortfall / percentile;   //expected shortfall
	shortfall = -shortfall;
	VaR = -VaR;
}

void VaR_object::Bootstrap()
{
    /* initialize random seed: */
//    srand (time(NULL));
    int iSecret;
    double VaR_sum=0;
    double shortfall_sum=0;
    double percentile=1-C_level*0.01;
    vector<double> sample;
    vector <double> VaR_trial;
    VaR_trial.assign(N_trial,0); 
	base_generator_type generator(static_cast<unsigned int>(time(NULL)));
	boost::random::uniform_int_distribution<> uniform_int(0, N_history - 2);
//    N_trial=500;

//	weight.assign(N_trial, 1.0 / N_trial);
	weight.assign(N_history - 1, 1.0 / (N_history - 1));
    cout << "Begin Bootstrap" << endl;
    cout << "N_trial" << setw(5)<< N_trial << endl;
    cout << "N_day" << setw(5)<< N_day << endl;
    int icount=0;
    for (int i=0; i<N_trial ; i++){
//		cout << "trial," << setw(5) << i << endl;
        sample.assign(N_history-1,0);
        for (int j=0; j<N_history-1 ; j++){
                /* generate secret number between 0 and N_sample-1: */
//           iSecret = rand() % (N_history-1) + 0;
			iSecret = uniform_int(generator);
//                cout << "iSecret" << setw(5)<< iSecret<< endl;
//                samplerial[i]=samplerial[i]+return_daily[iSecret];
            for (int k=0; k<N_item ; k++){
                sample[j]=sample[j]+portfolio[N_history-1][k]*return_daily[iSecret][k];
            }
                icount=icount+1;
        }
        Quantile_data2(N_history-1, percentile, sample);
        VaR_sum+=VaR;
        VaR_trial[i]=-VaR;
        shortfall_sum+=shortfall;
    }
	weight.assign(N_trial, 1.0 / N_trial);
    percentile=(1-C_level_sample*0.01)/2;
    Quantile_data2(N_trial, percentile, VaR_trial);
    C_interval[1]=VaR;
    percentile=1-(1-C_level_sample*0.01)/2;
    Quantile_data2(N_trial, percentile, VaR_trial);
    C_interval[0]=VaR;
    VaR=VaR_sum/N_trial;
    shortfall=shortfall_sum/N_trial;
    cout << "VaR=" << setw(5)<< VaR << endl;
    cout << "confidence interval:[" << C_interval[0] << "," << C_interval[1] << "]" <<endl;
    
//    Quantile_data(N_trial, samplerial);
//    Risk(N_sample, N_day, C_level, VaR, shortfall, samplerial);
//      double VaR0;
//      double shortfall0;
//      Risk(N_sample, N_day, C_level, &VaR0, &shortfall0, samplerial);
//      *VaR=VaR0;
//      *shortfall=shortfall0;
    
}


void VaR_object::Var_cov(){
     
     double var_port=0;
     double mean=0.0;
     double sigma;
     for (int i=0;i<N_item;i++){
         for (int j=0;j<N_item;j++){
             var_port+=portfolio[N_history-1][i]*cov[i][j]*portfolio[N_history-1][j];
         }
     }
	 for (int i = 0;i < N_item;i++) {
		 mean += mean_v[i] * portfolio[N_history - 1][i];
	 }
//	 var_port = cov[0][0];
//	 mean = mean_v[0];
     cout << "variance"<< var_port << endl;
     sigma=sqrt(var_port);
     cout << "standard deviation"<< sigma << endl;
     cout << "mean"<< mean << endl;
     normal_distribution<> myNormal(mean, sigma);
     double percentile=1.0-C_level*0.01;
     VaR=-quantile(myNormal, percentile);
     cout << "VaR: " << VaR << endl;
     shortfall=0.0;
     shortfall=mean-pow(sigma,2)*(pdf(myNormal, -VaR))/(cdf(myNormal, -VaR));
     shortfall=-shortfall;
     cout << "Expected shortfall:"<<shortfall <<endl;
}



void VaR_object::Monte_Carlo(){
      cout <<"Monte-Carlo"<< endl;
	  double percentile = 1 - C_level*0.01;

	  base_generator_type generator(static_cast<unsigned int>(time(NULL)));
//      portfolio[N_history-1][i]*cov[i][j]*portfolio[N_history-1][j]
//      default_random_engine generator;
//      normal_distribution<double> distribution(mean,sigma);
//     Initialize Price[from 0 to N_trial-1][from 0 to (N_item-1)]=Portfolio[N_history-1][from 0 to (N_item-1)] // the initial prize is the prize in the last day
	  vector<double> sample;
	  sample.assign(N_trial, 0.0);
	  vector< vector<double> > price;
	  price.assign(N_trial, portfolio[N_history - 1]);
	  weight.assign(N_trial, 1.0 / N_trial);
	  cout << "Monte-Carlo: price initialization" << endl;
	  for (int k = 0;k < N_item;k++) {
		  cout << "0"<<k << setw(5)<<price[0][k]<<endl;
		  cout << "1"<<k << setw(5)<<price[1][k]<<endl;
	  }
	  cout << "Mean vector" << endl;
	  for (int k = 0;k < N_item;k++) {
		  cout << k << setw(5) << mean_v[k]<< endl;
	  }
	  cout << "Covariance" << endl;
	  for (int i = 0;i < N_item;i++) {
		  for (int j = 0;j < N_item;j++) {
			  cout << i << "," << j<< setw(5) << cov[i][j] << endl;
		  }
	  }
	  for (int i = 0;i<N_trial;i++) {
		  for (int j = 0;j < N_day;j++) {
			  multi_random multi(N_item, mean_v, cov, generator);
			  vector<double> rdn_multi = multi.get_rnd();
			  for (int k = 0;k < N_item;k++) {
				  price[i][k] = price[i][k] * (1+rdn_multi[k]);
			  }
		  }
      }

	  // Calculate the change of the value of portforlio.
	  for (int i = 0;i < N_trial;i++) {
		  for (int j = 0; j < N_item;j++) {
			  sample[i] += price[i][j] - portfolio[N_history - 1][j];
		  }
//		  cout << i << "," << samplerial[i] << "," << price[i][0] << "," << portfolio[N_history - 1][0] << endl;
	  }
	  Quantile_data(N_trial, percentile, sample);

}

void VaR_object::Historical(){
     
     double percentile=1-C_level*0.01;
     double percentile_limit=1-(1-C_level_sample*0.01)/2;
     double pd;
     double mean=0;
     double sigma=0;
     double variance=0;     
     

	 weight.assign(N_history - 1, 1.0 / (N_history - 1));
     for (int i=0;i<N_history-1;i++){
         mean+=value_senario[i];
     }
     mean=mean/(N_history-1);
     for (int i=0;i<N_history-1;i++){
         variance+=pow(value_senario[i]-mean,2);
     }
     sigma=sqrt(variance/(N_history-2));
     Quantile_data(N_history-1,percentile,value_senario);
     normal_distribution<> myNormal(mean, sigma);
     pd=pdf(myNormal,-VaR);
     standard_error=sqrt(percentile*(1-percentile)/(N_history-1))/pd;
     normal_distribution<> St_Normal(0, 1);
     double u=quantile(St_Normal, percentile_limit);
     C_interval[0]=VaR-u*standard_error;
     C_interval[1]=VaR+u*standard_error;
     cout << "standard error" << setw(5) << standard_error <<endl;
     cout << "confidence interval:[" << C_interval[0] << "," << C_interval[1] << "]" <<endl;
}

void VaR_object::VS_calculation(int N_method, double Clevel){
     
     C_level=Clevel;
     switch(N_method){
         case 0:
              cout << "Method choose: historical simulation------ \n";
             Historical();
             break;
         case 1:
             cout << "Method choose: Bootstrap------ \n"; 
             Bootstrap();
             break;
         case 2:
             cout << "Method choose: Variance-covariance------ \n"; 
             Var_cov();
             break;    
         case 3:
             cout << "Method choose: Monte-Carlo------ \n"; 
			 Monte_Carlo();
             break;    
         default:
             cout << "The input parameter for VS_calculation must be 0,1, 2 or 3 \n"; 
             break;
     }
     cout << "End of the calculation0 \n";
//     Bootstrap();
     
     }

void VaR_object::Binomial_testing()
{
    double test=(1.0-C_level_back*0.01)/2.0;
    double prob=0.0;
    double p=1.0-C_level*0.01;
    int k=0;
    while ( prob < test)
    {
        k=k+1;  
        prob=cdf(binomial(N_history, p), k);        
     }
     min_test=k;
     
    test=C_level_back*0.01+(1.0-C_level_back*0.01)/2.0;
    prob=1.0;
    k=N_history;
    while ( prob > test)
    {
        k=k-1;  
        prob=cdf(binomial(N_history, p), k);
     }
    max_test=k;     
     
    int icount=0;
    for (int i=0; i<N_history-1 ; i++){
        if (value_senario[i]< (-VaR)){
            icount++;
        }
    }
    N_day_VaR=icount;
    if ( ( (N_day_VaR)>= min_test) && ( (N_day_VaR)<= max_test)){
         Model_accp="accepted";
    }
    else {
         Model_accp="rejected";
    }
    cout << "This model is"<< setw(5) << Model_accp <<  endl;
    cout << "The number of dates with loss exceeded:" << N_day_VaR << endl;
    cout << "The minimal allowed dates:" << min_test << endl;
    cout << "The maximal allowed dates:" << max_test << endl;
}

void Binomial_testing2(VaR_object* port_VaR, int* max_test, int* min_test, int* N_day_VaR ,double C_level_back){

    double test=(1.0-C_level_back*0.01)/2.0;
    double prob=0.0;
    double p=1.0-(port_VaR->getClevel())*0.01;
    int N_history=port_VaR->getNhistory();
    double VaR=port_VaR->VaR;
    vector <double> value_senario=port_VaR->value_senario;
    string Model_accp=port_VaR->Model_accp;
    int k=0;
    while ( prob < test)
    {
        k=k+1;  
        prob=cdf(binomial(N_history, p), k);        
     }
     *min_test=k;
     
    test=C_level_back*0.01+(1.0-C_level_back*0.01)/2.0;
    prob=1.0;
    k=N_history;
    while ( prob > test)
    {
        k=k-1;  
        prob=cdf(binomial(N_history, p), k);
     }
    *max_test=k;     
     
    int icount=0;
    for (int i=0; i<N_history-1 ; i++){
        if (value_senario[i]< (-VaR)){
            icount++;
        }
    }
    *N_day_VaR=icount;
    if ( ( (*N_day_VaR)>= *min_test) && ( (*N_day_VaR)<= *max_test)){
         Model_accp="accepted";
    }
    else {
         Model_accp="rejected";
    }
    cout << "This model is"<< setw(5) << Model_accp <<  endl;
    cout << "The number of dates with loss exceeded:" << *N_day_VaR << endl;
    cout << "The minimal allowed dates:" << *min_test << endl;
    cout << "The maximal allowed dates:" << *max_test << endl;

     }
