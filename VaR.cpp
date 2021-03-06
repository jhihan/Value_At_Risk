// VaR.cpp: the main program
//

#include "stdafx.h"
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
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/math/distributions/binomial.hpp>
using boost::math::binomial;
#include <boost/math/distributions/normal.hpp>
using boost::math::normal_distribution;
#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
//#include <boost/math/distributions/pareto.hpp>

//#include <algorithm>
//#include <boost/math/distributions/normal.hpp>

typedef boost::mt19937 base_generator_type;

#include "read_write_vector.h"
#include "VaR.h"
using namespace std;

// --To call random number in normal distriburion
//  std::default_random_engine generator;
//  std::normal_distribution<double> distribution(mean,standard deviation);
//------In the main code
//  number = distribution(generator);   // number the the random number we want




int main() {
//	char* file_name = "MSFT.txt";
	char* file_name = "stock.txt";
//	char* file_name = "HAL.txt";
//	int N_row = 253;
	int N_row = 0;   // Number of the data read. This number must be smller than the number of input data. If this number is set to be 0, then it will finally be determined by the input data.
	int N_column = 2;  // Number of the item in the porfolio.
	int n1 = N_column;
	int n2 = 0;
//  -----VaR related parameters---
	int N_data; // Number of historical data
	int N_day = 1; // Number of day for the VaR calculation
	int N_item = N_column; // Number of item in the portfolio
	int N_trial = 10000; // The number of trial data for bootstrap and Monte-Carlo simulation
	int C_level = 95.0;  // input confidence level for the VaR calculation

	vector<double> row;
	row.assign(n2, 0);// To allocate an 1D vector-array with size n2=0.
	vector< vector<double> > array_2D;
	cout << array_2D.size() << endl;
	array_2D.assign(n1, row);//To allocate a 2D vector-array with the size (n1=N_column)*(n2=0)
	cout << array_2D.size() << endl;


	//---------------------------------------------------
	//  Read the data from the file
	vector_read(N_row, array_2D.begin(), array_2D.end(), file_name);
	N_row = array_2D.begin()->size();
	N_data = N_row;
	cout << "Number of data=" << N_data << endl;

	vector< vector<double> > portfolio;  //The portforlio array
	vector<double> portfolio_row;
	portfolio_row.assign(N_item, 0);
	portfolio.assign(N_data, portfolio_row);
	for (int i = 0; i<N_data; i++) {
		for (int j = 0; j<N_item; j++) {
			portfolio[i][j] = array_2D[j][i];  // The number of data becomes the first dimension
		}
	}
	cout << "Begin VaR calculation" << endl;
	//-------begin the calculation
	ofstream file;
	file.open("Output.txt");
	file << "Input data file:" << file_name << endl;
	file << "Number of item in the portfolio:" << N_item << endl;
	file << "Number of historical data in the portfolio:" << N_data << endl;
	file << "Number of date in the VaR calculation:" << N_day << endl;
	file << "Number of trial data generated in the bootstrap or Monte-Carlo:" << N_trial << endl;


	VaR_object port1(N_data, N_item, N_day, N_trial, C_level);
	port1.pr_calculation(portfolio);
	file << "Covariance Matrix:" << endl;
	for (const auto &row_v : port1.cov)
	{
		for (const auto &i : row_v) file << i << ' ';
		file << std::endl;
	}
	file << "Mean value for each item" << endl;
	for (auto i = port1.mean_v.begin(); i != port1.mean_v.end(); i++)
		file << *i << endl;
//	file << "Expectation value=" << port1.mean << endl;
//	file << "Standard deviation=" << port1.mean << endl;

	port1.VS_calculation(0, C_level);
	port1.Binomial_testing();
	cout << "End of the calculation: historical simulation" << endl;
	cout << "Confident level =" << port1.getClevel() << "(%)" << endl;
	cout << "VaR=" << port1.VaR << endl;
	cout << "Expected shortfall=" << port1.shortfall << endl;
	file << "In the basic historical simulation:" << endl;
	file << "VaR=" << port1.VaR << endl;
	file << "Expected shortfall=" << port1.shortfall << endl;
	file << "In the Binomial backtesting, the expected number of the date with value exceeding VaR: [" << port1.min_test <<","<< port1.max_test<<"]"<<endl;
	file << "In this method, the number of date:" << port1.N_day_VaR << endl;
	file << "This model is"<< " " << port1.Model_accp << endl;
	file << " " << endl;
//	exit(0);

	port1.VS_calculation(1, C_level);
	port1.Binomial_testing();
	cout << "End of the calculation: bootstrap" << endl;
	cout << "Confident level =" << port1.getClevel() << "(%)" << endl;
	cout << "VaR=" << port1.VaR << endl;
	cout << "Expected shortfall=" << port1.shortfall << endl;
	file << "In the bootstrap method:" << endl;
	file << "VaR=" << port1.VaR << endl;
	file << "Expected shortfall=" << port1.shortfall << endl;
	file << "In the Binomial backtesting, the expected number of the date with value exceeding VaR: [" << port1.min_test << "," << port1.max_test << "]" << endl;
	file << "In this method, the number of date:" << port1.N_day_VaR << endl;
	file << "This model is" << " " << port1.Model_accp << endl;
	file << " " << endl;

	port1.VS_calculation(2, C_level);
	port1.Binomial_testing();
	cout << "End of the calculation: Variance-covariance" << endl;
	cout << "Confident level =" << port1.getClevel() << "(%)" << endl;
	cout << "VaR=" << port1.VaR << endl;
	cout << "Expected shortfall=" << port1.shortfall << endl;
	file << "In the variance-covariance method:" << endl;
	file << "VaR=" << port1.VaR << endl;
	file << "Expected shortfall=" << port1.shortfall << endl;
	file << "In the Binomial backtesting, the expected number of the date with value exceeding VaR: [" << port1.min_test << "," << port1.max_test << "]" << endl;
	file << "In this method, the number of date:" << port1.N_day_VaR << endl;
	file << "This model is" << " " << port1.Model_accp << endl;
	file << " " << endl;

	port1.VS_calculation(3, C_level);
	port1.Binomial_testing();
	cout << "End of the calculation: Monte-Carlo" << endl;
	cout << "Confident level =" << port1.getClevel() << "(%)" << endl;
	cout << "VaR=" << port1.VaR << endl;
	cout << "Expected shortfall=" << port1.shortfall << endl;
	file << "In Monte-Carlo method:" << endl;
	file << "VaR=" << port1.VaR << endl;
	file << "Expected shortfall=" << port1.shortfall << endl;
	file << "In the Binomial backtesting, the expected number of the date with value exceeding VaR: [" << port1.min_test << "," << port1.max_test << "]" << endl;
	file << "In this method, the number of date:" << port1.N_day_VaR << endl;
	file << "This model is" << " " << port1.Model_accp << endl;
	file << " " << endl;

	file.close();
//	int max_test;
//	int min_test;
//	int N_day_VaR;
//	Binomial_testing2(&port1, &max_test, &min_test, &N_day_VaR, 95);


	int iiii;
	cin >> iiii;
	return 0;

	//-------------


}
