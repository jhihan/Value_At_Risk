# Value_At_Risk
This project is developed to evaluate Value at Risk (VAR) with binomial backtesting for Stock Portfolio. This code is constructed in the end of 2017 in order to practice C++ and financial engineering skills. Further improvement will be made soon.
# Getting Started
## Environment
The code is run in the Visual C++ 17 environment.
## Installing
Boost and Eigen library
## files
### VaR.cpp: 
Main file
### VaR.h:
The calsses and functions used in VaR.cpp
### read_write_vector.h
The functions for reading adn writting the vector from/to a file.
### stock.txt
The historical data of stock portfolio. Each coloumn means different stock and each row means the value of different days.
# usage:
This program is used to do the risk management calculation, which including VaR and expected shortfall. In the beginning, the historical stock prices data must be prepared in the file stock.txt. When running the program, the covariance matrix of the historical stock price will be calculated first. And then different mwthods -- historical simulation, variance-covariance method, and Monte Carlo simulation-- will be performed. In the end, binomial backtesting is used to test the VaR models.
