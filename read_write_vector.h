using namespace std;

void write_data(int N_row, int N_column, double **oarray, char* filename)
{
    cout << "writing data in to the file:" << filename << endl;

    ofstream file(filename);
//    file.setf(ios:right,ios:adjustfield);
    for(int ir=0; ir<N_row; ir++){
           for(int ic=0; ic<N_column; ic++){
               file.width(5);
               file <<    *(*(oarray+ir)+ic) ;
               if (ic == N_column-1) {
                      file << endl; 
                  }
              }
    }

    file.close();         
     
}

// T=vector< vector<int> >::iterator
//template <class TT>
template <typename Tite>
// Tite= vector< vector<double> >::iterator
void vector_read(int N_row, Tite ite1 , Tite ite2 , char* filename)
{
       cout << "type name test" << endl;
       cout << "reading data from the file:" << filename << endl;
       fstream file;
       file.open(filename);
//	   string line;
       Tite ite;
       double x;
	   if (N_row > 0) {
		   for (int ir = 0; ir < N_row; ir++)
		   {   
			   if (!file.eof()) {
				   for (ite = ite1; ite != ite2;ite++)
				   {
					   file >> x;
					   ite->push_back(x);
					   //               cout << x << setw(5) << endl;
				   }
			   }
		   }
	   }
	   else if (N_row == 0) {
		   int i = 0;
		   while(!file.eof()) {
			   for (ite = ite1; ite != ite2;ite++)
			   {
				   file >> x;
				   if (!file.eof()) {
					   ite->push_back(x);
					   i = i + 1;
				   }
			   }
		   }
	   }
	   else {
		   cout << "Please enter a positive integer.\n";
	   }
	   file.close();
}


void read_data(int N_row, int N_column, double **oarray, char* filename)
{
    cout << "reading data from the file:" << filename << endl;
    fstream file;
    file.open(filename);

        for(int ir=0; ir<N_row; ir++){
           for(int ic=0; ic<N_column; ic++){
                 file >> *(*(oarray+ir)+ic);
            }
        }
    file.close();
     
}

void test_data(int N_row, int N_column, double **oarray)
{
    cout << "test the data (out put it to the screen),"<< endl;
    for(int ir = 0; ir < N_row; ir++) {
            for(int ic = 0; ic < N_column; ic++){
             cout.width(5);
             cout << oarray[ir][ic];
             if (ic == N_column-1) {
                      cout << endl; 
                  }
             }
        }
}
