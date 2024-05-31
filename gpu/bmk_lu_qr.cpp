#include <iostream>


#include "matrix.hpp"
#include "timer.hpp"


// helper functions
void Copy2Device(double *hptr, int, double *&dptr);

void LUPP(double*, int);
void CPQR(double*, int);


int main(int argc, char *argv[]) {

  int n = 4;

  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
  }

  std::cout
    <<"\n---------------------"
    <<"\nMatrix size:\t"<<n<<std::endl
    <<"---------------------\n"<<std::endl;  

  Mat A = Mat::Random(n,n);

  double *dA;


  Timer t; 
  std::cout.precision(3);
  
  Copy2Device(A.data(), n*n, dA);
  LUPP(dA, n);

  Copy2Device(A.data(), n*n, dA);
  t.start();
  LUPP(dA, n);
  t.stop();
  
  std::cout<<"LUPP: "<<t.elapsed_time()<<std::endl;

  
  Copy2Device(A.data(), n*n, dA);
  CPQR(dA, n);
  
  Copy2Device(A.data(), n*n, dA);
  t.start();
  CPQR(dA, n);
  t.stop();
  
  std::cout<<"CPQR: "<<t.elapsed_time()<<std::endl;




  return 0;
}

