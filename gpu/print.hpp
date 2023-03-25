#ifndef _print_hpp_
#define _print_hpp_

template <typename T>
void print(const T& vec, const std::string &name) {
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (unsigned i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl<<std::endl;
}


template <typename T>
void print(const T& vec, int m, int n, const std::string &name) {
  // print matrix in column ordering
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++)
      std::cout<<vec[i+j*m]<<" ";
    std::cout<<std::endl;
  }
}

#endif
