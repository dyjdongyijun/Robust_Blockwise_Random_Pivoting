#ifndef _util_hpp_
#define _util_hpp_

#include <string>
#include <iostream>


template <typename T>
void print(T *v, int n, std::string name) {
  std::cout<<"\n"<<name<<":\n";
  for (int i=0; i<n; i++)
    std::cout<<v[i]<<" ";
  std::cout<<std::endl;
}


#endif
