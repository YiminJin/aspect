// c++ -std=c++17 check_event.cc -lboost_serialization -o /tmp/bp3-event && /tmp/bp3-event
#include "first_event.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cassert>
#include <sstream>
#include <vector>
#include <iostream>

int main()
{
  constexpr double threshold=1.e-3;
  const std::vector<double> peaks= {1.e-9, threshold, .5, .0007, .0006, threshold, .0009, .0008, .0007, .0006, .0005};
  BP3::FirstEvent clean, continued;
  for (unsigned int i=0; i<peaks.size(); ++i)
    {
      const bool milestone=clean.observe(i,peaks[i]);
      assert(milestone==(i==1 || i==10));
      // A declined attempt at the third clock never reaches the accepted observer.
      auto declined=continued;
      declined.observe(i,1.e-4);
      continued.observe(i,peaks[i]);
      assert(continued.below==clean.below && continued.onset==clean.onset && continued.completion==clean.completion);
      std::stringstream bytes;
      {
        boost::archive::binary_oarchive out(bytes);
        out << continued;
      }
      continued= {};
      {
        boost::archive::binary_iarchive in(bytes);
        in >> continued;
      }
    }
  assert(continued.onset==1 && continued.completion==10 && continued.below==5);
  std::cout << "Onset 1, equality reset 5, completion 10; all 11 serialization boundaries agree.\n";
}
