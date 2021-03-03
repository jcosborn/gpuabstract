#include <complex>

template <typename T>
struct CMat3 {
  //using M = CMat3<T>;
  using V = std::complex<T>[3];
  std::complex<T> m[3][3];
  //V &operator[](int i) { return &m[i][0]; }
  auto operator[](int i) { return m[i]; }
  //auto operator[](M &x, int i) { return x.m[i]; }
};

template <typename T, typename U>
inline void set(CMat3<T> &r, U y)
{
  T x = (T)y;
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {
      r[i][j].real( 1/(x+i+j) );
      r[i][j].imag( 1/(x+i+j+1) );
    }
  }
}

template <typename T>
inline double norm2(CMat3<T> &x)
{
  double r = 0.0;
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {
      double tr = (double) x[i][j].real();
      double ti = (double) x[i][j].imag();
      r += tr*tr + ti*ti;
    }
  }
  return r;
}

template <typename T>
inline void operator-=(CMat3<T> &r, CMat3<T> &x)
{
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {
      r[i][j] -= x[i][j];
    }
  }
}

template <typename T>
inline void mul(CMat3<T> &r, CMat3<T> &x, CMat3<T> &y)
{
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {
      auto t = x.m[i][0] * y.m[0][j];
      for(int k=1; k<3; k++) {
	t += x.m[i][k] * y.m[k][j];
      }
      r[i][j] = t;
    }
  }
}
