#include <array>
#include <random>

class Deck {
public:
  Deck();

  int popTop();

  void shuffle();

  int getLength() const;

private:
  std::array<int, 52> cards;

  std::mt19937 rng;
  int activeSize;
};
