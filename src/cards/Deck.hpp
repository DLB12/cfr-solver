#pragma once

#include "Card.hpp"
#include <array>
#include <random>

class Deck {
private:
  std::array<Card, 52> cards;
  size_t activeSize;
  std::mt19937 rng;

public:
  Deck();
  Card popTop();
  void shuffle();
  int getLength() const;
  void restore(int size) {
    if (size < 0 || size > 52)
      throw std::out_of_range("Invalid deck size");
    activeSize = size;
  }
};
