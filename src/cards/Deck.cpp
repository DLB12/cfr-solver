#include "Deck.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

Deck::Deck() : cards{}, rng(std::random_device{}()), activeSize(52) {
  for (int i = 0; i < 52; ++i) {
    cards[i] = i + 1;
  }
}

int Deck::popTop() {
  if (activeSize <= 0) {
    std::cerr << "Error: No cards left in deck\n";
    throw std::out_of_range("No cards left in deck");
  }

  return cards[--activeSize];
}

void Deck::shuffle() {
  activeSize = 52;
  std::shuffle(cards.begin(), cards.begin() + activeSize, rng);
}

int Deck::getLength() const { return activeSize; }
