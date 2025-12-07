#include "Card.hpp"
#include <fmt/core.h>

Card::Card(RANK rank, SUIT suit) : rank(rank), suit(suit) {}
Card::Card() : rank(RANK::INVALID), suit(SUIT::INVALID) {}

Card::RANK Card::getRank() const { return rank; }

Card::SUIT Card::getSuit() const { return suit; }

int Card::getValue() const { return static_cast<int>(rank); }
