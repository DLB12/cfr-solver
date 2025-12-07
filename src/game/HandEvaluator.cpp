#include "HandEvaluator.hpp"
#include <algorithm>

std::array<Card, 7>
HandEvaluator::mergeHand(const std::array<Card, 2> &hand,
                         const std::array<Card, 5> &communityCards) {
  std::array<Card, 7> fullHand;
  std::copy(hand.begin(), hand.end(), fullHand.begin());
  std::copy(communityCards.begin(), communityCards.end(),
            fullHand.begin() + hand.size());
  return fullHand;
}

constexpr int getSuitValue(Card::SUIT suit) {
  switch (suit) {
  case Card::SUIT::HEARTS:
    return 0;
  case Card::SUIT::DIAMONDS:
    return 1;
  case Card::SUIT::SPADES:
    return 2;
  case Card::SUIT::CLUBS:
    return 3;
  default:
    return 0; // Should never get here
  }
}

// Evaluates the best possible hand rank given a player's hole cards and
// community cards
int HandEvaluator::evaluateHand(const std::array<Card, 2> &hand,
                                const std::array<Card, 5> &communityCards) {
  std::array<Card, 7> fullHand = mergeHand(hand, communityCards);

  std::array<int, 7> cardIndices{};
  int suitHash = 0;
  std::array<int, 4> suitBinary{};
  static constexpr std::array<int, 4> SUIT_SHIFT = {1, 8, 64, 512};

  for (size_t i = 0; i < fullHand.size(); i++) {
    int rankValue = fullHand[i].getValue() - 2;
    int suitValue = getSuitValue(fullHand[i].getSuit());

    cardIndices[i] = (rankValue * 4) + suitValue;
    suitHash += SUIT_SHIFT[suitValue];
    suitBinary[suitValue] |= (1 << rankValue);
  }

  if (SUITS_TABLE[suitHash]) {
    return FLUSH_TABLE[suitBinary[SUITS_TABLE[suitHash] - 1]];
  }

  std::array<unsigned char, 13> rankQuinary{};
  for (const auto &index : cardIndices) {
    rankQuinary[index / 4]++;
  }

  return NOFLUSH_TABLE[hashQuinaryResult(rankQuinary)];
}

// Computes a unique hash for a hand based on rank frequency using precomputed
// DP table
int HandEvaluator::hashQuinaryResult(
    const std::array<unsigned char, 13> &rankQuinary) {
  int sum = 0;
  int remainingCards = 7;

  for (size_t rank = 0; rank < rankQuinary.size(); rank++) {
    sum += DP_TABLE[rankQuinary[rank]][13 - rank - 1][remainingCards];
    remainingCards -= rankQuinary[rank];

    if (remainingCards <= 0)
      break;
  }

  return sum;
}

// Map 0..3 to Card::SUIT using the same convention as getSuitValue
static Card::SUIT suitFromIndex(int s) {
  switch (s) {
  case 0:
    return Card::SUIT::HEARTS;
  case 1:
    return Card::SUIT::DIAMONDS;
  case 2:
    return Card::SUIT::SPADES;
  case 3:
    return Card::SUIT::CLUBS;
  default:
    return Card::SUIT::HEARTS;
  }
}

// Convert integer card id (0..51) to Card
// Assumes: rank = id % 13 gives 0..12 for 2..A
//          suit = id / 13 gives 0..3 for hearts, diamonds, spades, clubs
static Card intToCard(int id) {
  int rankIndex = id % 13; // 0..12
  int suitIndex = id / 13; // 0..3

  int value = rankIndex + 2; // 2..14 (2..A)
  Card::SUIT suit = suitFromIndex(suitIndex);

  // Adjust this line if your Card constructor signature is different
  return Card(static_cast<Card::RANK>(value), suit);
}

// Integer based evaluate - for CFR where cards are 0..51
int HandEvaluator::evaluateHandInts(const std::array<int, 2> &handIdx,
                                    const std::array<int, 5> &boardIdx) {
  std::array<Card, 2> handCards;
  std::array<Card, 5> boardCards;

  for (int i = 0; i < 2; ++i) {
    handCards[i] = intToCard(handIdx[i]);
  }
  for (int i = 0; i < 5; ++i) {
    boardCards[i] = intToCard(boardIdx[i]);
  }

  return evaluateHand(handCards, boardCards);
}
