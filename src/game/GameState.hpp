#pragma once
#include "HandEvaluator.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

constexpr double SMALL_BLIND = 0.5;
constexpr double BIG_BLIND = 1.0;
constexpr double STACK_SIZE = 100.0;
constexpr int MAX_RAISES = 2;

enum class Street { PREFLOP, FLOP, TURN, RIVER, SHOWDOWN };
enum class ActionType { FOLD, CHECK_CALL, BET_RAISE };

struct Action {
  ActionType type;
  double amount;
};

// A lightweight, stack-allocated replacement for std::vector<Action>
struct FixedActions {
  std::array<Action, 3> data;
  int count = 0;

  void push_back(const Action &a) { data[count++] = a; }

  int size() const { return count; }

  const Action *begin() const { return data.data(); }
  const Action *end() const { return data.data() + count; }
  const Action &operator[](int i) const { return data[i]; }
};

class GameState {
public:
  GameState();

  Street street;
  double pot;
  std::array<double, 2> bets;
  std::array<double, 2> stack;
  std::vector<int> board;
  static constexpr int MAX_HISTORY_LENGTH = 64;
  char history[MAX_HISTORY_LENGTH] = "";
  int history_length = 0;
  int active_player;
  int raises_this_street;
  bool is_folded;

  bool isTerminal() const;
  FixedActions getLegalActions() const;
  void applyAction(const Action &action);

  std::string getInfoSetKey(const std::vector<int> &hole_cards) const;
  uint64_t getInfoSetKeyHash(const std::vector<int> &hole_cards) const;
  double getPayoff(const std::vector<int> &p0, const std::vector<int> &p1,
                   HandEvaluator &eval) const;

private:
  void nextStreet();
};
