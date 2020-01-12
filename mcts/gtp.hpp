#pragma once
#include "agent.hpp"
#include "board.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

struct Position {
  Position() = default;
  Position(const Position &) = default;
  Position(Position &&) noexcept = default;
  explicit Position(size_t p) : p0(p % 9), p1(p / 9) {}
  explicit operator size_t() const { return p0 + p1 * 9; }
  Position &operator=(const Position &) = default;
  Position &operator=(Position &&) noexcept = default;
  ~Position() = default;

  friend std::istream &operator>>(std::istream &in, Position &p) {
    std::string ipos;
    in >> ipos;
    // assert(ipos.size() == 2);
    auto pp0 = static_cast<size_t>(tolower(ipos[0]) - 'a');
    p.p0 = pp0 > 8 ? pp0 - 1 : pp0;
    p.p1 = static_cast<size_t>(8 - (ipos[1] - '1'));
    return in;
  }
  friend std::ostream &operator<<(std::ostream &out, const Position &p) {
    out << static_cast<char>((p.p0 >= 8 ? 1 : 0) + p.p0 + 'A')
        << static_cast<char>((8 - p.p1) + '1');
    return out;
  }

  size_t p0, p1;
};

namespace detail {
constexpr uint32_t fnv1a_32(const char *s, size_t count) {
  // FNV-1a 32bit hashing algorithm.
  return ((count > 0u ? fnv1a_32(s, count - 1) : 2166136261u) ^
          static_cast<uint32_t>(s[count])) *
         16777619u;
}
} // namespace detail

constexpr uint32_t operator"" _hash(const char *s, size_t count) {
  return detail::fnv1a_32(s, count);
}

class GTPHelper {
public:
  static GTPHelper &getInstance() {
    static GTPHelper instance;
    return instance;
  }
  GTPHelper() {
    std::ios::sync_with_stdio(false);
    history_.reserve(81);
  }
  ~GTPHelper() = default;
  GTPHelper(GTPHelper const &) = delete;
  GTPHelper(GTPHelper &&) = delete;
  GTPHelper &operator=(GTPHelper const &) = delete;
  GTPHelper &operator=(GTPHelper &&) = delete;

public:
  bool execute() {
    std::string cmd;
    std::cin >> cmd;
    const auto cmd_hash = detail::fnv1a_32(cmd.c_str(), cmd.size());
    switch (cmd_hash) {
    // Adminstrative Commands
    case "quit"_hash:
      std::cout << "=\n\n";
      return false;
    case "protocol_version"_hash:
      protocol_version();
      break;
    case "name"_hash:
      name();
      break;
    case "version"_hash:
      version();
      break;
    case "known_command"_hash:
      known_command();
      break;
    case "list_commands"_hash:
      list_commands();
      break;
    // Setup Commands
    case "boardsize"_hash:
      boardsize();
      break;
    case "clear_board"_hash:
      clear_board();
      break;
    case "komi"_hash:
      komi();
      break;
    case "play"_hash:
      play();
      break;
    case "genmove"_hash:
      genmove();
      break;
    case "undo"_hash:
      undo();
      break;
    // Tournament Commands
    case "final_score"_hash:
      final_score();
      break;
    // Debug Commands
    case "showboard"_hash:
      showboard();
      break;
    // GoGui Commands
    /*case "gogui-rules_game_id"_hash:
      gogui_rules_game_id();
      break;
    case "gogui-rules_board"_hash:
      gogui_rules_board();
      break;
    case "gogui-rules_board_size"_hash:
      gogui_rules_board_size();
      break;
    case "gogui-rules_legal_moves"_hash:
      gogui_rules_legal_moves();
      break;
    case "gogui-rules_side_to_move"_hash:
      gogui_rules_side_to_move();
      break;
    case "gogui-rules_final_result"_hash:
      gogui_rules_final_result();
      break;*/
    default:
      std::getline(std::cin, cmd);
      std::cout << "? unknown command\n\n";
    }
    return true;
  }

  void registerAgent(const std::string &model_file) {
    agent_ = std::make_unique<MCTSAgent>();
    agent_->load_model(model_file);
  }

private:
  /* Adminstrative Commands */
  void protocol_version() const { std::cout << "= 2\n\n"; }
  void name() const { std::cout << "= GuaGua\n\n"; }
  void version() const { std::cout << "= 1.0\n\n"; }
  void known_command() const {
    std::string cmd;
    std::cin >> cmd;
    if (std::find(std::begin(all_commands_), std::end(all_commands_), cmd) !=
        std::end(all_commands_)) {
      std::cout << "= true\n\n";
    } else {
      std::cout << "= false\n\n";
    }
  }
  void list_commands() const {
    std::cout << "=\n";
    std::copy(std::begin(all_commands_), std::end(all_commands_),
              std::ostream_iterator<std::string>(std::cout, "\n"));
    std::cout << "\n";
  }

private:
  /* Setup Commands */
  void boardsize() const {
    int size;
    std::cin >> size;
    if (size == 9) {
      std::cout << "=\n\n";
    } else {
      std::cout << "? unacceptable size\n\n";
    }
  }
  void clear_board() {
    Board b;
    std::swap(board_, b);
    history_.clear();
    gogui_turns_ = true;
    std::cout << "=\n\n";
  }
  void komi() const {
    // not used
    float komi;
    std::cin >> komi;
    std::cout << "=\n\n";
  }

private:
  /* Core Play Commands */
  void play() {
    std::string sbw;
    Position pos;
    std::cin >> sbw >> pos;
    auto bw = static_cast<size_t>(tolower(sbw[0]) == 'w');
    if (board_.place(bw, static_cast<size_t>(pos))) {
      std::cout << "=\n\n";
      history_.push_back(board_);
      gogui_turns_ = !gogui_turns_;
    } else {
      std::cout << "? illegal move\n\n";
    }
  }
  void genmove() {
    std::string sbw;
    std::cin >> sbw;
    auto bw = static_cast<size_t>(tolower(sbw[0]) == 'w');
    auto move = agent_->take_action(board_, bw);
    if (move < 81) {
      std::cout << "= " << Position(move) << "\n\n";
      board_.place(bw, move);
      history_.push_back(board_);
    } else {
      std::cout << "= resign\n\n";
    }
  }
  void undo() {
    if (history_.empty()) {
      std::cout << "? cannot undo\n\n";
      return;
    }
    board_ = history_.back();
    history_.pop_back();
    std::cout << "=\n\n";
  }

private:
  /* Tournament Commands */
  // void time_settings() { ; }
  void final_score() const {
    std::cout << "= " << (!gogui_turns_ ? "B" : "W") << "+1\n\n";
  }

private:
  /* Regression Commands */
  // void loadsgf() { ; }
  // void reg_genmove() { ; }

private:
  /* Debug Commands */
  void showboard() const { std::cout << "=\n" << board_; }

private:
  /* GoGui Rules */
  // void gogui_analyze_command() const { ; }
  void gogui_rules_game_id() const { std::cout << "= Nogo\n\n"; }
  void gogui_rules_board() const { std::cout << "=\n" << board_; }
  void gogui_rules_board_size() const { std::cout << "= 9\n\n"; }
  void gogui_rules_legal_moves() const {
    size_t bw = gogui_turns_ ? 0 : 1;
    std::cout << "=";
    auto moves = board_.get_legal_moves(bw);
    for (size_t p = moves._Find_first(); p != 81; p = moves._Find_next(p)) {
      std::cout << " " << Position(p);
    }
    std::cout << "\n\n";
  }
  void gogui_rules_side_to_move() {
    std::cout << "= " << (gogui_turns_ ? "black" : "white") << "\n\n";
  }
  void gogui_rules_final_result() const {
    std::cout << "= " << (!gogui_turns_ ? "Black" : "White") << " wins.\n\n";
  }

private:
  std::unique_ptr<MCTSAgent> agent_;
  Board board_;
  std::vector<Board> history_;
  bool gogui_turns_ = true;
  static const constexpr std::array<const char *, 14> all_commands_ = {
      // Adminstrative Commands
      "quit", "protocol_version", "name", "version", "known_command",
      "list_commands",
      // Core Play Commands
      "genmove", "play", "undo",
      // Setup Commands
      "boardsize", "clear_board", "komi",
      // Tournament Commands
      "final_score",
      // Debug Commands
      "showboard",
      // GoGui Commands
      /*"gogui-rules_game_id", "gogui-rules_board", "gogui-rules_board_size",
      "gogui-rules_legal_moves", "gogui-rules_side_to_move",
      "gogui-rules_final_result"*/};
};