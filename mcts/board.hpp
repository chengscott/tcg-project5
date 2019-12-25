#pragma once
#include "dset.hpp"
#include <array>
#include <bitset>
#include <iostream>
#include <random>
#include <vector>
#define BIT_TEST _Unchecked_test

const auto MASK_RIGHT = ~std::bitset<81>("100000000"
                                         "100000000"
                                         "100000000"
                                         "100000000"
                                         "100000000"
                                         "100000000"
                                         "100000000"
                                         "100000000"
                                         "100000000"),
           MASK_LEFT = MASK_RIGHT << 1;

class Board {
public:
  using board_t = std::bitset<81>;

  constexpr size_t operator[](size_t p) const noexcept {
    return static_cast<size_t>(board_[0][p]) +
           static_cast<size_t>(board_[1][p]) * 2u;
  }
  bool place(size_t bw, size_t p) noexcept {
    // assert(bw == 0 || bw == 1);
    if (forbid_[bw].BIT_TEST(p)) {
      return false;
    }
    bw_ = bw;
    // place
    auto &forbid = forbid_[bw], &forbid_op = forbid_[1 - bw];
    auto &board = board_[bw].set(p), &board_op = board_[1 - bw];
    forbid.set(p);
    forbid_op.set(p);
    // union
    auto &dset = dset_[bw], &dset_op = dset_[1 - bw];
    const size_t dirlen = dir_len_[p];
    for (size_t i = 0; i < dirlen; ++i) {
      const size_t x = dir_[p][i];
      if (board.BIT_TEST(x)) {
        dset.unions(p, x);
      }
    }
    // check current component
    check_valid(dset, p, board, board_op, forbid, forbid_op);
    // check neighbors
    for (size_t i = 0; i < dirlen; ++i) {
      const size_t x = dir_[p][i];
      if (board_op.BIT_TEST(x)) {
        check_valid(dset_op, x, board_op, board, forbid_op, forbid);
      } else if (!board.BIT_TEST(x)) {
        check_no_liberty(dset_op, x, board_op, board, forbid_op);
      }
    }
    return true;
  }

  bool has_legal_move(size_t bw) const noexcept { return !forbid_[bw].all(); }

  board_t get_legal_moves(size_t bw) const noexcept { return ~forbid_[bw]; }

  float *get_features() noexcept {
    const auto &bw = bw_;
    features_.resize(9 * 9 * 4);
    fill(board_[bw], 0);
    fill(~forbid_[1 - bw], 1);
    fill(~forbid_[bw], 2);
    fill(board_[1 - bw], 3);
    return features_.data();
  }

  template <class PRNG>
  inline size_t random_legal_move(size_t bw, PRNG &rng) const noexcept {
    return random_move_from_board(~forbid_[bw], rng);
  }

  template <class PRNG>
  static size_t random_move_from_board(const board_t &valid,
                                       PRNG &rng) noexcept {
    // assert(valid.any());
    const size_t index = rng() % valid.count();
    size_t action = valid._Find_first();
    for (size_t i = 0; i < index; ++i) {
      action = valid._Find_next(action);
    }
    return action;
  }

private:
  void fill(const board_t &board, size_t base) noexcept {
    for (size_t i = 0; i < 9; ++i) {
      for (size_t j = 0; j < 9; ++j) {
        features_[base * 81 + i * 9 + j] =
            static_cast<float>(board.test(i * 9 + j));
      }
    }
  }

public:
  friend std::ostream &operator<<(std::ostream &out, const Board &b) {
    out << "  A B C D E F G H J\n";
    for (size_t i = 0; i < 9; ++i) {
      out << 9 - i << ' ';
      for (size_t j = 0; j < 9; ++j) {
        out << ".XO"[b[i * 9 + j]] << ' ';
      }
      out << 9 - i << '\n';
    }
    out << "  A B C D E F G H J\n\n";
    return out;
  }

private:
  static void check_valid(const DisjointSet &dset, size_t p,
                          const board_t &board, const board_t &board_op,
                          board_t &forbid, board_t &forbid_op) noexcept {
    // find component
    const auto &component = dset.get_component(p);
    // find liberty
    const auto &liberty =
        ((component << 9) | (component >> 9) | ((component & MASK_RIGHT) << 1) |
         ((component & MASK_LEFT) >> 1)) &
        ~(board | board_op);
    if (liberty.count() == 1) {
      const size_t x = liberty._Find_first();
      forbid_op.set(x);
      check_no_liberty(dset, x, board, board_op, forbid);
    }
  }

  static void check_no_liberty(const DisjointSet &dset, size_t x, board_t board,
                               const board_t &board_op,
                               board_t &forbid) noexcept {
    board.set(x);
    // find component
    auto component = dset.get_component(x);
    const size_t dirlen = dir_len_[x];
    for (size_t i = 0; i < dirlen; ++i) {
      const size_t y = dir_[x][i];
      if (board.BIT_TEST(y)) {
        component |= dset.get_component(y);
      }
    }
    // find liberty
    const auto &liberty =
        ((component << 9) | (component >> 9) | ((component & MASK_RIGHT) << 1) |
         ((component & MASK_LEFT) >> 1)) &
        ~(board | board_op);
    if (liberty.none()) {
      forbid.set(x);
    }
    // board.reset(x);
  }

private:
  size_t bw_ = 0;
  std::vector<float> features_;
  board_t board_[2], forbid_[2];
  DisjointSet dset_[2];
  const static constexpr auto dir_ = []() constexpr {
    std::array<std::array<size_t, 4>, 81> ret{};
    for (size_t i = 0; i < 81; ++i) {
      auto &ri = ret[i];
      size_t j = 0;
      if (i / 9 > 0) {
        ri[j++] = i - 9;
      }
      if (i % 9 > 0) {
        ri[j++] = i - 1;
      }
      if (i % 9 < 8) {
        ri[j++] = i + 1;
      }
      if (i / 9 < 8) {
        ri[j++] = i + 9;
      }
    }
    return ret;
  }
  ();
  const static constexpr auto dir_len_ = []() constexpr {
    std::array<size_t, 81> ret{};
    for (size_t i = 0; i < 81; ++i) {
      ret[i] = static_cast<size_t>(i / 9 > 0) + static_cast<size_t>(i % 9 > 0) +
               static_cast<size_t>(i % 9 < 8) + static_cast<size_t>(i / 9 < 8);
    }
    return ret;
  }
  ();
};