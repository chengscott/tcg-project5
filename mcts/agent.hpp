#pragma once
#include "board.hpp"
#include "random.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <torch/script.h>
#include <unordered_map>

#ifndef MODEL_FILE
#define MODEL_FILE ("model.pt")
#endif

class MCTSAgent {
private:
  class Node {
  public:
    constexpr void init_bw(size_t bw) noexcept { bw_ = bw; }
    constexpr Node *get_parent() const noexcept { return parent_; };
    constexpr float get_z_value() const noexcept { return z_value_; };
    constexpr bool has_children() const noexcept { return children_size_ > 0; }
    template <class PRNG>
    Node *select_child(PRNG &rng, size_t &bw, size_t &pos) {
      float max_score = -1;
      for (size_t i = 0; i < children_size_; ++i) {
        auto &child = children_[i];
        const float score =
            child.q_value_ +
            1.5f * prob_[child.pos_] * sqrt_visits_ / (1 + child.visits_);
        child.uct_score_ = score;
        max_score = std::max(score, max_score);
      }
      Board::board_t max_children{};
      for (size_t i = 0; i < children_size_; ++i) {
        if (std::abs(children_[i].uct_score_ - max_score) < .0001f) {
          max_children.set(i);
        }
      }
      size_t idx = Board::random_move_from_board(max_children, rng);
      auto &child = children_[idx];
      bw = child.bw_;
      pos = child.pos_;
      return &child;
    }
    bool expand(const Board &b, torch::jit::script::Module &net) noexcept {
      auto moves(b.get_legal_moves(1 - bw_));
      const size_t size = moves.count();
      if (size == 0) {
        return false;
      }
      // expand children
      children_size_ = size;
      children_ = std::make_unique<Node[]>(size);
      torch::Tensor inputs =
          torch::empty({long(size), 4, 9, 9}).to(torch::kCUDA);
      for (size_t i = 0, pos = moves._Find_first(); i < size;
           ++i, pos = moves._Find_next(pos)) {
        children_[i].init(1 - bw_, pos, this);
        Board board(b);
        board.place(1 - bw_, pos);
        inputs[i] =
            torch::from_blob(board.get_features(), {4, 9, 9}).to(torch::kCUDA);
      }
      // forward children
      const auto &outputs = net.forward({inputs}).toTuple();
      const auto &p_tensor =
          torch::softmax(outputs->elements()[0].toTensor(), 1).to(torch::kCPU);
      const auto &p_view = p_tensor.accessor<float, 2>();
      const auto &v_tensor = outputs->elements()[1].toTensor().to(torch::kCPU);
      const auto &v_view = v_tensor.accessor<float, 2>();
      // store (p[], v)
      for (size_t i = 0; i < size; ++i) {
        auto &child = children_[i];
        for (size_t j = 0; j < 81; ++j) {
          child.prob_[j] = p_view[i][j];
        }
        child.z_value_ = v_view[i][0];
      }
      return true;
    }
    void update(float z) noexcept {
      ++visits_;
      sqrt_visits_ = std::sqrt(visits_);
      q_value_ += (z - q_value_) / visits_;
    }
    void get_children_visits(std::unordered_map<size_t, size_t> &visits) const
        noexcept {
      for (size_t i = 0; i < children_size_; ++i) {
        const auto &child = children_[i];
        visits.emplace(child.pos_, child.visits_);
      }
    }

  private:
    inline constexpr void init(size_t bw, size_t pos, Node *parent) noexcept {
      bw_ = bw;
      pos_ = pos;
      parent_ = parent;
    }

  private:
    size_t children_size_ = 0;
    std::unique_ptr<Node[]> children_;
    size_t bw_, pos_ = 81;
    Node *parent_ = nullptr;

  private:
    size_t visits_ = 0;
    float q_value_ = 0.f, sqrt_visits_ = 0.f, uct_score_;
    float z_value_ = -2.f, prob_[81];
  };

public:
  void load_model() {
    net_ = torch::jit::load(MODEL_FILE);
    net_.to(torch::kCUDA);
    std::cerr << "> Load model done." << std::endl;
  }

public:
  using hclock = std::chrono::high_resolution_clock;
  const static constexpr auto threshold_time = std::chrono::seconds(1);

  size_t take_action(const Board &b, size_t bw) {
    if (!b.has_legal_move(bw)) {
      return 81;
    }
    size_t total_counts = 0, cbw = 1 - bw, cpos = 81;
    const auto start_time = hclock::now();
    Node root;
    root.init_bw(1 - bw);
    do {
      Node *node = &root;
      Board board(b);
      // selection
      while (node->has_children()) {
        node = node->select_child(engine_, cbw, cpos);
        board.place(cbw, cpos);
      }
      // expansion
      if (node->expand(board, net_)) {
        node = node->select_child(engine_, cbw, cpos);
        board.place(cbw, cpos);
      }
      // rollout
      float z = node->get_z_value();
      // backpropogation
      while (node != nullptr) {
        node->update(z);
        node = node->get_parent();
        z = -z;
      }
    } while (++total_counts < 87);
    // (hclock::now() - start_time) < threshold_time);
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                              hclock::now() - start_time)
                              .count();
    std::cerr << duration << " ms" << std::endl
              << total_counts << " simulations" << std::endl;

    std::unordered_map<size_t, size_t> visits;
    root.get_children_visits(visits);
    size_t best_move = std::max_element(std::begin(visits), std::end(visits),
                                        [](const auto &p1, const auto &p2) {
                                          return p1.second < p2.second;
                                        })
                           ->first;

    return best_move;
  }

private:
  torch::jit::script::Module net_;
  splitmix seed_{};
  xorshift engine_{seed_()};
};