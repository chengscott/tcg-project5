#pragma once
#include "board.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <torch/script.h>
#include <unordered_map>

#ifdef SELF_PLAY
constexpr const bool g_SELF_PLAY = true;
constexpr const size_t g_TOTAL_COUNTS = 400;
#else
constexpr const bool g_SELF_PLAY = false;
constexpr const size_t g_TOTAL_COUNTS = 87;
#endif

#ifdef SHOW_INFO
constexpr const bool g_SHOW_INFO = true;
#else
constexpr const bool g_SHOW_INFO = false;
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
        const float score = child.q_value_ + 1.5f * prob_[child.pos_] *
                                                 sqrt_visits_ /
                                                 (1 + child.visits_);
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
      for (size_t i = 0, pos = moves._Find_first(); i < size;
           ++i, pos = moves._Find_next(pos)) {
        children_[i].init(1 - bw_, pos, this);
      }
      // forward children
      const auto &lsize = static_cast<long>(size);
      torch::Tensor inputs = torch::empty({lsize, 4, 9, 9}).to(torch::kCUDA);
      for (long i = 0; i < lsize; ++i) {
        const auto &child = children_[static_cast<size_t>(i)];
        Board board(b);
        board.place(child.bw_, child.pos_);
        inputs[i] =
            torch::from_blob(board.get_features(), {4, 9, 9}).to(torch::kCUDA);
      }
      const auto &outputs = net.forward({inputs}).toTuple();
      // outputs viewer
      const auto &p_tensor =
          torch::softmax(outputs->elements()[0].toTensor(), 1).to(torch::kCPU);
      const auto &p_view = p_tensor.accessor<float, 2>();
      const auto &v_tensor = outputs->elements()[1].toTensor().to(torch::kCPU);
      const auto &v_view = v_tensor.accessor<float, 2>();
      // store (p[], v)
      for (long i = 0; i < lsize; ++i) {
        auto &child = children_[static_cast<size_t>(i)];
        for (long j = 0; j < 81; ++j) {
          child.prob_[j] = p_view[i][j];
        }
        child.z_value_ = v_view[i][0];
      }
      return true;
    }
    template <class PRNG>
    bool expand_root(const Board &b, PRNG &rng,
                     torch::jit::script::Module &net) noexcept {
      // forward root
      Board board(b);
      torch::Tensor inputs =
          torch::from_blob(board.get_features(), {1, 4, 9, 9}).to(torch::kCUDA);
      const auto &outputs = net.forward({inputs}).toTuple();
      const auto &p_tensor =
          torch::softmax(outputs->elements()[0].toTensor(), 1).to(torch::kCPU);
      const auto &p_view = p_tensor.accessor<float, 2>();
      const auto &v_tensor = outputs->elements()[1].toTensor().to(torch::kCPU);
      const auto &v_view = v_tensor.accessor<float, 2>();
      // store (p[], v)
      for (long i = 0; i < 81; ++i) {
        prob_[i] = p_view[0][i];
      }
      z_value_ = v_view[0][0];
      // expand child
      expand(b, net);
      if constexpr (g_SELF_PLAY) {
        // root add dirichlet
        add_dirichlet(children_size_, rng);
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
      if constexpr (g_SELF_PLAY) {
        std::cerr << "==========DIST=BEGIN==========" << std::endl;
      }
      for (size_t i = 0; i < children_size_; ++i) {
        const auto &child = children_[i];
        if (child.visits_ > 0) {
          visits.emplace(child.pos_, child.visits_);
          // show self-play info
          if constexpr (g_SELF_PLAY || g_SHOW_INFO) {
            size_t p0 = child.pos_ % 9, p1 = child.pos_ / 9;
            auto cp0 = static_cast<char>((p0 >= 8 ? 1 : 0) + p0 + 'A'),
                 cp1 = static_cast<char>((8 - p1) + '1');
            std::cerr << cp0 << cp1;
          }
          if constexpr (g_SELF_PLAY) {
            std::cerr << ' ' << child.pos_ << ' ' << child.visits_;
          }
          if constexpr (g_SHOW_INFO) {
            std::cerr << ' ' << child.z_value_ << ' ' << child.q_value_;
          }
          if constexpr (g_SELF_PLAY || g_SHOW_INFO) {
            std::cerr << std::endl;
          }
        }
      }
      if constexpr (g_SELF_PLAY) {
        std::cerr << "==========DIST=END==========" << std::endl << std::endl;
      }
      if constexpr (g_SHOW_INFO) {
        std::cerr << "before: " << z_value_ << std::endl
                  << "after:  " << q_value_ << std::endl;
      }
    }

  private:
    inline constexpr void init(size_t bw, size_t pos, Node *parent) noexcept {
      bw_ = bw;
      pos_ = pos;
      parent_ = parent;
    }

    template <class PRNG> void add_dirichlet(size_t size, PRNG &rng) {
      float noise[81];
      std::gamma_distribution<float> gamma(.03f);
      float sum = 0.f;
      for (size_t i = 0; i < size; ++i) {
        noise[i] = gamma(rng);
        sum += noise[i];
      }
      const constexpr float eps = .25f;
      for (size_t i = 0; i < size; ++i) {
        prob_[i] = (1 - eps) * prob_[i] + eps * (noise[i] / sum);
      }
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
  void load_model(const std::string &model_file) {
    std::cerr << "> Load model from " << model_file << std::endl;
    net_ = torch::jit::load(model_file);
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
    root.expand_root(b, engine_, net_);
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
    } while (++total_counts < g_TOTAL_COUNTS);
    // (hclock::now() - start_time) < threshold_time);
    if constexpr (g_SHOW_INFO) {
      const auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(hclock::now() -
                                                                start_time)
              .count();
      std::cerr << duration << " ms" << std::endl
                << total_counts << " simulations" << std::endl;
    }

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
  std::random_device seed_{};
  std::default_random_engine engine_{seed_()};
};