#include "agent.hpp"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model path> <save sgf path> <games>"
              << std::endl;
    exit(1);
  }
  auto model_path = std::string(argv[1]);
  auto sgf_path = std::string(argv[2]);
  auto self_play_games = std::stoul(argv[3]);

  std::ofstream fout(sgf_path, std::ofstream::app);

  MCTSAgent agent;
  agent.load_model(model_path);

  const std::string sgf_header = "(;FF[4]CA[UTF-8]AP[atuno_evaluator]SZ[9]KM[7."
                                 "5]PB[GuaGua:1.0]PW[GuaGua:1.0]DT[";

  using clock_ = std::chrono::system_clock;
  const auto start_time = clock_::now();

  for (size_t i = 1; i <= self_play_games; ++i) {
    Board board;
    size_t j = 0;
    const auto start_time_loop = clock_::now();
    auto time = clock_::to_time_t(start_time_loop);
    std::string time_string = std::ctime(&time);

    std::string sgf_string = sgf_header +
                             time_string.substr(0, time_string.length() - 1) +
                             "]RE[",
                game_progress;
    do {
      size_t action = agent.take_action(board, j % 2);
      board.place(j % 2, action);

      // Add move
      game_progress += {"BW"[j % 2]};
      game_progress += "[";
      game_progress += {static_cast<char>((action % 9) + 'a'),
                        static_cast<char>((action / 9) + 'a')};
      game_progress += "]";
      // Add policy distribution
      auto policy = agent.get_policy();
      game_progress += "C[";
      for (auto &&[move, visits] : policy) {
        game_progress +=
            std::to_string(move) + ":" + std::to_string(visits) + ",";
      }
      game_progress += "];";
    } while (board.has_legal_move(++j % 2));

    const auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                              clock_::now() - start_time_loop)
                              .count();
    const auto total_duration =
        std::chrono::duration_cast<std::chrono::seconds>(clock_::now() -
                                                         start_time)
            .count();

    std::string result = (j % 2 == 0) ? "W+R" : "B+R";
    sgf_string.append(result)
        .append("]C[time used: ")
        .append(std::to_string(duration))
        .append(" sec];")
        .append(game_progress)
        .append(")");

    fout << sgf_string << std::endl;
    std::cerr << "Finish game " << i << " / " << self_play_games
              << ", time used: " << duration << " (sec), estimate time left: ";
    if ((double)total_duration < 3600) {
      std::cerr << std::setprecision(3)
                << (double)total_duration * (self_play_games - i) / i / 60
                << " (min)" << std::endl;
    } else {
      std::cerr << std::setprecision(3)
                << (double)total_duration * (self_play_games - i) / i / 3600
                << " (hour)" << std::endl;
    }
  }
  fout.close();
  return 0;
}