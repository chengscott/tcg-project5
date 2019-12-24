#pragma once
#include <array>
#include <bitset>

const auto COMPONENT_INIT = []() {
  std::array<std::bitset<81>, 81> ret{};
  for (size_t i = 0; i < 81; ++i) {
    ret[i].set(i);
  }
  return ret;
}();

struct DisjointSet {
  size_t find(size_t x) { return x == dset[x] ? x : (dset[x] = find(dset[x])); }
  void unions(size_t x, size_t y) {
    size_t fx = find(x), fy = find(y);
    dset[fx] = fy;
    component[fy] |= component[fx];
  }

  size_t cfind(size_t x) const { return x == dset[x] ? x : cfind(dset[x]); }
  const std::bitset<81> &get_component(size_t x) const {
    return component[cfind(x)];
  }

  std::array<size_t, 81> dset = []() constexpr {
    std::array<size_t, 81> ret{};
    for (size_t i = 0; i < 81; ++i) {
      ret[i] = i;
    }
    return ret;
  }
  ();
  std::array<std::bitset<81>, 81> component = COMPONENT_INIT;
};