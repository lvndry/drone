// Copyright 2026 Drone contributors
// SPDX-License-Identifier: MIT
// Standalone reference wrapper; uses unmodified libkeyfinder defaults.
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "audiodata.h"
#include "constants.h"
#include "keyfinder.h"

namespace {
// Explicit named-enum mapping, independent of enum ordering assumptions.
constexpr int drone_class(int raw) {
  switch (raw) {
    case KeyFinder::C_MAJOR: return 0;
    case KeyFinder::C_MINOR: return 1;
    case KeyFinder::D_FLAT_MAJOR: return 2;
    case KeyFinder::D_FLAT_MINOR: return 3;
    case KeyFinder::D_MAJOR: return 4;
    case KeyFinder::D_MINOR: return 5;
    case KeyFinder::E_FLAT_MAJOR: return 6;
    case KeyFinder::E_FLAT_MINOR: return 7;
    case KeyFinder::E_MAJOR: return 8;
    case KeyFinder::E_MINOR: return 9;
    case KeyFinder::F_MAJOR: return 10;
    case KeyFinder::F_MINOR: return 11;
    case KeyFinder::G_FLAT_MAJOR: return 12;
    case KeyFinder::G_FLAT_MINOR: return 13;
    case KeyFinder::G_MAJOR: return 14;
    case KeyFinder::G_MINOR: return 15;
    case KeyFinder::A_FLAT_MAJOR: return 16;
    case KeyFinder::A_FLAT_MINOR: return 17;
    case KeyFinder::A_MAJOR: return 18;
    case KeyFinder::A_MINOR: return 19;
    case KeyFinder::B_FLAT_MAJOR: return 20;
    case KeyFinder::B_FLAT_MINOR: return 21;
    case KeyFinder::B_MAJOR: return 22;
    case KeyFinder::B_MINOR: return 23;
    case KeyFinder::SILENCE: return -1;
    default: return -2;
  }
}
// Independently written golden output for each raw enum value in pinned 2.2.8.
constexpr std::array<int, 25> golden = {
  18,19,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,-1
};
constexpr bool enum_contract() {
  for (int raw = 0; raw < 25; ++raw)
    if (drone_class(raw) != golden[raw]) return false;
  return drone_class(-1) == -2 && drone_class(25) == -2;
}
static_assert(enum_contract(), "libkeyfinder 2.2.8 enum contract changed");
constexpr const char* tonics[] = {
  "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};
void result(int raw) {
  int code = drone_class(raw);
  if (code == -2) throw std::runtime_error("unknown libkeyfinder enum");
  std::cout << "{\"schema\":\"drone.keyfinder-reference.v1\",\"raw_enum\":" << raw;
  if (code == -1) {
    std::cout << ",\"status\":\"silence\",\"tonic\":null,\"mode\":null,\"drone_class\":null}\n";
  } else {
    std::cout << ",\"status\":\"key\",\"tonic\":\"" << tonics[code / 2]
              << "\",\"mode\":\"" << (code % 2 ? "minor" : "major")
              << "\",\"drone_class\":" << code << "}\n";
  }
}
uint32_t u32(const unsigned char* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint16_t u16(const unsigned char* p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
void read(std::ifstream& in, unsigned char* data, std::streamsize n) {
  if (!in.read(reinterpret_cast<char*>(data), n)) throw std::runtime_error("truncated WAV");
}
KeyFinder::key_t classify(const char* path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open WAV");
  const auto end = in.tellg();
  if (end < 12 || uint64_t(end) > uint64_t(UINT32_MAX) + 8)
    throw std::runtime_error("invalid RIFF size");
  in.seekg(0);
  unsigned char header[12]; read(in, header, 12);
  if (std::string(reinterpret_cast<char*>(header),4) != "RIFF" ||
      std::string(reinterpret_cast<char*>(header+8),4) != "WAVE" ||
      uint64_t(u32(header+4)) + 8 != uint64_t(end))
    throw std::runtime_error("invalid RIFF/WAVE container");
  bool have_fmt = false, have_data = false;
  uint32_t rate = 0, data_size = 0;
  std::streampos data_start = 0;
  while (in.tellg() < end) {
    const uint64_t pos = uint64_t(in.tellg());
    if (uint64_t(end) - pos < 8) throw std::runtime_error("truncated chunk header");
    unsigned char chunk[8]; read(in, chunk, 8);
    const uint32_t size = u32(chunk+4);
    const uint64_t next = pos + 8 + uint64_t(size) + (size & 1);
    if (next > uint64_t(end)) throw std::runtime_error("chunk exceeds RIFF");
    std::string id(reinterpret_cast<char*>(chunk),4);
    if (id == "fmt ") {
      if (have_fmt || size < 16) throw std::runtime_error("invalid fmt chunk");
      unsigned char fmt[16]; read(in, fmt, 16);
      rate = u32(fmt+4);
      if (u16(fmt) != 1 || u16(fmt+2) != 1 || !rate || u16(fmt+14) != 16 ||
          u16(fmt+12) != 2 || uint64_t(u32(fmt+8)) != uint64_t(rate)*2)
        throw std::runtime_error("require mono PCM16 with consistent rate/alignment");
      have_fmt = true;
    } else if (id == "data") {
      if (have_data || size == 0 || size % 2) throw std::runtime_error("invalid data chunk");
      data_start = in.tellg(); data_size = size; have_data = true;
    }
    in.seekg(static_cast<std::streamoff>(next));
    if (!in) throw std::runtime_error("invalid chunk offset");
  }
  if (!have_fmt || !have_data) throw std::runtime_error("missing fmt/data chunk");
  KeyFinder::AudioData audio;
  audio.setFrameRate(rate); audio.setChannels(1); audio.addToSampleCount(data_size/2);
  in.seekg(data_start);
  for (uint32_t i=0; i<data_size/2; ++i) {
    unsigned char sample[2]; read(in, sample, 2);
    const int value = int(u16(sample));
    audio.setSample(i, (value >= 32768 ? value - 65536 : value) / 32768.0);
  }
  KeyFinder::KeyFinder finder;
  return finder.keyOfAudio(audio);
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: keyfinder_reference WAV | --self-test");
    if (std::string(argv[1]) == "--self-test") {
      if (!enum_contract()) throw std::runtime_error("enum contract failed");
      for (int raw=0; raw<25; ++raw) result(raw);
      return 0;
    }
    if (argv[1][0] == '-') throw std::runtime_error("unknown option");
    result(static_cast<int>(classify(argv[1])));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "keyfinder_reference: " << error.what() << '\n';
    return 1;
  }
}
