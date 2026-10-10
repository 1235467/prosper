// test_refused_shader_inputs (#4808) -- a refused compute program keeps the inputs it was refused
// with, and an offline replay on them reaches the live decision.
//
// WHAT EACH ARM KILLS:
//   RoundTripKeepsEveryInput      a config field or table field the sidecar drops (the replay then
//                                 silently defaults it), or buffer bytes not kept under the cap
//   PayloadAboveCapIsOmitted      an unbounded sidecar: bytes above the cap must become opaque
//   ReplayReproducesLiveRefusal   a sidecar that is written but not what the live compile used;
//                                 the replay must name the live reject, not the table-less one,
//                                 and over the live recompile span, not the whole dumped window
//   TablelessFallbackDiffers      the mutation of replaying without the recorded inputs: on this
//                                 fixture it must give a different answer, or the test above
//                                 could pass with the inputs ignored
//   LaterInputSetsAreCounted      a counter that never moves, or one that counts repeats
//   DumpWithoutInputsStillWorks   a pre-#4808 dump (no sidecar) treated as an error
//   CorruptSidecarIsRefused       a truncated or edited file parsed as inputs
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/execute/compute_program_facts.hpp"
#include "gpu/execute/refused_shader_source.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace fs = std::filesystem;

namespace {

fs::path fresh_root(const char* name) {
    const fs::path root = prosper_test::test_scratch_dir() / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    reset_refused_shader_dump_for_test(root.string());
    reset_compute_program_facts_for_test();
    return root;
}

// v_* setup, buffer_store_dwordx4 through the V# in s[0:3] at pc 6, then an instruction compute
// does not lower, then s_endpgm. With the live table the store resolves and the recompile stops at
// the later instruction; without one it stops at the store.
constexpr uint32_t kUnsupported = 0xc8000100u;   // v_interp_p1_f32: no compute lowering
std::vector<uint32_t> fixture_code() {
    // The three words after s_endpgm stand for the rest of the readable window a real dump keeps:
    // the live compile key holds only the recompile span, and the reject text records its length.
    return {0xd7460004u, 0x04010c08u, 0x7e000204u, 0x7e020205u,  0x7e040206u,
            0x7e060207u, 0xe01c2000u, 0x80000004u, kUnsupported, 0xbf810000u,
            0x11111111u, 0x22222222u, 0x33333333u};
}

struct LiveInputs {
    ShaderResourceTable table;
    ComputeShaderConfig config;
};

LiveInputs live_inputs(uint32_t pattern = 7) {
    LiveInputs in;
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.binding = 2;
    r.sgpr_base = 0;
    r.format = DataFormat::Uint32;
    r.num_components = 4;
    r.stride = 16;
    r.gpu_addr = 0x200000000ull;
    r.size = 4096;
    in.table.resources = {r};
    in.config.user_sgprs = {0, 2u | (16u << 16), 256, 0, pattern, 2, 3, 4};
    in.config.tgid_x_en = true;
    in.config.exact_thread_extent = true;
    in.config.threads_x = 256;
    in.config.threads_y = in.config.threads_z = 1;
    return in;
}

std::vector<uint32_t> read_words(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(f), {}};
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

fs::path only_file(const fs::path& dir, const char* extension) {
    fs::path found;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == extension) {
            EXPECT_TRUE(found.empty()) << "more than one " << extension;
            found = e.path();
        }
    return found;
}

// The live step: the executor's recompile of this program with these inputs, then its refusal.
std::string refuse_live(const std::vector<uint32_t>& code, uint64_t address, const LiveInputs& in,
                        bool with_inputs = true) {
    const RecompileDiagnosticContext diagnostic{RecompileDiagnosticStage::Compute, address};
    const auto spirv =
        recompile_compute(code.data(), rdna2_recompile_code_span(code.data(), code.size()),
                          &in.table, in.config, diagnostic);
    EXPECT_TRUE(spirv.empty()) << "the fixture must be refused live";
    const auto facts = compute_program_facts(code.data(), code.size(), diagnostic);
    note_refused_compute_program(facts, address, 4, 1, 1, with_inputs ? &in.table : nullptr,
                                 with_inputs ? &in.config : nullptr);
    return last_terminal_reject_reason(address);
}

}   // namespace

TEST(RefusedShaderInputs, RoundTripKeepsEveryInput) {
    const fs::path root = fresh_root("refused_inputs_roundtrip");
    LiveInputs in = live_inputs();
    ComputeShaderConfig& c = in.config;
    c.local_x = 8, c.local_y = 4, c.local_z = 2;
    c.threads_x = 11, c.threads_y = 12, c.threads_z = 13;
    c.wave_size = 32;
    c.compute_pgm_rsrc1 = 0x12345u;
    c.tidig_comp_cnt = 2;
    c.tgid_y_en = c.tgid_z_en = c.tg_size_en = true;
    c.lds_bytes = 4096;
    c.native_subgroup_size = 64;
    c.wave64_exchange_width = 32;
    c.force_exchange_dispatcher = true;
    c.native_storage_format_support = 0xabcdu;
    c.storage_buffer_int64_atomics = true;
    c.packed_r11_storage = false;
    c.float_transport.profile = FloatTransportProfile::ExplicitNonFinite32;
    auto bytes = std::make_shared<std::vector<uint8_t>>(64);
    for (size_t i = 0; i < bytes->size(); ++i) (*bytes)[i] = uint8_t(i * 3 + 1);
    in.table.owned_host_data.push_back(bytes);
    ShaderResource& r = in.table.resources[0];
    r.host_data = bytes->data();
    r.host_data_size = bytes->size();
    r.srt_offset = 0x40;
    r.fetch_pc = 6;
    const fs::path path = root / "x.inputs";
    ASSERT_TRUE(write_refused_compute_inputs(path, 0x1234, &in.table, c, "reject-tag pc=8"));

    const RefusedShaderInputs read = read_refused_shader_inputs(path);
    EXPECT_EQ(read.stage, "cs");
    EXPECT_EQ(read.program_address, 0x1234u);
    EXPECT_EQ(read.live_reject, "reject-tag pc=8");
    EXPECT_TRUE(read.incomplete.empty()) << read.incomplete;
    EXPECT_FALSE(read.payload_omitted);
    const ComputeShaderConfig& d = read.compute;
    EXPECT_EQ(d.user_sgprs, c.user_sgprs);
    EXPECT_EQ(d.local_x, 8u);
    EXPECT_EQ(d.local_y, 4u);
    EXPECT_EQ(d.local_z, 2u);
    EXPECT_EQ(d.exact_thread_extent, true);
    EXPECT_EQ(d.threads_x, 11u);
    EXPECT_EQ(d.threads_y, 12u);
    EXPECT_EQ(d.threads_z, 13u);
    EXPECT_EQ(d.wave_size, 32u);
    EXPECT_EQ(d.compute_pgm_rsrc1, 0x12345u);
    EXPECT_EQ(d.tidig_comp_cnt, 2u);
    EXPECT_TRUE(d.tgid_x_en && d.tgid_y_en && d.tgid_z_en && d.tg_size_en);
    EXPECT_EQ(d.lds_bytes, 4096u);
    EXPECT_EQ(d.native_subgroup_size, 64u);
    EXPECT_EQ(d.wave64_exchange_width, 32u);
    EXPECT_TRUE(d.force_exchange_dispatcher);
    EXPECT_EQ(d.native_storage_format_support, 0xabcdu);
    EXPECT_TRUE(d.storage_buffer_int64_atomics);
    EXPECT_FALSE(d.packed_r11_storage);
    EXPECT_EQ(d.float_transport.profile, FloatTransportProfile::ExplicitNonFinite32);
    ASSERT_TRUE(read.has_resources);
    ASSERT_EQ(read.resources.resources.size(), 1u);
    const ShaderResource& rr = read.resources.resources[0];
    EXPECT_EQ(rr.cls, ResourceClass::ConstantBuffer);
    EXPECT_EQ(rr.gpu_addr, r.gpu_addr);
    EXPECT_EQ(rr.size, r.size);
    EXPECT_EQ(rr.srt_offset, 0x40u);
    EXPECT_EQ(rr.fetch_pc, 6u);
    ASSERT_EQ(read.backing_owned.size(), 1u);
    EXPECT_TRUE(read.backing_present[0] && read.backing_owned[0]);
    ASSERT_NE(rr.host_data, nullptr);
    ASSERT_EQ(rr.host_data_size, bytes->size());
    EXPECT_EQ(std::vector<uint8_t>(rr.host_data, rr.host_data + rr.host_data_size), *bytes);
    EXPECT_NE(rr.host_data, bytes->data()) << "the replay must own its bytes";
}

TEST(RefusedShaderInputs, PayloadAboveCapIsOmitted) {
    const fs::path root = fresh_root("refused_inputs_cap");
    LiveInputs in = live_inputs();
    auto bytes = std::make_shared<std::vector<uint8_t>>(kRefusedShaderInputsMaxPayloadBytes + 16);
    in.table.owned_host_data.push_back(bytes);
    in.table.resources[0].host_data = bytes->data();
    in.table.resources[0].host_data_size = bytes->size();
    const fs::path path = root / "big.inputs";
    ASSERT_TRUE(write_refused_compute_inputs(path, 0x99, &in.table, in.config, ""));
    EXPECT_LT(fs::file_size(path), 64u * 1024u);
    const RefusedShaderInputs read = read_refused_shader_inputs(path);
    EXPECT_TRUE(read.payload_omitted);
    ASSERT_EQ(read.backing_present.size(), 1u);
    EXPECT_TRUE(read.backing_present[0]);
    EXPECT_FALSE(read.backing_owned[0]);
    EXPECT_EQ(read.resources.resources[0].host_data, nullptr);
}

TEST(RefusedShaderInputs, ReplayReproducesLiveRefusal) {
    fresh_root("refused_inputs_replay");  // resets the dump and facts state; the path is unused
    const auto code = fixture_code();
    const LiveInputs in = live_inputs();
    const std::string live = refuse_live(code, 0x5000, in);
    ASSERT_FALSE(live.empty());

    const fs::path dir = refused_shader_dump_directory();
    const fs::path bin = only_file(dir, ".bin");
    ASSERT_FALSE(bin.empty());
    const fs::path sidecar = refused_shader_inputs_path(bin);
    ASSERT_TRUE(fs::exists(sidecar)) << sidecar;
    const RefusedShaderInputs inputs = read_refused_shader_inputs(sidecar);
    EXPECT_EQ(inputs.live_reject, live);

    const RefusedShaderReplay replay = replay_refused_shader_inputs(read_words(bin), inputs);
    EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    EXPECT_FALSE(replay.produced);
    EXPECT_EQ(replay.reject, live);

    std::ifstream index(dir / "index.txt");
    const std::string text{std::istreambuf_iterator<char>(index), {}};
    EXPECT_NE(text.find(" inputs=" + sidecar.filename().string()), std::string::npos) << text;
}

TEST(RefusedShaderInputs, TablelessFallbackDiffers) {
    fresh_root("refused_inputs_fallback");
    const auto code = fixture_code();
    const LiveInputs in = live_inputs();
    const std::string live = refuse_live(code, 0x6000, in);
    const fs::path bin = only_file(refused_shader_dump_directory(), ".bin");
    const RefusedShaderInputs inputs = read_refused_shader_inputs(refused_shader_inputs_path(bin));
    const auto words = read_words(bin);
    const RefusedShaderReplay with = replay_refused_shader_inputs(words, inputs, true);
    const RefusedShaderReplay without = replay_refused_shader_inputs(words, inputs, false);
    EXPECT_FALSE(without.produced);
    EXPECT_FALSE(without.reject.empty());
    EXPECT_NE(without.reject, live) << "on this fixture the table-less answer must differ, or the "
                                       "replay arm cannot tell recorded inputs from none";
    EXPECT_EQ(with.reject, live);
    // The two answers stop at different instructions: the live table carries the store at pc 6
    // through, so the live and replayed rejects name pc 8; without it the store itself refuses.
    EXPECT_NE(live.find(" pc=8 "), std::string::npos) << live;
    EXPECT_NE(without.reject.find(" pc=6 "), std::string::npos) << without.reject;
}

TEST(RefusedShaderInputs, LaterInputSetsAreCounted) {
    fresh_root("refused_inputs_count");
    const auto code = fixture_code();
    refuse_live(code, 0x7000, live_inputs(7));
    EXPECT_EQ(refused_shader_input_sets("cs", 0x7000), 1u);
    refuse_live(code, 0x7000, live_inputs(7));
    EXPECT_EQ(refused_shader_input_sets("cs", 0x7000), 1u) << "a repeat is not a new set";
    refuse_live(code, 0x7000, live_inputs(9));
    EXPECT_EQ(refused_shader_input_sets("cs", 0x7000), 2u);
    const fs::path dir = refused_shader_dump_directory();
    const fs::path bin = only_file(dir, ".bin");
    fs::path counter = bin;
    counter.replace_extension(".input_sets");
    std::ifstream f(counter);
    std::string line;
    ASSERT_TRUE(std::getline(f, line));
    EXPECT_EQ(line, "distinct_input_sets=2");
    // Only the first set's inputs are kept.
    const RefusedShaderInputs inputs = read_refused_shader_inputs(refused_shader_inputs_path(bin));
    EXPECT_EQ(inputs.compute.user_sgprs[4], 7u);
}

TEST(RefusedShaderInputs, DumpWithoutInputsStillWorks) {
    fresh_root("refused_inputs_legacy");
    const auto code = fixture_code();
    refuse_live(code, 0x8000, live_inputs(), /*with_inputs=*/false);
    const fs::path dir = refused_shader_dump_directory();
    const fs::path bin = only_file(dir, ".bin");
    ASSERT_FALSE(bin.empty());
    EXPECT_FALSE(fs::exists(refused_shader_inputs_path(bin)));
    EXPECT_EQ(refused_shader_input_sets("cs", 0x8000), 0u);
    std::ifstream index(dir / "index.txt");
    const std::string text{std::istreambuf_iterator<char>(index), {}};
    EXPECT_EQ(text.find(" inputs="), std::string::npos);
    // The table-less offline answer still runs on the bare dump.
    const RefusedShaderReplay bare = replay_refused_shader_inputs(read_words(bin), {}, false);
    EXPECT_FALSE(bare.produced);
    EXPECT_TRUE(bare.incomplete.empty());
}

TEST(RefusedShaderInputs, CorruptSidecarIsRefused) {
    const fs::path root = fresh_root("refused_inputs_corrupt");
    const LiveInputs in = live_inputs();
    const fs::path path = root / "c.inputs";
    ASSERT_TRUE(write_refused_compute_inputs(path, 0x42, &in.table, in.config, "r"));
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(f), {}};
    EXPECT_NO_THROW(decode_refused_shader_inputs(bytes));
    auto edited = bytes;
    edited[20] ^= 1u;
    EXPECT_THROW(decode_refused_shader_inputs(edited), std::runtime_error);
    auto truncated = bytes;
    truncated.resize(truncated.size() / 2);
    EXPECT_THROW(decode_refused_shader_inputs(truncated), std::runtime_error);
}
