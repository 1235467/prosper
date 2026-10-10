// #4847: a bypass-only reader keeps its pre-load physical register definition. A raw-wide
// snapshot must still refuse a reader reached from both lifetimes before either is overwritten.
// These code-only tests do not authenticate resource bytes or claim live NGG admission.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLoadPc = 3u;
constexpr uint32_t kBypassPc = 7u;
constexpr uint32_t kJoinPc = 8u;
constexpr uint32_t kNullOffset = 0xfa000000u;
constexpr uint32_t kRegisterOffset = 0x04000000u;

// 0 s_mov_b32 s16, 42           the bypass's genuine previous definition
// 1 s_cmp_eq_u32 s0, 0          both branch outcomes are possible
// 2 s_cbranch_scc1 7
// 3 s_load_dwordx4 s[16:19], s[28:29], 0
// 5 v_mov_b32 v0, s16           numeric use makes this a raw-data load
// 6 s_branch 8                 the load path skips the bypass reader
// 7 v_mov_b32 v1, s16           sees 42, never the load's definition
// 8..11 s_mov_b32 s[16:19], 0   end both physical lifetimes before the later join
// 12 s_endpgm
std::vector<uint32_t> code(uint32_t offset = kNullOffset) {
    return {0xbe9003aau, 0xbf068000u, 0xbf850004u, 0xf408040eu, offset,
            0x7e000210u, 0xbf820001u, 0x7e020210u, 0xbe900380u, 0xbe910380u,
            0xbe920380u, 0xbe930380u, 0xbf810000u};
}

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(words.data(), words.size(), instructions), words.size());
    return instructions;
}

const Rdna2Inst& at(const std::vector<Rdna2Inst>& instructions, uint32_t pc) {
    for (const auto& in : instructions)
        if (in.pc == pc) return in;
    ADD_FAILURE() << "missing decoded pc " << pc;
    return instructions.front();
}

TEST(RawWideDisjointPaths, DecodePreconditions) {
    const auto ins = decode(code());
    EXPECT_EQ(at(ins, 0u).dst.value, 16);
    EXPECT_EQ(at(ins, 0u).src[0].value, 42);
    EXPECT_EQ(at(ins, kLoadPc).fmt, Rdna2Format::SMEM);
    EXPECT_EQ(at(ins, kLoadPc).opcode, 2u);
    EXPECT_EQ(at(ins, kLoadPc).dst.value, 16);
    EXPECT_EQ(at(ins, kLoadPc).src[0].value, 28);
    EXPECT_EQ(at(ins, 2u).simm16 + 3, static_cast<int>(kBypassPc));
    EXPECT_EQ(at(ins, 6u).simm16 + 7, static_cast<int>(kJoinPc));
    EXPECT_EQ(at(ins, kBypassPc).src[0].value, 16);
    EXPECT_EQ(rdna2_raw_wide_data_loads(ins), std::vector<uint32_t>{kLoadPc});
}

TEST(RawWideDisjointPaths, BypassOnlyReaderRetainsItsEarlierDefinition) {
    EXPECT_EQ(rdna2_proven_raw_immediate_wide_data_loads(decode(code())),
              std::vector<uint32_t>{kLoadPc});
    EXPECT_EQ(rdna2_proven_raw_register_wide_entry_loads(decode(code(kRegisterOffset))),
              std::vector<uint32_t>{kLoadPc});
}

TEST(RawWideDisjointPaths, EightWordLoadKeepsTheHighestWordSeparate) {
    auto words = code();
    words[0] = 0xbe9703aau; // pre-load s23 = 42
    words[kLoadPc] = 0xf40c040eu; // x8 s[16:23]
    words[kBypassPc] = 0x7e020217u; // bypass reads highest word s23
    EXPECT_EQ(at(decode(words), kLoadPc).opcode, 3u);
    EXPECT_EQ(rdna2_proven_raw_immediate_wide_data_loads(decode(words)),
              std::vector<uint32_t>{kLoadPc});
    words[6] = 0xbf820000u;
    EXPECT_TRUE(rdna2_proven_raw_immediate_wide_data_loads(decode(words)).empty());
}

TEST(RawWideDisjointPaths, SameWordAtARealJoinRemainsRefused) {
    auto words = code();
    words[6] = 0xbf820000u; // load path now reaches the bypass's same-word reader
    EXPECT_TRUE(rdna2_proven_raw_immediate_wide_data_loads(decode(words)).empty());
    words[4] = kRegisterOffset;
    EXPECT_TRUE(rdna2_proven_raw_register_wide_entry_loads(decode(words)).empty());
}

TEST(RawWideDisjointPaths, LoadedArmOverwriteBeforeAJoinEndsOnlyThatWord) {
    auto words = code();
    words[6] = 0xbe910387u; // load path replaces s17 with 7; its s16 remains loaded data
    words[7] = 0x7e020211u; // join reads s17, whose load lifetime ended on the load arm
    EXPECT_EQ(rdna2_proven_raw_immediate_wide_data_loads(decode(words)),
              std::vector<uint32_t>{kLoadPc});
    words[7] = 0x7e020210u; // same CFG, s16 still has both reaching lifetimes
    EXPECT_TRUE(rdna2_proven_raw_immediate_wide_data_loads(decode(words)).empty());
}

TEST(RawWideDisjointPaths, ReadBeforeItsBypassOverwriteStillRefuses) {
    auto words = code();
    words[6] = 0xbf820000u;
    // The very next instruction overwrites s16; reads are checked before kills.
    EXPECT_TRUE(rdna2_proven_raw_immediate_wide_data_loads(decode(words)).empty());
    words[7] = 0xbe900387u;
    EXPECT_EQ(rdna2_proven_raw_immediate_wide_data_loads(decode(words)),
              std::vector<uint32_t>{kLoadPc});
}

TEST(RawWideDisjointPaths, ABackEdgeThatReachesTheBypassReaderIsARealJoin) {
    auto words = code();
    words[6] = 0xbf820004u; // load path jumps past the bypass to pc11
    words[11] = 0xbf85fffbu; // conditional back edge pc11 -> pc7
    EXPECT_TRUE(rdna2_proven_raw_immediate_wide_data_loads(decode(words)).empty());
    words[11] = 0xbf850000u; // no back edge: load path exits without the bypass reader
    EXPECT_EQ(rdna2_proven_raw_immediate_wide_data_loads(decode(words)),
              std::vector<uint32_t>{kLoadPc});
}

// Execute the actual production module's typed Function storage/phi path with a closed
// integer CPU oracle. This is emitted SOURCE evidence, not a hardware or picture comparison.
TEST(RawWideDisjointPaths, EmittedBypassUsesTheEarlierWordAndLoadArmUsesBacking) {
    auto words = code();
    words[7] = 0x7e000210u; // both paths publish their actual s16 through v0
    words.pop_back();
    words.insert(words.end(), {0x7e020280u, 0xe0702000u, 0x80010001u, 0xbf810000u});
    ASSERT_EQ(rdna2_proven_raw_immediate_wide_data_loads(decode(words)),
              std::vector<uint32_t>{kLoadPc});
    const auto ins = decode(words);
    const auto& store = at(ins, 13u);
    ASSERT_EQ(store.fmt, Rdna2Format::MUBUF);
    ASSERT_TRUE(rdna2_may_write_guest_memory(store));
    ASSERT_EQ(store.src[1].value, 4);
    std::array<uint32_t, 4> source{0x12345678u, 2u, 3u, 4u};
    ShaderResourceTable table;
    ShaderResource input;
    input.cls = ResourceClass::ConstantBuffer;
    input.format = DataFormat::Uint32;
    input.num_components = 1u;
    input.stride = sizeof(uint32_t);
    input.binding = 2u;
    input.fetch_pc = kLoadPc;
    input.sgpr_base = 28u;
    input.gpu_addr = 0x100000000ull;
    input.size = sizeof(source);
    input.host_data = reinterpret_cast<uint8_t*>(source.data());
    input.host_data_size = sizeof(source);
    table.resources.push_back(input);
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1u;
    output.stride = sizeof(uint32_t);
    output.binding = 3u;
    output.fetch_pc = store.pc;
    output.sgpr_base = 4u;
    output.gpu_addr = 0x200000000ull;
    output.size = sizeof(uint32_t);
    table.resources.push_back(output);
    for (uint32_t selector : {0u, 1u}) {
        SCOPED_TRACE(selector);
        ComputeShaderConfig config;
        config.local_x = 1u;
        config.wave_size = 32u;
        config.tgid_x_en = true; // genuine dynamic s0 from the workgroup ID
        const auto module = recompile_compute(words.data(), words.size(), &table, config);
        ASSERT_FALSE(module.empty());
        bpermute_oracle::Interpreter vm(module);
        const auto got = vm.run_buffers(
            1u, {{2u, {source.begin(), source.end()}}, {3u, {0xdeadbeefu}}}, 3u, selector);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(got, std::vector<uint32_t>{selector == 0u ? 42u : source[0]});
        // A wrong bypass value cannot be mistaken for the source or untouched sink sentinel.
        EXPECT_NE(got, std::vector<uint32_t>{0u});
    }
}

}   // namespace
