// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <bit>
#include <numeric>
#include "common/cityhash.h"
#include "common/scm_rev.h"
#include "common/scope_exit.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "shader_recompiler/info_serialize.h"
#include "common/zstd_compression.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vertex_repack.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_cooperative_hmma_test.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_binaries.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_records.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_cache_v2.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_cache_v2/cache_file.h"
#include "video_core/shader_cache_v2/records.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GuessMissing;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = VideoCommon::VulkanPipelineCacheVersion;
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr auto& VULKAN_CACHE_MAGIC_NUMBER = VideoCommon::VulkanDriverPipelineCacheMagic;

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        if (device.IsMoltenVK()) {
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                if (format == Tegra::RenderTargetFormat::NONE) {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                    continue;
                }
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    // Transform feedback would capture the scaled values, a passthrough geometry shader doesn't
    // scale the outputs it writes itself, and indexed attribute reads in the fragment stage don't
    // undo the scaling, so all three keep native noperspective interpolation.
    const bool geometry_passthrough{key.unique_hashes[4] != 0 &&
                                    programs[4].is_geometry_passthrough};
    if (key.unique_hashes[5] != 0 && !geometry_passthrough && key.state.xfb_enabled == 0 &&
        !programs[5].info.loads_indexed_attributes) {
        const bool has_tess{key.unique_hashes[3] != 0};
        const bool last_pre_raster{
            (stage == Shader::Stage::VertexB && !has_tess && !has_geometry) ||
            (stage == Shader::Stage::TessellationEval && !has_geometry) ||
            (stage == Shader::Stage::Geometry && has_geometry)};
        if (stage == Shader::Stage::Fragment) {
            info.scaled_noperspective_inputs =
                previous_program && (previous_program->stage == Shader::Stage::VertexB ||
                                     previous_program->stage == Shader::Stage::TessellationEval ||
                                     previous_program->stage == Shader::Stage::Geometry);
        }
        if (last_pre_raster) {
            const auto& fragment_info{programs[5].info};
            for (size_t index = 0; index < 32; ++index) {
                if (fragment_info.interpolation[index] == Shader::Interpolation::NoPerspective) {
                    info.noperspective_outputs.set(index);
                }
            }
        }
    }
    return info;
}

/// SPIR-V emission of run-time pipelines: graphics on the GPU thread, graphics on workers and
/// compute on workers (shader cache v2)
std::array<std::atomic<std::chrono::nanoseconds::rep>, 3> runtime_spirv_ns{};

/// Runtime pipelines built from precompiled infos, and those whose recorded answers differed
/// (by the first answer that did), since the last 10 s report.
std::atomic<u32> precompiled_drawn{};
std::atomic<u32> precompiled_failed{};
std::atomic<u32> precompiled_created{};
std::atomic<u32> precompiled_from_spirv{};
std::atomic<u32> precompiled_per_stage{};
std::atomic<std::chrono::nanoseconds::rep> precompiled_gpu_ns{};
std::mutex precompiled_reasons_mutex;
std::map<std::string, u32> precompiled_reasons;

void CountPrecompiledReason(std::string_view reason) {
    std::scoped_lock lock{precompiled_reasons_mutex};
    ++precompiled_reasons[std::string{reason}];
}

// NX2INFO4: a table of each stage's info by its program and recorded environment follows the
// blobs. A file of another version is prepared again.
constexpr std::string_view PrecompiledInfosMagic{"NX2INFO4"};

/// Most recorded environments a stage's lookup checks against the game's state: a program kept
/// in many environments (other texture types, pointer states) is checked against the first ones.
constexpr size_t MaxStageCandidates = 16;

/// The pixel format traits a translation used: integer formats always (sampled textures), the
/// others only with descriptors that use them (FileEnvironment::AnswersHoldFor).
u32 UsedFormatTraits(const Shader::Info& info) {
    u32 traits = FileEnvironment::FormatTraitInteger;
    if (std::ranges::any_of(info.image_descriptors,
                            [](const auto& desc) { return desc.is_byte_addressed; })) {
        traits |= FileEnvironment::FormatTraitByteAddressed;
    }
    if (!info.texture_buffer_descriptors.empty()) {
        traits |= FileEnvironment::FormatTraitSnorm;
    }
    return traits;
}

/// EDEN_NX2_PRECOMPILED_FAST=0: precompiled pipelines are translated on the GPU thread like the
/// others (their kept infos and SPIR-V unused), to compare.
bool PrecompiledFastPath() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_FAST");
        const bool on = !(value && value[0] == '0');
        if (!on) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_PRECOMPILED_FAST=0: precompiled pipelines are "
                                    "translated at runtime");
        }
        return on;
    }();
    return enabled;
}

bool PrecompiledAudit() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_AUDIT");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
std::atomic<u32> audit_same{};
std::atomic<u32> audit_differ{};

/// The words of a kept SPIR-V blob, on the GPU thread.
std::vector<u32> ReadKeptSpirv(std::ifstream& file, std::pair<u64, u32> blob) {
    std::vector<u8> compressed(blob.second);
    file.clear();
    file.seekg(static_cast<std::streamoff>(blob.first));
    file.read(reinterpret_cast<char*>(compressed.data()), blob.second);
    const std::vector<u8> bytes = Common::Compression::DecompressDataZSTD(compressed);
    std::vector<u32> words(bytes.size() / sizeof(u32));
    std::memcpy(words.data(), bytes.data(), words.size() * sizeof(u32));
    return words;
}

/// A shader stage from the SPIR-V the preparation step kept (precompiled_infos), on a worker.
/// Counts the loads of the kept infos: a pipeline made from one load reads its SPIR-V from that
/// load's file only, and each worker reopens its stream for a new load (the file is rewritten
/// under the same name when the step prepares again).
std::atomic<u64> kept_infos_generation{1};

vk::ShaderModule BuildKeptSpirv(const Device& device, const std::filesystem::path& path,
                                u64 generation, std::pair<u64, u32> blob, u64 pipeline_hash) {
    thread_local std::ifstream file;
    thread_local std::filesystem::path file_path;
    thread_local u64 file_generation{};
    thread_local std::vector<u8> compressed;
    if (generation != kept_infos_generation.load(std::memory_order_acquire)) {
        throw Shader::LogicError("Precompiled pipeline {:016x}: its kept SPIR-V was replaced",
                                 pipeline_hash);
    }
    if (!file.is_open() || file_path != path || file_generation != generation) {
        file.close();
        file.clear();
        file.open(path, std::ios::binary);
        file_path = path;
        file_generation = generation;
    }
    const auto [offset, size] = blob;
    compressed.resize(size);
    file.clear();
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(compressed.data()), size);
    const std::vector<u8> words =
        file ? Common::Compression::DecompressDataZSTD(compressed) : std::vector<u8>{};
    if (words.empty() || words.size() % sizeof(u32) != 0) {
        file.close();
        throw Shader::LogicError("Precompiled pipeline {:016x}: kept SPIR-V unreadable",
                                 pipeline_hash);
    }
    return BuildShader(device, std::span<const u32>(reinterpret_cast<const u32*>(words.data()),
                                                    words.size() / sizeof(u32)));
}
constexpr u32 NoInfo = ~0U;

void AddRuntimeSpirvTime(size_t thread, std::chrono::steady_clock::time_point begin) {
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    runtime_spirv_ns[thread].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
        std::memory_order::relaxed);
}

vk::ShaderModule EmitShaderModule(const Device& device, const Shader::Profile& profile,
                                 const Shader::RuntimeInfo& runtime_info,
                                 Shader::IR::Program& program, Shader::Backend::Bindings& binding,
                                 u64 shader_hash, size_t stage_index,
                                 std::span<const u32> reference = {},
                                 std::vector<u32>* code_out = nullptr) {
    const auto code =
        EmitPipelineShader(profile, runtime_info, program, binding, shader_hash, reference);
    if (code_out) {
        *code_out = code;
    }
    device.SaveShader(code);
    vk::ShaderModule module{BuildShader(device, code)};

    // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
    // is independent and gated only by gpu_log_shader_dumps.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry",
                                               "fragment"};
        const std::string shader_name =
            fmt::format("shader_{:016x}_{}", shader_hash, stage_names[stage_index]);
        if (should_log) {
            const std::string shader_info = fmt::format(
                "SPIR-V size: {} bytes, hash: {:016x}", code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                         std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const std::string name{fmt::format("Shader {:016x}", shader_hash)};
        module.SetObjectNameEXT(name.c_str());
    }
    return module;
}

vk::ShaderModule EmitComputeModule(const Device& device, const Shader::Profile& profile,
                                   Shader::IR::Program& program, u64 shader_hash,
                                   std::span<const u32> reference = {},
                                   std::vector<u32>* code_out = nullptr) {
    const std::vector<u32> code{EmitComputeShader(profile, program, shader_hash, reference)};
    if (code_out) {
        *code_out = code;
    }
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", shader_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", shader_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    return spv_module;
}

/// A compute program whose SPIR-V the CPU stage of its pipeline's job emits (shader cache v2)
struct DeferredCompute {
    std::unique_ptr<QueuedIrBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    Shader::IR::Program program;
    std::vector<u32> reference_spirv;
};

struct DeferredStages {
    std::unique_ptr<QueuedIrBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    /// Pools of the stages translated on workers (ParallelTranslation)
    std::vector<std::shared_ptr<ShaderPools>> stage_pools;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
};

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

} // Anonymous namespace

struct PipelineCache::GraphicsStages {
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    /// Per pipeline stage (vertex B to fragment), for the stages the pipeline has.
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
    std::optional<MeshEmulationSetup> mesh_emulation;
};

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    if (unique_hashes != rhs.unique_hashes) {
        return false;
    }
    return std::memcmp(reinterpret_cast<const char*>(this) + sizeof(unique_hashes),
                       reinterpret_cast<const char*>(&rhs) + sizeof(unique_hashes),
                       state.Size()) == 0;
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, render_pass_cache{render_pass_cache_},
      buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue() &&
                                !device_.DriverCachesPipelines()},
      serialization_thread(1, "VkPipelineSerialization", {}, Common::ThreadPlacement::Background) {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .max_compute_shared_memory_size = device.GetMaxComputeSharedMemorySize(),
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_spirv_fragment_shuffle = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY,
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };
    profile.support_cooperative_hmma =
        device.IsCooperativeHmmaSupported() && TestCooperativeHmma(device);
    profile.support_cooperative_hmma_16x16x16 =
        !profile.support_cooperative_hmma && device.IsCooperativeHmma16x16x16Supported();
    if (profile.support_cooperative_hmma_16x16x16) {
        LOG_INFO(Render_Vulkan, "Guest tensor operations run on 16x16x16 cooperative matrices");
    }

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_per_stage_descriptor_storage_buffers = device.GetMaxPerStageDescriptorStorageBuffers(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .support_byte_addressed_r16 = true,
    };
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));

    dynamic_features = {};
    // Keys without the driver, so that a driver update keeps the recorded pipelines
    dynamic_features.driver_id = 0;
    dynamic_features.driver_version = 0;

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();
    dynamic_features.has_static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();

    const size_t cpu_threads = GetTotalPipelineWorkers();
    const size_t driver_threads =
        device.HasBrokenParallelShaderCompiling()
            ? 1
            : std::min(cpu_threads, std::max<size_t>(std::thread::hardware_concurrency() / 2, 2));
    shader_jobs.emplace(VideoCommon::ShaderJobConfig{
        .cpu_threads = cpu_threads,
        .driver_threads = driver_threads,
        .name = "VkShader",
    });
}

namespace {
// Pipelines the last two sessions used are built at boot. Each holds ~0.45 MB of driver memory:
// title 1 built 4,800 from eight sessions, 1.5 GB more than building them on first use (but that
// stalled frames 4x as often), while one session of its title screen uses ~1,500.
// MoltenVK keeps the previous eight sessions: a pipeline built while playing goes through
// SPIR-V -> MSL -> Metal, far more than the 2-3 ms the shorter window assumes, so each one
// built at boot is a stutter avoided.
// Ocarina still built pipelines for 1-2 s of every 10 s early in play with two sessions kept:
// keep sixteen (#161).
constexpr u8 UsedPipelinesMaxAge = 16;
constexpr u8 UsedPipelinesMaxAgeMoltenVK = 8;

u8 MaxUsedPipelineAge(const Device& device) {
    if (device.IsMoltenVK()) {
        return UsedPipelinesMaxAgeMoltenVK;
    }
    return UsedPipelinesMaxAge;
}

// Without a list of the pipelines recent sessions used (the first boot with this cache, or a session
// that ended before saving it), this many cached graphics pipelines at most are built at boot and the
// rest when first needed. A cache that grew over many builds held 48k pipelines, ~21 GB of driver
// memory built before the game started; the list is only saved at exit or after two minutes, so a
// boot that ran out of memory never saved one and the next boot did the same.
constexpr size_t MaxPipelinesWithoutUsedList = 3000;

// Captured environments the shader precache holds for skipped graphics pipelines; the rest wait
// for a later precache.
constexpr size_t PrecacheCaptureLimit = size_t{1} << 30;

// Shader cache v2: the records of the background prebuild the queue may hold, by payload bytes
constexpr u64 PrebuildCaptureLimit = u64{256} << 20;

// Shader cache v2: bytes a job's stage is taken to hold for the scheduler's budget. EGLA-80
// measured ~100 MiB of commit per driver compile in flight (24 at once held ~2.5 GB more than
// idle). A translation's IR pools grow with the program; first guesses until measured.
constexpr u64 DriverCompileBytes = u64{96} << 20;
/// What a precompiled pipeline's job holds while it translates its recorded environments
constexpr u64 PrecompiledTranslationBytes = u64{32} << 20;

u64 TranslationBytes(u64 record_bytes) {
    return std::clamp<u64>(record_bytes * 512, u64{8} << 20, u64{512} << 20);
}

u64 IrBytes(std::span<const Shader::IR::Program> programs) {
    u64 bytes{};
    for (const Shader::IR::Program& program : programs) {
        for (const Shader::IR::Block* const block : program.blocks) {
            bytes += sizeof(Shader::IR::Block) + block->Instructions().size() * sizeof(Shader::IR::Inst);
        }
    }
    return bytes;
}

// EDEN_NX2_PRELOAD_ALL_PIPELINES=1 builds every cached pipeline at boot, as before.
bool PreloadAllPipelines() {
    static const bool all = [] {
        const char* value = std::getenv("EDEN_NX2_PRELOAD_ALL_PIPELINES");
        return value != nullptr && value[0] == '1';
    }();
    return all;
}
} // Anonymous namespace

PipelineCache::~PipelineCache() {
    serialization_stop.request_stop();
    StopBackgroundJobs();
    // The pipelines' builds end before the driver pipeline cache they build into goes.
    shader_jobs->Shutdown();
    serialization_thread.WaitForRequests();
    SaveUsedPipelines();
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
    // After the records the serialization thread queued: its writer drains on destruction
    shader_cache_v2 = nullptr;
    cache_v2.reset();
    graphics_cache.clear();
    evicted_pipelines.clear();
    if (pipeline_binaries) {
        pipeline_binaries->LogCounts("at the end of the session");
        pipeline_binaries.reset();
    }
}

void PipelineCache::SaveUsedPipelines() {
    used_pipelines_saved = std::chrono::steady_clock::now();
    // A session too short to say what the game uses (a boot to the menu) keeps the old list.
    constexpr size_t MinimumUsed = 64;
    if (!cache_v2 || used_this_session.size() + used_and_evicted.size() < MinimumUsed) {
        return;
    }
    // This session's record, written by the cache's thread; older sessions keep theirs.
    VideoCommon::ShaderCacheV2::UsedRecord record{.session_time = used_session_time};
    for (const auto& [key, pipeline] : graphics_cache) {
        if (used_this_session.contains(pipeline.get())) {
            record.pipelines.push_back(key.Hash());
        }
    }
    record.pipelines.insert(record.pipelines.end(), used_and_evicted.begin(),
                            used_and_evicted.end());
    std::ranges::sort(record.pipelines);
    record.pipelines.erase(std::ranges::unique(record.pipelines).begin(), record.pipelines.end());
    cache_v2->Writer().Append(VideoCommon::ShaderCacheV2::RecordKind::Used, used_session_key,
                              VideoCommon::ShaderCacheV2::EncodeUsed(record));
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {

    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    if (maxwell3d->GetNativeMacroEngine().native_turing) {
        native_state.Refresh(*maxwell3d, dynamic_features);
        graphics_key.state = native_state;
        const VideoCommon::ShaderInfo* const vertex{shader_infos[1]};
        u32 mask = ~0U;
        if (maxwell3d->regs.IsMeshEnabled()) {
            mask = 0;
        } else if (vertex && !IsUnmaskedVertexProgram(vertex->unique_hash)) {
            mask = vertex->generic_read_mask;
        }
        graphics_key.state.CanonicalizeNative(mask);
    } else {
        graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    }

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            if (current_pipeline->IsPrecompiled() && current_pipeline->Failed()) {
                return ReplaceFailedPrecompiled();
            }
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* ComputePipelineCacheEntry::Get(bool native, GPUVAddr address,
                                                const ComputePipelineCacheKey& key) const {
    if (!pipeline && native) {
        throw Shader::RuntimeError(
            "Native compute pipeline unavailable: address={:#x} shader={:#016x} pipeline={:#016x}: {}",
            address, key.unique_hash, key.Hash(),
            failure.empty() ? "No compiler diagnostic was recorded" : failure);
    }
    return pipeline.get();
}

constexpr u32 MaxComputeRebuilds = 4;

namespace {
/// Which pointer slots a native program's arguments fill depends on the area (Ocarina of Time's
/// foliage culling fills different list slots in Hyrule Field and Castle Town), so a translation
/// rebuilt in one area may lack pointers the first cached one binds. Try first the translations
/// that bind every pointer the first one binds, those binding the most first, then the others in
/// cache order.
void OrderComputeTranslations(std::vector<CachedComputeTranslation>& candidates) {
    if (candidates.size() < 2) {
        return;
    }
    std::vector<std::vector<u64>> keys;
    keys.reserve(candidates.size());
    for (const CachedComputeTranslation& candidate : candidates) {
        keys.push_back(candidate.env.MappedStoragePointerKeys());
    }
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), size_t{0});
    const auto keeps_first = [&](size_t index) {
        return std::ranges::includes(keys[index], keys.front());
    };
    std::ranges::stable_sort(order, [&](size_t lhs, size_t rhs) {
        const bool lhs_keeps{keeps_first(lhs)};
        if (lhs_keeps != keeps_first(rhs)) {
            return lhs_keeps;
        }
        return lhs_keeps && keys[lhs].size() > keys[rhs].size();
    });
    std::vector<CachedComputeTranslation> ordered;
    ordered.reserve(candidates.size());
    for (const size_t index : order) {
        ordered.push_back(std::move(candidates[index]));
    }
    candidates = std::move(ordered);
}
} // Anonymous namespace

bool PipelineCache::StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const {
    const auto& qmd{kepler_compute->launch_description};
    for (const auto& [index, offset] : entry.null_pointer_slots) {
        if (index >= qmd.const_buffer_config.size() ||
            ((qmd.const_buffer_enable_mask >> index) & 1) == 0) {
            continue;
        }
        const auto& cbuf{qmd.const_buffer_config[index]};
        if (offset + sizeof(u64) > cbuf.size) {
            continue; // The translator reads past the bound size as zero, too
        }
        const GPUVAddr slot{cbuf.Address() + offset};
        if (!gpu_memory->IsFullyMappedRange(slot, sizeof(u64))) {
            continue;
        }
        const GPUVAddr address{gpu_memory->Read<u64>(slot)};
        if (address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
            gpu_memory->IsFullyMappedRange(address, 1)) {
            LOG_INFO(Render_Vulkan,
                     "Rebuilding compute program {:#x}: c{}[{:#x}] was null at translation and "
                     "now points to {:#x}",
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, index, offset, address);
            return true;
        }
    }
    return false;
}

ComputePipeline* PipelineCache::CurrentComputePipeline(u64& native_key) {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    native_key = shader->lowered_native ? shader->lowered_native->native_program_key : 0;
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& entry{pair->second};
    if (is_new) {
        entry = CreateComputePipeline(key, shader);
    } else if (entry.pipeline && entry.pipeline->IsPrecompiled() &&
               entry.pipeline->BuildFailed()) {
        precompiled_rejected.insert(key.Hash());
        precompiled_failed.fetch_add(1, std::memory_order_relaxed);
        retired_compute_pipelines.emplace_back(scheduler.CurrentTick(), std::move(entry.pipeline));
        entry = CreateComputePipeline(key, shader);
    } else if (!entry.null_pointer_slots.empty() && entry.rebuilds < MaxComputeRebuilds &&
               StaleNullPointerSlot(entry)) {
        // The translation dropped accesses through a pointer that was null back then. Which slots
        // the game fills depends on the area, so the new one keeps binding every pointer this one
        // binds, null now or not: a null pointer binds nothing.
        auto rebuilt{CreateComputePipeline(key, shader, entry.mapped_pointers)};
        if (rebuilt.pipeline) {
            if (entry.pipeline) {
                retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                       std::move(entry.pipeline));
            }
            rebuilt.rebuilds = entry.rebuilds + 1;
            rebuilt.used = entry.used;
            entry = std::move(rebuilt);
        } else {
            // Keep the working translation when the new one cannot be built (for example a
            // narrow access that now stays global across two buffers), and stop retrying.
            LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed: {}",
                        key.unique_hash, rebuilt.failure);
            entry.rebuilds = MaxComputeRebuilds;
        }
    }
    if (VideoCommon::StutterTrace::Enabled() && !std::exchange(entry.used, true)) {
        VideoCommon::StutterTrace::CountLookup(
            is_new ? VideoCommon::StutterTrace::Lookup::ComputeNew
                   : VideoCommon::StutterTrace::Lookup::ComputePrebuilt);
    }
    return entry.Get(kepler_compute->IsNativeMode(),
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, key);
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    if (!pipeline_cache_filename.empty()) {
        StopBackgroundJobs();
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
        shader_cache_v2 = nullptr;
        cache_v2.reset();
        pipeline_binaries.reset();
    }

    namespace StutterTrace = VideoCommon::StutterTrace;
    const auto load_begin = std::chrono::steady_clock::now();
    StutterTrace::BootMarker("shader_load_begin");
    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    namespace CacheV2 = VideoCommon::ShaderCacheV2;
    pipeline_cache_filename = base_dir / CacheV2::FileName;
    const auto migration =
        MigrateLegacyPipelineCache(base_dir, pipeline_cache_filename, MaxUsedPipelineAge(device));
    if (migration.version) {
        LOG_INFO(Render_Vulkan,
                 "Shader cache v2 started from the version {} cache: {} graphics and {} "
                 "compute pipelines ({} records collapsed), {} used-list entries",
                 *migration.version, migration.graphics, migration.compute, migration.collapsed,
                 migration.used);
    } else if (migration.failed) {
        LOG_WARNING(Render_Vulkan, "Couldn't start shader cache v2 from the older cache");
    }
    cache_v2 = std::make_unique<CacheV2::Cache>(pipeline_cache_filename);
    shader_cache_v2 = cache_v2.get();
    pipeline_binaries = PipelineBinaries::Open(device, base_dir);
    // Without the lock: another emulator's appends are complete records or a cut-off tail.
    // The payload hashes are checked on the shader job workers.
    auto log_v2 = std::make_shared<CacheV2::LogFile>(CacheV2::ReadLogFile(
        pipeline_cache_filename, [this](size_t count, const std::function<void(size_t)>& task) {
            VideoCommon::ParallelFor(*shader_jobs, count, task);
        }));
    cache_v2->LoadPrograms(log_v2->parsed.live);
    const CacheV2::LogStats& stats = log_v2->parsed.stats;
    LOG_INFO(Render_Vulkan,
             "Shader cache v2: {} bytes, {} live records, {} superseded, {} tombstones, {} "
             "damaged headers, {} damaged records, {} bytes skipped, {} bytes cut off",
             stats.file_size, stats.live, stats.superseded, stats.tombstones, stats.bad_headers,
             stats.bad_payloads, stats.skipped_bytes, stats.torn_tail_bytes);
    std::vector<u64> expired_sessions;
    used_pipelines =
        UsedPipelinesFromLog(log_v2->parsed.live, MaxUsedPipelineAge(device), expired_sessions);
    for (const u64 session : expired_sessions) {
        cache_v2->Writer().Append(CacheV2::RecordKind::Tombstone, session,
                                  CacheV2::TombstonePayload(CacheV2::RecordKind::Used));
    }
    used_session_time = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count());
    used_session_key = used_session_time ^ std::random_device{}();
    used_pipelines_saved = std::chrono::steady_clock::now();

    // With the shader precache this boot, the cached graphics pipelines skipped below are built
    // on the loading screen too, unless an earlier precache built them for this driver.
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const bool precache = ShaderWarm::PrecacheBeforeStart(
        Settings::values.precache_shaders.GetValue(), ShaderWarm::TakeWarmBeforeStart());
    std::unordered_set<u64> precached_pipelines;
    if (precache) {
        precached_pipelines = ShaderWarm::ReadPrecachedPipelines(
            ShaderWarm::PrecachedPipelinesPath(title_id),
            {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION});
    }
    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename =
            VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_pipelines", CACHE_VERSION);
        if (VideoCommon::CarryOverPipelineCache(
                base_dir, "vulkan_pipelines", VULKAN_CACHE_MAGIC_NUMBER,
                VideoCommon::CompatibleVulkanPipelineCacheVersions, CACHE_VERSION)
                .failed) {
            LOG_WARNING(Render_Vulkan,
                        "Couldn't copy the older driver pipeline cache; this one starts empty");
        }
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }
    LoadPipelinesV2(title_id, std::move(log_v2), stop_loading, callback, load_begin, precache,
                    std::move(precached_pipelines));
}

bool PipelineCache::KeyFitsDevice(const GraphicsPipelineCacheKey& key) const {
    if ((key.state.extended_dynamic_state != 0) != dynamic_features.has_extended_dynamic_state ||
        (key.state.extended_dynamic_state_2 != 0) !=
            dynamic_features.has_extended_dynamic_state_2 ||
        (key.state.extended_dynamic_state_2_logic_op != 0) !=
            dynamic_features.has_extended_dynamic_state_2_logic_op ||
        (key.state.extended_dynamic_state_3_blend != 0) !=
            dynamic_features.has_extended_dynamic_state_3_blend ||
        (key.state.extended_dynamic_state_3_enables != 0) !=
            dynamic_features.has_extended_dynamic_state_3_enables ||
        (key.state.color_write_enable_dynamic != 0) != dynamic_features.has_color_write_enable ||
        (key.state.dynamic_vertex_input != 0) != dynamic_features.has_dynamic_vertex_input) {
        return false;
    }
    const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
    if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
        return false;
    }
    const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
    return !key_uses_transform_feedback || !key_requests_provoking_last ||
           dynamic_features.has_provoking_vertex_tf_preserve;
}

PipelineBuildQueue PipelineCache::BuildQueue(bool build_in_parallel, u64 cpu_bytes) {
    if (!build_in_parallel) {
        return {.owner_builds = true};
    }
    return {.jobs = &*shader_jobs, .cpu_bytes = cpu_bytes, .driver_bytes = DriverCompileBytes};
}

std::shared_ptr<ShaderPools> PipelineCache::TakeSparePools() {
    {
        std::scoped_lock lock{spare_pools_mutex};
        if (!spare_pools.empty()) {
            auto pools = std::move(spare_pools.back());
            spare_pools.pop_back();
            return pools;
        }
    }
    return std::make_shared<ShaderPools>();
}

void PipelineCache::ReturnPools(std::shared_ptr<ShaderPools> pools) {
    pools->ReleaseContents();
    std::scoped_lock lock{spare_pools_mutex};
    if (spare_pools.size() < 4) {
        spare_pools.push_back(std::move(pools));
    }
}

void PipelineCache::StopBackgroundJobs() {
    scheduled_prebuild.reset();
}

std::unique_ptr<GraphicsPipeline> PipelineCache::TranslateCachedGraphics(
    std::span<const u8> payload, const GraphicsPipelineCacheKey& key,
    PipelineStatistics* statistics, bool& incomplete) {
    std::vector<FileEnvironment> envs;
    Common::UniqueFunction<void, std::istream&, FileEnvironment> read_compute{
        [](std::istream&, FileEnvironment) {
            throw std::ios_base::failure("Compute pipeline in another kind of record");
        }};
    Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>> read_graphics{
        [&](std::istream& file, std::vector<FileEnvironment> read) {
            GraphicsPipelineCacheKey read_key;
            file.read(reinterpret_cast<char*>(&read_key), sizeof(read_key));
            VideoCommon::CheckGraphicsRecord(read_key.unique_hashes, read.size());
            envs = std::move(read);
        }};
    VideoCommon::ReadPipelinePayload(payload, read_compute, read_graphics);
    ShaderPools pools;
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> env_ptrs;
    for (auto& env : envs) {
        env_ptrs.push_back(&env);
    }
    auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), statistics, false)};
    incomplete = !pipeline && std::ranges::any_of(envs, [this](const FileEnvironment& env) {
        return env.LacksRecordedValue(host_info.EnvironmentReadProperties());
    });
    return pipeline;
}

void PipelineCache::LoadPipelinesV2(u64 title_id,
                                    std::shared_ptr<const VideoCommon::ShaderCacheV2::LogFile> log,
                                    std::stop_token stop_loading,
                                    const VideoCore::DiskResourceLoadCallback& callback,
                                    std::chrono::steady_clock::time_point load_begin,
                                    bool precache, std::unordered_set<u64> precached_pipelines) {
    namespace CacheV2 = VideoCommon::ShaderCacheV2;
    namespace StutterTrace = VideoCommon::StutterTrace;
    using CacheV2::RecordKind;
    using VideoCommon::ShaderJobPriority;
    auto& jobs = *shader_jobs;
    const u32 read_properties = host_info.EnvironmentReadProperties();

    // The boot builds (Soon), which Loading Shaders waits for, and the background prebuild. A
    // stopped load cancels both; the workers stay for the game.
    const VideoCommon::ShaderJobBatch boot = jobs.CreateBatch();
    scheduled_prebuild.emplace(jobs, PrebuildCaptureLimit);
    const std::stop_callback cancel_load{
        stop_loading, [&jobs, boot, prebuild = scheduled_prebuild->Batch()] {
            jobs.Cancel(boot);
            jobs.Cancel(prebuild);
        }};

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
        /// Records the load never chooses, as in the old cache; begin is the index in records
        std::vector<VideoCommon::PipelineRecord> incomplete;
        std::vector<std::pair<u64, VideoCommon::PipelineRecord>> incomplete_graphics;
        std::vector<VideoCommon::PipelineRecord> copies;
    } state;
    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    // Under state.mutex
    const auto built_one = [&state, &callback] {
        ++state.built;
        if (state.has_loaded) {
            callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
        }
    };
    // The graphics and compute records read, by the index a PipelineRecord's begin stands for
    std::vector<const CacheV2::LogRecord*> records;
    const auto pipeline_record = [&records](size_t index) {
        return VideoCommon::PipelineRecord{.begin = index, .end = index + 1, .hash = records[index]->key};
    };
    // Intact, but not a record this build reads: it goes.
    const auto unreadable = [this](const CacheV2::LogRecord& record, const char* reason) {
        LOG_ERROR(Common_Filesystem, "Shader cache v2 record {:016x}: {}", record.key, reason);
        cache_v2->Writer().Append(RecordKind::Tombstone, record.key,
                                  CacheV2::TombstonePayload(record.kind));
    };

    // One pass over the records without reading them, which the jobs do in parallel: all
    // translations of a compute key go to one job, graphics records by their key's hash.
    ::Common::unordered_map<ComputePipelineCacheKey, std::vector<size_t>> compute_keys;
    std::vector<std::pair<size_t, GraphicsPipelineCacheKey>> boot_graphics;
    struct PrebuildRecord {
        GraphicsPipelineCacheKey key;
        std::vector<u8> payload;
    };
    std::vector<PrebuildRecord> prebuild;
    u64 prebuild_bytes = 0;
    std::vector<PrecacheJob> precache_graphics;
    u64 precache_capture_bytes = 0;
    std::vector<CacheV2::ProgramRecord> stale_programs;
    for (const CacheV2::LogRecord& record : log->parsed.live) {
        if (record.kind != RecordKind::Program) {
            continue;
        }
        const auto stamp = CacheV2::ProgramStamp(record.payload);
        if (!stamp) {
            continue;
        }
        cache_v2->AddProgram(record.key, *stamp);
        if (*stamp != CacheV2::LoweringStamp()) {
            if (auto program = CacheV2::DecodeProgram(record.payload)) {
                stale_programs.push_back(std::move(*program));
            }
        }
    }
    // Programs another build recorded are lowered again before any of their pipelines is built:
    // the records of those whose code or texture bound buffer changed move to this build's code
    // (a pipeline key holds the code's hash alone, so one built from the old record would be hit
    // with the new texture bound buffer). Their own records follow last, so that a load cut short
    // does this again at the next boot.
    std::vector<CacheV2::LogAppend> relinked_appends;
    std::vector<CacheV2::LogRecord> relinked_records;
    std::unordered_set<u64> replaced_records;
    if (!stale_programs.empty() && !stop_loading.stop_requested()) {
        const size_t stale_count = stale_programs.size();
        CacheV2::ForEach for_each = [&jobs](size_t count, const std::function<void(size_t)>& task) {
            VideoCommon::ParallelFor(jobs, count, task);
        };
        RevalidatedPrograms revalidated =
            RevalidateProgramRecords(std::move(stale_programs), for_each);
        relinked_appends =
            RelinkPipelineRecords(log->parsed.live, revalidated.relinked,
                                  [&stop_loading] { return stop_loading.stop_requested(); });
        if (!stop_loading.stop_requested()) {
            for (const CacheV2::LogAppend& append : relinked_appends) {
                cache_v2->Writer().Append(append.kind, append.key, append.payload);
                replaced_records.insert(append.key);
                if (append.kind == RecordKind::Tombstone) {
                    continue;
                }
                relinked_records.push_back({.kind = append.kind,
                                            .flags = 0,
                                            .key = append.key,
                                            .offset = 0,
                                            .payload = append.payload});
            }
            for (const CacheV2::LogAppend& append :
                 RelinkUsedPipelines(log->parsed.live, relinked_appends, MaxUsedPipelineAge(device),
                                     used_pipelines)) {
                cache_v2->Writer().Append(append.kind, append.key, append.payload);
            }
            for (const CacheV2::ProgramRecord& program : revalidated.unchanged) {
                cache_v2->RecordProgram(program);
            }
            for (const auto& [hash, relinked] : revalidated.relinked) {
                cache_v2->RecordProgram(relinked.program);
            }
            LOG_INFO(Render_Vulkan,
                     "Shader cache v2: {} programs of another build lowered again, {} to other "
                     "code or texture bound buffer, {} pipeline records moved to it",
                     stale_count, revalidated.relinked.size(), relinked_records.size());
        }
    }
    std::vector<const CacheV2::LogRecord*> plan;
    plan.reserve(log->parsed.live.size() + relinked_records.size());
    for (const CacheV2::LogRecord& record : log->parsed.live) {
        if (record.kind == RecordKind::Program ||
            ((record.kind == RecordKind::Graphics || record.kind == RecordKind::Compute) &&
             replaced_records.contains(record.key))) {
            continue;
        }
        plan.push_back(&record);
    }
    for (const CacheV2::LogRecord& record : relinked_records) {
        plan.push_back(&record);
    }

    size_t graphics_records = 0;
    size_t compute_records = 0;
    size_t skipped_pipelines = 0;
    size_t queued_graphics = 0;
    // Native keys are canonical (FixedPipelineState::CanonicalizeNative). A record an older build
    // wrote holds its key as that draw made it: it is built under the canonical key, which draws
    // now look up, and the used list may name it by either. Reading the environments for this
    // runs on the shader job workers.
    std::vector<std::optional<GraphicsPipelineCacheKey>> canonical_keys(plan.size());
    VideoCommon::ParallelFor(jobs, plan.size(), [&](size_t index) {
        const CacheV2::LogRecord& record = *plan[index];
        if (record.kind != RecordKind::Graphics || stop_loading.stop_requested()) {
            return;
        }
        Common::UniqueFunction<void, std::istream&, FileEnvironment> read_compute{
            [](std::istream&, FileEnvironment) {}};
        Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>> read_graphics{
            [&](std::istream& file, std::vector<FileEnvironment> envs) {
                GraphicsPipelineCacheKey key;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                if (!file || envs.size() > Maxwell::MaxShaderProgram) {
                    return;
                }
                boost::container::static_vector<const Shader::Environment*,
                                                Maxwell::MaxShaderProgram>
                    env_ptrs;
                for (const auto& env : envs) {
                    env_ptrs.push_back(&env);
                }
                CanonicalizeNativeKey(key, env_ptrs);
                canonical_keys[index] = key;
            }};
        try {
            VideoCommon::ReadPipelinePayload(record.payload, read_compute, read_graphics);
        } catch (const std::exception&) {
            // Unreadable: the plan below reports it when it reads the key, or its build does.
        }
    });
    for (size_t plan_index = 0; plan_index < plan.size(); ++plan_index) {
        const CacheV2::LogRecord& record = *plan[plan_index];
        if (stop_loading.stop_requested()) {
            break;
        }
        if (record.kind == RecordKind::Compute) {
            ++compute_records;
            const auto key = PipelineRecordKey<ComputePipelineCacheKey>(record.payload);
            if (!key) {
                unreadable(record, "Compute pipeline record without a key");
                continue;
            }
            compute_keys[*key].push_back(records.size());
            records.push_back(&record);
            continue;
        }
        if (record.kind != RecordKind::Graphics) {
            continue;
        }
        ++graphics_records;
        auto key = PipelineRecordKey<GraphicsPipelineCacheKey>(record.payload);
        if (!key) {
            unreadable(record, "Graphics pipeline record without a key");
            continue;
        }
        const u64 stored_hash = key->Hash();
        if (canonical_keys[plan_index]) {
            key = canonical_keys[plan_index];
        }
        const size_t index = records.size();
        records.push_back(&record);
        if (!KeyFitsDevice(*key)) {
            continue;
        }
        const u64 hash = key->Hash();
        if (!disk_pipelines.insert(hash).second) {
            continue;
        }
        if (PreloadAllPipelines() ||
            (used_pipelines.empty() ? queued_graphics < MaxPipelinesWithoutUsedList
                                    : used_pipelines.contains(hash) ||
                                          used_pipelines.contains(stored_hash))) {
            ++queued_graphics;
            boot_graphics.emplace_back(index, *key);
            continue;
        }
        ++skipped_pipelines;
        const u64 bytes = record.payload.size();
        std::vector<u8> payload(record.payload.begin(), record.payload.end());
        if (precache && !precached_pipelines.contains(hash) &&
            bytes <= PrecacheCaptureLimit - precache_capture_bytes) {
            precache_capture_bytes += bytes;
            precache_graphics.push_back({
                .graphics_hash = hash,
                .translate = [this, key = *key, payload = std::move(payload)]()
                    -> Common::UniqueFunction<bool> {
                    bool incomplete{};
                    auto pipeline = TranslateCachedGraphics(payload, key, nullptr, incomplete);
                    if (!pipeline) {
                        return {};
                    }
                    return [pipeline = std::move(pipeline)] {
                        pipeline->Build();
                        return !pipeline->Failed();
                    };
                },
                .cpu_bytes = TranslationBytes(bytes),
            });
        } else if (bytes <= PrebuildCaptureLimit - prebuild_bytes) {
            prebuild_bytes += bytes;
            prebuild.push_back({*key, std::move(payload)});
        }
    }

    for (auto& [key, indices] : compute_keys) {
        ++state.total;
        u64 bytes = 0;
        for (const size_t index : indices) {
            bytes = std::max<u64>(bytes, records[index]->payload.size());
        }
        jobs.Submit(
            {
                .priority = ShaderJobPriority::Soon,
                .cpu_bytes = TranslationBytes(bytes),
                .driver_bytes = DriverCompileBytes,
                .cpu = [this, &state, &built_one, &records, &pipeline_record, &unreadable, key,
                        indices = std::move(indices),
                        read_properties](VideoCommon::ShaderJobContext& context)
                    -> VideoCommon::ShaderDriverStage {
                    std::vector<CachedComputeTranslation> candidates;
                    for (const size_t index : indices) {
                        const CacheV2::LogRecord& record = *records[index];
                        Common::UniqueFunction<void, std::istream&, FileEnvironment> read_compute{
                            [&](std::istream& file, FileEnvironment env) {
                                ComputePipelineCacheKey read_key;
                                file.read(reinterpret_cast<char*>(&read_key), sizeof(read_key));
                                VideoCommon::CheckComputeRecord(env, read_key.shared_memory_size,
                                                                read_key.workgroup_size);
                                candidates.push_back({std::move(env), pipeline_record(index)});
                            }};
                        Common::UniqueFunction<void, std::istream&, std::vector<FileEnvironment>>
                            read_graphics{[](std::istream&, std::vector<FileEnvironment>) {
                                throw std::ios_base::failure(
                                    "Graphics pipeline in another kind of record");
                            }};
                        try {
                            VideoCommon::ReadPipelinePayload(record.payload, read_compute,
                                                             read_graphics);
                        } catch (const std::ios_base::failure& e) {
                            unreadable(record, e.what());
                        }
                    }
                    OrderComputeTranslations(candidates);
                    ShaderPools pools;
                    ComputePipelineCacheEntry entry;
                    size_t chosen{};
                    for (; chosen < candidates.size() && !context.StopRequested(); ++chosen) {
                        pools.ReleaseContents();
                        entry = CreateComputePipeline(pools, key, candidates[chosen].env,
                                                      state.statistics.get(), false);
                        if (entry.pipeline) {
                            break;
                        }
                    }
                    std::scoped_lock lock{state.mutex};
                    AddUnchosenComputeTranslations(candidates, chosen, read_properties,
                                                   state.incomplete, state.copies);
                    if (!entry.pipeline) {
                        built_one();
                        return {};
                    }
                    return [this, &state, &built_one, key, entry = std::move(entry)]() mutable {
                        entry.pipeline->Build();
                        std::scoped_lock driver_lock{state.mutex};
                        compute_cache.emplace(key, std::move(entry));
                        built_one();
                    };
                },
            },
            &boot);
    }
    compute_keys.clear();
    for (const auto& [index, key] : boot_graphics) {
        ++state.total;
        const CacheV2::LogRecord& record = *records[index];
        jobs.Submit(
            {
                .priority = ShaderJobPriority::Soon,
                .cpu_bytes = TranslationBytes(record.payload.size()),
                .driver_bytes = DriverCompileBytes,
                .cpu = [this, &state, &built_one, &pipeline_record, &unreadable, &record, index,
                        key](VideoCommon::ShaderJobContext& context)
                    -> VideoCommon::ShaderDriverStage {
                    std::unique_ptr<GraphicsPipeline> pipeline;
                    bool incomplete{};
                    if (!context.StopRequested()) {
                        try {
                            pipeline = TranslateCachedGraphics(record.payload, key,
                                                               state.statistics.get(), incomplete);
                        } catch (const std::ios_base::failure& e) {
                            unreadable(record, e.what());
                        }
                    }
                    if (!pipeline) {
                        std::scoped_lock lock{state.mutex};
                        if (incomplete) {
                            state.incomplete_graphics.emplace_back(key.Hash(),
                                                                   pipeline_record(index));
                        }
                        built_one();
                        return {};
                    }
                    return [this, &state, &built_one, key,
                            pipeline = std::move(pipeline)]() mutable {
                        pipeline->Build();
                        std::scoped_lock lock{state.mutex};
                        graphics_cache.emplace(key, std::move(pipeline));
                        built_one();
                    };
                },
            },
            &boot);
    }
    boot_graphics.clear();

    // With the precache, the prebuild starts after it.
    size_t deferred_pipelines = 0;
    const auto start_prebuild = [&] {
        for (PrebuildRecord& record : prebuild) {
            const u64 bytes = record.payload.size();
            const bool added = scheduled_prebuild->Add(
                record.key, bytes, TranslationBytes(bytes), DriverCompileBytes,
                [this, key = record.key, payload = std::move(record.payload)] {
                    bool incomplete{};
                    try {
                        return TranslateCachedGraphics(payload, key, nullptr, incomplete);
                    } catch (const std::ios_base::failure&) {
                        return std::unique_ptr<GraphicsPipeline>{};
                    }
                });
            deferred_pipelines += added ? 1 : 0;
        }
        prebuild.clear();
    };
    if (!precache) {
        start_prebuild();
    }

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    if (skipped_pipelines != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached graphics pipelines skipped at boot, {} {} for background prebuild",
                 skipped_pipelines, precache ? prebuild.size() : deferred_pipelines,
                 precache ? "held" : "admitted");
    }
    {
        std::scoped_lock lock{state.mutex};
        callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
        state.has_loaded = true;
    }
    boot.Wait();

    std::vector<DuplicateGraphicsRecord> no_duplicates;
    AddIncompleteGraphics(state.incomplete_graphics, no_duplicates, disk_pipelines,
                          state.incomplete);
    // The records the load never chooses go by tombstones, and the log is compacted once a
    // quarter of it is records no load reads.
    if (!stop_loading.stop_requested()) {
        for (const auto* removed : {&state.incomplete, &state.copies}) {
            for (const VideoCommon::PipelineRecord& record : *removed) {
                const CacheV2::LogRecord& read = *records[record.begin];
                cache_v2->Writer().Append(RecordKind::Tombstone, read.key,
                                          CacheV2::TombstonePayload(read.kind));
            }
        }
        const CacheV2::LogStats& stats = log->parsed.stats;
        if (stats.dead_bytes != 0 && stats.dead_bytes >= stats.file_size / 4) {
            cache_v2->Writer().Compact();
        }
    }
    records.clear();
    log.reset();

    PreparePrecompiled(pipeline_cache_filename.parent_path(), stop_loading, callback);

    if (use_vulkan_pipeline_cache) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        last_flush = std::chrono::steady_clock::now();
    }
    if (state.statistics) {
        state.statistics->Report();
    }
    if (StutterTrace::Enabled()) {
        const auto bytes = [](const std::filesystem::path& path) {
            std::error_code error;
            const auto size = std::filesystem::file_size(path, error);
            return error ? std::uintmax_t{0} : size;
        };
        StutterTrace::SetCacheFile(Common::FS::PathToUTF8String(pipeline_cache_filename));
        StutterTrace::BootMarker(
            "shader_load_end",
            fmt::format("seconds={:.3f} graphics_records={} compute_records={} queued={} skipped={} "
                        "deferred={} cache_bytes={} driver_cache_bytes={}",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - load_begin)
                            .count(),
                        graphics_records, compute_records, state.total, skipped_pipelines,
                        precache ? prebuild.size() : deferred_pipelines,
                        bytes(pipeline_cache_filename), bytes(vulkan_pipeline_cache_filename)));
    }
    if (precache && !stop_loading.stop_requested()) {
        PrecacheShaders(title_id, std::move(precache_graphics), std::move(precached_pipelines),
                        stop_loading, callback);
    }
    if (stop_loading.stop_requested()) {
        scheduled_prebuild.reset();
    } else if (precache) {
        start_prebuild();
        LOG_INFO(Render_Vulkan, "{} cached graphics pipelines admitted for background prebuild",
                 deferred_pipelines);
    }
}

void PipelineCache::PrecacheShaders(u64 title_id, std::vector<PrecacheJob> jobs,
                                    std::unordered_set<u64> precached, std::stop_token stop_loading,
                                    const VideoCore::DiskResourceLoadCallback& callback) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const size_t graphics_jobs = jobs.size();
    const std::filesystem::path warm_file = ShaderWarm::ComputeWarmPath(title_id);
    const std::string path = Common::FS::PathToUTF8String(warm_file);
    const auto stamp = [&](u64 entries) {
        return ShaderWarm::CurrentWarmStamp(warm_file, entries, device.GetDriverVersion(),
                                            device.GetDeviceID(), CACHE_VERSION);
    };
    // Compute programs of the precache file, unless a matching marker says they were built.
    size_t file_entries = 0;
    bool compute_pending = false;
    std::error_code ec;
    const auto current = stamp(0);
    const auto marker = ShaderWarm::ReadWarmMarker(warm_file);
    if (!std::filesystem::exists(warm_file, ec)) {
        LOG_INFO(Render_Vulkan, "Shader precache: no precache file at {}", path);
    } else if (current && marker && marker->Matches(*current)) {
        LOG_INFO(Render_Vulkan, "Shader precache: compute programs already done for this driver");
    } else {
        // Read the whole file here (its own reader: the disk cache loader deletes files it rejects).
        try {
            std::ifstream file(warm_file, std::ios::binary | std::ios::ate);
            file.exceptions(std::ifstream::failbit);
            const auto end{file.tellg()};
            file.seekg(0, std::ios::beg);
            std::array<char, 8> magic;
            u32 version{};
            file.read(magic.data(), magic.size())
                .read(reinterpret_cast<char*>(&version), sizeof(version));
            constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
            if (magic != expected_magic || version != CACHE_VERSION) {
                LOG_WARNING(Render_Vulkan, "Shader precache: {} is not a version {} file", path,
                            CACHE_VERSION);
                throw std::runtime_error("wrong version");
            }
            while (file.tellg() != end) {
                u32 num_envs{};
                file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                FileEnvironment env;
                env.Deserialize(file, end);
                if (num_envs != 1 || env.ShaderStage() != Shader::Stage::Compute) {
                    LOG_WARNING(Render_Vulkan, "Shader precache: unexpected entry in {}", path);
                    throw std::runtime_error("unexpected entry");
                }
                ComputePipelineCacheKey key;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                ++file_entries;
                // Programs already in the disk cache were built above; leave them out.
                if (compute_cache.contains(key)) {
                    continue;
                }
                const u64 cpu_bytes = TranslationBytes(env.MemorySize());
                jobs.push_back({
                    .translate = [this, key,
                                  env = std::move(env)]() mutable -> Common::UniqueFunction<bool> {
                        static const GuessMissing guess{};
                        env.SetGuessMissing(&guess);
                        ShaderPools pools;
                        auto pipeline =
                            CreateComputePipeline(pools, key, env, nullptr, false).pipeline;
                        if (!pipeline) {
                            return {};
                        }
                        return [pipeline = std::move(pipeline)] {
                            pipeline->Build();
                            return true;
                        };
                    },
                    .cpu_bytes = cpu_bytes,
                });
            }
            compute_pending = true;
        } catch (const std::exception& e) {
            LOG_WARNING(Render_Vulkan, "Shader precache: reading {} failed: {}", path, e.what());
            jobs.resize(graphics_jobs);
            file_entries = 0;
        }
    }
    if (jobs.empty() && !compute_pending) {
        LOG_INFO(Render_Vulkan, "Shader precache: nothing to precache");
        return;
    }

    const auto begin = std::chrono::steady_clock::now();
    const size_t num_threads = GetTotalPipelineWorkers();
    std::vector<u8> built(jobs.size());
    std::atomic<size_t> done{};
    bool skipped = false;
    LOG_INFO(Render_Vulkan,
             "Shader precache: {} cached graphics pipelines and {} of {} compute programs to "
             "build on {} threads",
             graphics_jobs, jobs.size() - graphics_jobs, file_entries, num_threads);
    VideoCommon::StutterTrace::BootMarker("precache_begin", fmt::format("jobs={}", jobs.size()));
    if (!jobs.empty()) {
        // Background jobs of the shader workers, each built and destroyed: only the driver's
        // cache keeps it.
        const VideoCommon::ShaderJobBatch batch = shader_jobs->CreateBatch();
        for (size_t index = 0; index < jobs.size(); ++index) {
            shader_jobs->Submit(
                {
                    .priority = VideoCommon::ShaderJobPriority::Background,
                    .cpu_bytes = jobs[index].cpu_bytes,
                    .driver_bytes = DriverCompileBytes,
                    .cpu = [&built, &done, index, translate = std::move(jobs[index].translate)](
                               VideoCommon::ShaderJobContext& context) mutable
                    -> VideoCommon::ShaderDriverStage {
                        Common::UniqueFunction<bool> build;
                        if (!context.StopRequested()) {
                            try {
                                build = translate();
                            } catch (const std::exception& e) {
                                LOG_DEBUG(Render_Vulkan, "Shader precache: {}", e.what());
                            }
                        }
                        translate = {};
                        if (!build) {
                            done.fetch_add(1);
                            return {};
                        }
                        return [&built, &done, index, build = std::move(build)]() mutable {
                            try {
                                built[index] = build();
                            } catch (const std::exception& e) {
                                LOG_DEBUG(Render_Vulkan, "Shader precache: {}", e.what());
                            }
                            done.fetch_add(1);
                        };
                    },
                },
                &batch);
        }
        while (!batch.Idle()) {
            if (stop_loading.stop_requested() || ShaderWarm::TakeSkipWarm()) {
                // The builds not started are dropped; the running ones end first.
                shader_jobs->Cancel(batch);
                skipped = true;
                break;
            }
            if (callback) {
                callback(VideoCore::LoadCallbackStage::Warm, done, jobs.size());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        batch.Wait();
    }
    const size_t total_built = static_cast<size_t>(std::count(built.begin(), built.end(), u8{1}));
    if (compute_pending && !skipped) {
        if (const auto stamp_done = stamp(file_entries)) {
            ShaderWarm::WriteWarmMarker(warm_file, *stamp_done);
        }
    }
    const size_t precached_before = precached.size();
    for (size_t index = 0; index < graphics_jobs; ++index) {
        if (built[index]) {
            precached.insert(jobs[index].graphics_hash);
        }
    }
    if (precached.size() != precached_before) {
        ShaderWarm::WritePrecachedPipelines(
            ShaderWarm::PrecachedPipelinesPath(title_id),
            {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION}, precached);
    }
    if (!stop_loading.stop_requested() && total_built != 0) {
        // Without this, the builds reach the file only at the next boot, game change or clean
        // close, and are lost if the app ends otherwise.
        QueueVulkanPipelineCacheSave();
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    LOG_INFO(Render_Vulkan, "Shader precache: {}, {} built, {} failed, {} not reached, {:.1f} s",
             stop_loading.stop_requested() ? "stopped" : (skipped ? "skipped" : "finished"),
             total_built, done - total_built, jobs.size() - std::min(jobs.size(), done.load()),
             seconds);
    VideoCommon::StutterTrace::BootMarker(
        "precache_end", fmt::format("built={} seconds={:.3f}", total_built, seconds));
}

void PipelineCache::CanonicalizeNativeKey(
    GraphicsPipelineCacheKey& key,
    std::span<const Shader::Environment* const> envs) const noexcept {
    if (envs.empty() || !std::ranges::all_of(envs, [](const Shader::Environment* env) {
            return env->IsNativeProgram();
        })) {
        return;
    }
    u32 mask = ~0U;
    for (const Shader::Environment* const env : envs) {
        if (env->ShaderStage() == Shader::Stage::Task || env->ShaderStage() == Shader::Stage::Mesh) {
            mask = 0;
            break;
        }
        if (env->ShaderStage() == Shader::Stage::VertexB &&
            !IsUnmaskedVertexProgram(key.unique_hashes[1])) {
            mask = VideoCommon::GenericReadMask(env->SPH());
        }
    }
    key.state.CanonicalizeNative(mask);
}

namespace {
/// Whether the driver keeps compiled pipelines in a cache of its own (the proprietary NVIDIA and
/// AMD drivers), where the preparation step leaves them.
bool DriverCachesPipelines(const Device& device) {
    switch (device.GetDriverID()) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return true;
    default:
        return false;
    }
}
} // Anonymous namespace

void PipelineCache::PreparePrecompiled(const std::filesystem::path& base_dir,
                                       std::stop_token stop_loading,
                                       const VideoCore::DiskResourceLoadCallback& callback) {
    // A load before this one may hold the kept infos open, which the step rewrites and renames.
    precompiled_file.close();
    precompiled_infos_file.close();
    kept_infos_generation.fetch_add(1, std::memory_order_acq_rel);
    precompiled_records.clear();
    precompiled_compute_records.clear();
    precompiled_stages.clear();
    const char* enabled = std::getenv("EDEN_NX2_PRECOMPILED");
    if (enabled && enabled[0] == '0') {
        return;
    }
    // The prepared pipelines are kept by the driver's own cache. Without one they would go to the
    // emulator's pipeline cache file, which is loaded whole at boot: only on request there.
    if (!DriverCachesPipelines(device) && !(enabled && enabled[0] == '1')) {
        LOG_INFO(Render_Vulkan, "Precompiled pipelines skipped: the driver keeps no pipeline cache "
                                "of its own (EDEN_NX2_PRECOMPILED=1 prepares them anyway)");
        return;
    }
    // Layout (tools/nx2-precompile/README.md): "NX2PREP1", u32 cache version, u32 environment
    // count, u32 record count, u32 reserved; per environment u64 offset and u64 size of a
    // GenericEnvironment::Serialize blob; per record u8 kind (0 graphics, 1 compute), u8
    // environment count, u16 reserved, u32 environment indices in stage order, then the key.
    const auto path = base_dir / fmt::format("precompiled.fork-v{}.nx2prep", CACHE_VERSION);
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error) {
        // A file built for an older cache version is not read: it has to be built again.
        for (u32 version = CACHE_VERSION; version-- > 50;) {
            const auto older = base_dir / fmt::format("precompiled.fork-v{}.nx2prep", version);
            if (std::filesystem::exists(older, error)) {
                LOG_WARNING(Render_Vulkan,
                            "Precompiled pipelines skipped: {} is for cache version {}, this "
                            "build uses {}",
                            older.filename().string(), version, CACHE_VERSION);
                break;
            }
        }
        return;
    }
    const auto modified = std::filesystem::last_write_time(path, error).time_since_epoch().count();
    // The driver keeps what it compiled; the step runs again for another file, driver, build or
    // shader settings: the resolution scale is folded into the shaders (RescalingPass), and the
    // bloom fix, loop safety checks and debug renderer change their translation or SPIR-V.
    const auto stamp_path =
        base_dir / fmt::format("precompiled_prepared.fork-v{}.txt", CACHE_VERSION);
    const auto infos_path =
        base_dir / fmt::format("precompiled_infos.fork-v{}.bin", CACHE_VERSION);
    const auto& resolution = Settings::values.resolution_info;
    const std::string stamp = fmt::format(
        "{} {} {:#x} {} {} scale {} {}/{} bloom {} loops {} debug {}\n", file_size, modified,
        device.GetDriverVersion(), Common::g_scm_rev, device.GetModelName(), resolution.active,
        resolution.up_scale, resolution.down_shift, Settings::values.fix_bloom_effects.GetValue(),
        Settings::values.disable_shader_loop_safety_checks.GetValue(),
        Settings::values.renderer_debug.GetValue());
    const char* force = std::getenv("EDEN_NX2_PRECOMPILED_FORCE");
    if (!(force && force[0] == '1')) {
        std::ifstream stamp_file(stamp_path, std::ios::binary);
        const std::string previous{std::istreambuf_iterator<char>(stamp_file), {}};
        if (previous == stamp) {
            if (LoadPrecompiledInfos(path, infos_path, stamp)) {
                LOG_INFO(Render_Vulkan, "Precompiled pipelines already prepared for this driver");
                return;
            }
            LOG_INFO(Render_Vulkan, "Precompiled pipelines were prepared without keeping their "
                                    "shader infos: preparing them again");
        }
    }

    struct Record {
        bool compute{};
        std::vector<u32> envs;
        GraphicsPipelineCacheKey graphics_key{};
        ComputePipelineCacheKey compute_key{};
    };
    std::vector<std::pair<u64, u64>> env_index;
    std::vector<Record> records;
    try {
        std::ifstream file(path, std::ios::binary);
        file.exceptions(std::ifstream::failbit);
        std::array<char, 8> magic{};
        u32 version{}, env_count{}, record_count{}, reserved{};
        file.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version))
            .read(reinterpret_cast<char*>(&env_count), sizeof(env_count))
            .read(reinterpret_cast<char*>(&record_count), sizeof(record_count))
            .read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
        if (std::string_view{magic.data(), magic.size()} != "NX2PREP1" || version != CACHE_VERSION) {
            LOG_WARNING(Render_Vulkan, "Precompiled pipeline file {} has another format or version",
                        Common::FS::PathToUTF8String(path));
            return;
        }
        // Counts the file is too small for are damage, caught before anything is allocated: an
        // index entry is 16 bytes and a record at least 8.
        constexpr u64 HeaderSize = 24;
        if (file_size < HeaderSize ||
            u64{env_count} * sizeof(env_index[0]) + u64{record_count} * 8 >
                file_size - HeaderSize) {
            LOG_ERROR(Render_Vulkan, "Precompiled pipeline file {} is damaged",
                      Common::FS::PathToUTF8String(path));
            return;
        }
        env_index.resize(env_count);
        file.read(reinterpret_cast<char*>(env_index.data()), env_count * sizeof(env_index[0]));
        for (u32 i = 0; i < record_count; ++i) {
            u8 kind{}, count{};
            u16 pad{};
            file.read(reinterpret_cast<char*>(&kind), 1)
                .read(reinterpret_cast<char*>(&count), 1)
                .read(reinterpret_cast<char*>(&pad), 2);
            Record record;
            record.compute = kind == 1;
            record.envs.resize(count);
            file.read(reinterpret_cast<char*>(record.envs.data()), count * sizeof(u32));
            if (record.compute) {
                file.read(reinterpret_cast<char*>(&record.compute_key), sizeof(record.compute_key));
            } else {
                file.read(reinterpret_cast<char*>(&record.graphics_key), sizeof(record.graphics_key));
            }
            if (kind > 1 || count == 0 || (record.compute && count != 1) ||
                std::ranges::any_of(record.envs, [&](u32 env) { return env >= env_count; })) {
                LOG_ERROR(Render_Vulkan, "Precompiled pipeline file {} is damaged",
                          Common::FS::PathToUTF8String(path));
                return;
            }
            if (!record.compute) {
                // One environment per stage the key names, as for the disk cache's records.
                VideoCommon::CheckGraphicsRecord(record.graphics_key.unique_hashes, count);
            }
            records.push_back(std::move(record));
        }
    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Render_Vulkan, "Cannot read precompiled pipelines: {}", e.what());
        return;
    }

    const size_t total = records.size();
    // Environments are read from the file by each job; this bounds the records submitted and not
    // finished, so the checkpoint below stays close to the work done.
    const size_t max_in_flight = 256;

    // A checkpoint lets an interrupted step (closed, or a crash) continue where it stopped. A boot
    // that starts twice from the same record without getting further skips past it.
    const auto progress_path =
        base_dir / fmt::format("precompiled_progress.fork-v{}.txt", CACHE_VERSION);
    size_t start_index = 0;
    u32 restarts = 0;
    {
        std::ifstream progress_file(progress_path, std::ios::binary);
        std::string line;
        size_t next{};
        u32 count{};
        if (std::getline(progress_file, line) && line + '\n' == stamp &&
            (progress_file >> next >> count)) {
            start_index = std::min(next, total);
            restarts = count + 1;
        }
    }
    if (restarts >= 2) {
        const size_t jump = std::min(total - start_index, max_in_flight * 2);
        LOG_WARNING(Render_Vulkan,
                    "Precompiled pipelines stopped twice near record {}: skipping {} records",
                    start_index, jump);
        start_index += jump;
        restarts = 0;
    }
    const auto write_progress = [&](size_t next, u32 count) {
        std::ofstream progress_file(progress_path, std::ios::binary | std::ios::trunc);
        progress_file << stamp << next << ' ' << count << '\n';
    };
    write_progress(start_index, restarts);

    struct {
        std::mutex mutex;
        std::condition_variable_any cv;
        size_t in_flight{};
        size_t built{};
        size_t failed{};
        size_t skipped{};
        size_t low_water{};      ///< first record not finished yet
        size_t checkpoint{};
        std::vector<u8> done;
    } state;
    state.done.assign(total, 0);
    // The shader infos and SPIR-V of each graphics pipeline built, kept once each (most
    // pipelines share their programs' infos with others, and stages their SPIR-V), written to
    // the file as they come.
    // NX2INFO4: magic, u32 stamp size, the stamp, u64 index offset, the blobs (an info:
    // Shader::SerializeInfo; SPIR-V: one zstd frame of its words); at the index: u32 record
    // count, u32 blob count, per record u64 key hash, u8 environment count, u8 kind (0
    // graphics, 1 compute: its info and SPIR-V are stage 0's), 2 bytes 0, u32
    // environments[5], u32 info blobs[5] and u32 SPIR-V blobs[5] (per pipeline stage, ~0 for
    // none), per blob u64 offset, u32 size, u32 0; then u32 stage count, per stage u64 program
    // hash, u8 pipeline stage, 3 bytes 0, u32 environment, u32 info blob (graphics only).
    struct {
        std::mutex mutex;
        std::ofstream file;
        u64 offset{};
        std::unordered_map<u64, u32> infos_by_hash;
        std::unordered_map<u64, u32> spirv_by_hash;
        std::vector<std::pair<u64, u32>> blobs;
        std::vector<std::array<u32, 5>> record_infos;
        std::vector<std::array<u32, 5>> record_spirv;
        std::unordered_set<u32> legacy_loads; ///< Info blobs with legacy attribute loads
        u64 spirv_bytes{};
    } kept;
    kept.record_infos.assign(total, {NoInfo, NoInfo, NoInfo, NoInfo, NoInfo});
    kept.record_spirv.assign(total, {NoInfo, NoInfo, NoInfo, NoInfo, NoInfo});
    auto kept_temporary = infos_path;
    kept_temporary += ".tmp";
    kept.file.open(kept_temporary, std::ios::binary | std::ios::trunc);
    const auto put = [](std::ofstream& file, const auto& value) {
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    kept.file.write(PrecompiledInfosMagic.data(), PrecompiledInfosMagic.size());
    put(kept.file, static_cast<u32>(stamp.size()));
    kept.file.write(stamp.data(), static_cast<std::streamsize>(stamp.size()));
    const u64 index_offset_at = static_cast<u64>(kept.file.tellp());
    put(kept.file, u64{0});
    kept.offset = static_cast<u64>(kept.file.tellp());
    // Under kept.mutex: the blob's id, written once.
    const auto keep_blob = [&kept](std::unordered_map<u64, u32>& by_hash, u64 hash,
                                   std::span<const u8> bytes) -> u32 {
        const auto [it, added] = by_hash.try_emplace(hash, static_cast<u32>(kept.blobs.size()));
        if (added) {
            kept.file.write(reinterpret_cast<const char*>(bytes.data()),
                            static_cast<std::streamsize>(bytes.size()));
            kept.blobs.emplace_back(kept.offset, static_cast<u32>(bytes.size()));
            kept.offset += bytes.size();
        }
        return it->second;
    };
    // A compute pipeline's info and SPIR-V are its stage 0.
    const auto keep_infos = [&kept, &keep_blob](
                                size_t index, const Shader::Info* compute_info,
                                const GraphicsPipeline* graphics,
                                const std::array<std::vector<u32>, Maxwell::MaxShaderStage>& spirv) {
        if (graphics && graphics->IsEmulatedMesh()) {
            return;
        }
        std::array<std::vector<u8>, 5> infos;
        std::array<std::vector<u8>, 5> compressed;
        std::array<u64, 5> info_hash{};
        std::array<u64, 5> spirv_hash{};
        std::array<bool, 5> legacy_loads{};
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            const Shader::Info* const info =
                graphics ? graphics->StageInfo(stage) : (stage == 0 ? compute_info : nullptr);
            if (!info) {
                continue;
            }
            Shader::SerializeInfo(*info, infos[stage]);
            legacy_loads[stage] = info->loads.Legacy();
            info_hash[stage] = Common::CityHash64(reinterpret_cast<const char*>(infos[stage].data()),
                                                  infos[stage].size());
            if (!spirv[stage].empty()) {
                const auto* const words = reinterpret_cast<const u8*>(spirv[stage].data());
                const size_t size = spirv[stage].size() * sizeof(u32);
                spirv_hash[stage] =
                    Common::CityHash64(reinterpret_cast<const char*>(words), size);
                compressed[stage] = Common::Compression::CompressDataZSTD(words, size, 3);
            }
        }
        std::scoped_lock lock{kept.mutex};
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage].empty()) {
                continue;
            }
            kept.record_infos[index][stage] =
                keep_blob(kept.infos_by_hash, info_hash[stage], infos[stage]);
            if (legacy_loads[stage]) {
                kept.legacy_loads.insert(kept.record_infos[index][stage]);
            }
            if (!compressed[stage].empty()) {
                const size_t before = kept.blobs.size();
                kept.record_spirv[index][stage] =
                    keep_blob(kept.spirv_by_hash, spirv_hash[stage], compressed[stage]);
                if (kept.blobs.size() != before) {
                    kept.spirv_bytes += compressed[stage].size();
                }
            }
        }
    };
    state.low_water = state.checkpoint = start_index;
    const auto begin = std::chrono::steady_clock::now();
    const auto complete = [&](size_t index) {
        // Under state.mutex.
        state.done[index] = 1;
        while (state.low_water < total && state.done[state.low_water]) {
            ++state.low_water;
        }
        if (state.low_water >= state.checkpoint + 512) {
            state.checkpoint = state.low_water;
            write_progress(state.low_water, 0);
        }
        const size_t done = start_index + state.built + state.failed + state.skipped;
        if (done % 64 == 0 || done == total) {
            callback(VideoCore::LoadCallbackStage::Build, done, total);
        }
    };
    const auto finish = [&](size_t index, bool ok) {
        std::scoped_lock lock{state.mutex};
        --state.in_flight;
        ++(ok ? state.built : state.failed);
        complete(index);
        state.cv.notify_all();
    };
    const auto skip = [&](size_t index) {
        std::scoped_lock lock{state.mutex};
        ++state.skipped;
        complete(index);
    };
    // A job a stop cancelled before it built: its slot is freed, but it is not done, so the
    // checkpoint stays before it and the next boot builds it.
    const auto abandon = [&] {
        std::scoped_lock lock{state.mutex};
        --state.in_flight;
        state.cv.notify_all();
    };
    const auto read_envs = [&path, &env_index](const std::vector<u32>& ids) {
        std::ifstream file(path, std::ios::binary);
        file.exceptions(std::ifstream::failbit);
        std::vector<FileEnvironment> envs(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            file.clear();
            const auto [offset, size] = env_index[ids[i]];
            file.seekg(static_cast<std::streamoff>(offset));
            envs[i].Deserialize(file, static_cast<std::streamoff>(offset + size));
        }
        return envs;
    };
    // Each record is a job of the shader job scheduler, as the boot's and the precache's builds
    // are: its CPU stage reads the environments and translates, and returns the driver build, so
    // the IR is freed before the driver compiles. The scheduler's memory budget bounds the stages
    // that run at once. Soon: the game waits for this step, while the background prebuild doesn't.
    const VideoCommon::ShaderJobBatch batch = shader_jobs->CreateBatch();
    LOG_INFO(Render_Vulkan, "Preparing {} precompiled pipelines from {}, from record {}", total,
             Common::FS::PathToUTF8String(path), start_index);
    if (start_index != 0) {
        LOG_WARNING(Render_Vulkan,
                    "Continuing after a crash: the {} records before {} are drawn after translating "
                    "them, without kept infos",
                    start_index, start_index);
    }
    callback(VideoCore::LoadCallbackStage::Build, start_index, total);
    for (size_t index = start_index; index < total; ++index) {
        if (stop_loading.stop_requested()) {
            break;
        }
        auto& record = records[index];
        u64 env_bytes = 0;
        for (const u32 id : record.envs) {
            env_bytes += env_index[id].second;
        }
        // A compute record is built even when the disk cache built its key: a key has a record for
        // each storage pointer state the program can be translated in, and the boot built one.
        if (!record.compute) {
            auto& key = record.graphics_key;
            // Keys leave the driver out (as the cache's records do), so a record made before a
            // driver update keys the pipeline as draws will.
            ClearDriverFields(key);
            // Made for a device with other dynamic state. (Pipelines also in the disk cache are
            // built too: their shader infos are kept like the others'.)
            if (!KeyFitsDevice(key)) {
                skip(index);
                continue;
            }
        }
        {
            std::unique_lock lock{state.mutex};
            // Stopping cancels the jobs, so the slots would never free up.
            if (!state.cv.wait(lock, stop_loading,
                               [&] { return state.in_flight < max_in_flight; })) {
                break;
            }
            ++state.in_flight;
        }
        shader_jobs->Submit(
            {.priority = VideoCommon::ShaderJobPriority::Soon,
             .cpu_bytes = TranslationBytes(env_bytes),
             .driver_bytes = DriverCompileBytes,
             .cpu = [this, index, &record, &read_envs, &finish,
                     &abandon, &keep_infos](VideoCommon::ShaderJobContext& context) -> VideoCommon::ShaderDriverStage {
                 if (context.StopRequested()) {
                     abandon();
                     return {};
                 }
                 Common::UniqueFunction<bool> build;
                 try {
                     auto envs = read_envs(record.envs);
                     ShaderPools pools;
                     if (record.compute) {
                         std::array<std::vector<u32>, Maxwell::MaxShaderStage> spirv;
                         auto entry{CreateComputePipeline(pools, record.compute_key, envs.front(),
                                                          nullptr, false, nullptr, &spirv[0])};
                         if (entry.pipeline) {
                             build = [index, &keep_infos, spirv = std::move(spirv),
                                      pipeline = std::move(entry.pipeline)] {
                                 pipeline->Build();
                                 if (pipeline->BuildFailed()) {
                                     return false;
                                 }
                                 keep_infos(index, &pipeline->StageInfo(), nullptr, spirv);
                                 return true;
                             };
                         }
                     } else {
                         boost::container::static_vector<Shader::Environment*,
                                                         Maxwell::MaxShaderProgram>
                             env_ptrs;
                         for (auto& env : envs) {
                             env_ptrs.push_back(&env);
                         }
                         std::array<std::vector<u32>, Maxwell::MaxShaderStage> spirv;
                         auto pipeline{CreateGraphicsPipeline(pools, record.graphics_key,
                                                              MakeSpan(env_ptrs), nullptr, false,
                                                              nullptr, nullptr, &spirv)};
                         if (pipeline) {
                             build = [index, &keep_infos, spirv = std::move(spirv),
                                      pipeline = std::move(pipeline)] {
                                 pipeline->Build();
                                 if (pipeline->Failed()) {
                                     return false;
                                 }
                                 keep_infos(index, nullptr, pipeline.get(), spirv);
                                 return true;
                             };
                         }
                     }
                 } catch (const std::exception& e) {
                     LOG_DEBUG(Render_Vulkan, "Precompiled pipeline failed: {}", e.what());
                 }
                 if (!build) {
                     finish(index, false);
                     return {};
                 }
                 return [index, &finish, build = std::move(build)]() mutable {
                     bool ok = false;
                     try {
                         ok = build();
                     } catch (const std::exception& e) {
                         LOG_DEBUG(Render_Vulkan, "Precompiled pipeline failed: {}", e.what());
                     }
                     finish(index, ok);
                 };
             }},
            &batch);
    }
    while (!batch.Idle()) {
        if (stop_loading.stop_requested()) {
            shader_jobs->Cancel(batch);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    batch.Wait();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    LOG_INFO(Render_Vulkan, "Precompiled pipelines: {} prepared, {} failed, {} skipped, in {:.1f} s",
             state.built, state.failed, state.skipped, seconds);
    if (stop_loading.stop_requested()) {
        kept.file.close();
        std::filesystem::remove(kept_temporary, error);
        // The next boot prepares from the first record again, so the infos it keeps are
        // complete; only a crash (which leaves the checkpoint) continues from the checkpoint.
        std::filesystem::remove(progress_path, error);
        return;
    }
    {
        std::vector<size_t> kept_records;
        for (size_t index = 0; index < total; ++index) {
            // An entry holds 5 environments: a record with both vertex programs (6) is drawn
            // after translating it, as without the file.
            if (records[index].envs.size() <= 5 &&
                std::ranges::any_of(kept.record_infos[index], [](u32 id) { return id != NoInfo; })) {
                kept_records.push_back(index);
            }
        }
        const u64 index_offset = kept.offset;
        put(kept.file, static_cast<u32>(kept_records.size()));
        put(kept.file, static_cast<u32>(kept.blobs.size()));
        for (const size_t index : kept_records) {
            const auto& record = records[index];
            put(kept.file, record.compute ? record.compute_key.Hash() : record.graphics_key.Hash());
            put(kept.file, static_cast<u8>(record.envs.size()));
            put(kept.file, std::array<u8, 3>{static_cast<u8>(record.compute ? 1 : 0)});
            std::array<u32, 5> envs{NoInfo, NoInfo, NoInfo, NoInfo, NoInfo};
            std::ranges::copy(record.envs, envs.begin());
            put(kept.file, envs);
            put(kept.file, kept.record_infos[index]);
            put(kept.file, kept.record_spirv[index]);
        }
        for (const auto& [offset, size] : kept.blobs) {
            put(kept.file, offset);
            put(kept.file, size);
            put(kept.file, u32{0});
        }
        // Each graphics stage's info by its program and recorded environment, for pipelines
        // whose key was not predicted: a stage's translation depends on its own program and
        // environment, except where legacy attribute loads are mapped to the previous stage's
        // outputs (ConvertLegacyToGeneric), so those infos are left out. So is a program and
        // environment whose infos differ between pipelines (an info that depends on the pipeline).
        constexpr u32 Differs = ~0U - 1;
        std::map<std::tuple<u64, u8, u32>, u32> stage_infos;
        for (const size_t index : kept_records) {
            const auto& record = records[index];
            if (record.compute || record.graphics_key.unique_hashes[0] != 0) {
                continue;
            }
            size_t position = 0;
            for (size_t program = 1; program < Maxwell::MaxShaderProgram; ++program) {
                const u64 program_hash = record.graphics_key.unique_hashes[program];
                if (program_hash == 0) {
                    continue;
                }
                if (position >= record.envs.size()) {
                    break;
                }
                const u32 env = record.envs[position++];
                const u8 stage = static_cast<u8>(program - 1);
                const u32 info = kept.record_infos[index][stage];
                if (info == NoInfo || kept.legacy_loads.contains(info)) {
                    continue;
                }
                const auto [it, added] = stage_infos.try_emplace({program_hash, stage, env}, info);
                if (!added && it->second != info) {
                    it->second = Differs;
                }
            }
        }
        const auto differing = static_cast<size_t>(std::ranges::count_if(
            stage_infos, [](const auto& entry) { return entry.second == Differs; }));
        put(kept.file, static_cast<u32>(stage_infos.size() - differing));
        for (const auto& [stage_key, info] : stage_infos) {
            if (info == Differs) {
                continue;
            }
            const auto& [program_hash, stage, env] = stage_key;
            put(kept.file, program_hash);
            put(kept.file, std::array<u8, 4>{stage});
            put(kept.file, env);
            put(kept.file, info);
        }
        LOG_INFO(Render_Vulkan,
                 "Precompiled stage infos: {} by program and environment, {} left out (their "
                 "infos differ between pipelines)",
                 stage_infos.size() - differing, differing);
        kept.file.seekp(static_cast<std::streamoff>(index_offset_at));
        put(kept.file, index_offset);
        kept.file.close();
        std::filesystem::rename(kept_temporary, infos_path, error);
        LOG_INFO(Render_Vulkan,
                 "Precompiled shader infos and SPIR-V kept for {} pipelines: {} distinct infos, "
                 "{} distinct stages of SPIR-V ({} MiB compressed), {} MiB in all",
                 kept_records.size(), kept.infos_by_hash.size(), kept.spirv_by_hash.size(),
                 kept.spirv_bytes >> 20, kept.offset >> 20);
    }
    std::ofstream stamp_file(stamp_path, std::ios::binary | std::ios::trunc);
    stamp_file << stamp;
    stamp_file.close();
    std::filesystem::remove(progress_path, error);
    LoadPrecompiledInfos(path, infos_path, stamp);
}

bool PipelineCache::LoadPrecompiledInfos(const std::filesystem::path& precompiled,
                                         const std::filesystem::path& infos,
                                         const std::string& stamp) {
    // Streams of an earlier load would be read at this file's offsets.
    precompiled_file.close();
    precompiled_infos_file.close();
    kept_infos_generation.fetch_add(1, std::memory_order_acq_rel);
    precompiled_records.clear();
    precompiled_compute_records.clear();
    precompiled_stages.clear();
    try {
        std::ifstream file(infos, std::ios::binary);
        file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        const auto get = [&file](auto& value) {
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
        };
        std::array<char, 8> magic{};
        u32 stamp_size{}, record_count{}, blob_count{};
        u64 index_offset{};
        file.read(magic.data(), magic.size());
        get(stamp_size);
        if (std::string_view{magic.data(), magic.size()} != PrecompiledInfosMagic ||
            stamp_size != stamp.size()) {
            return false;
        }
        std::string file_stamp(stamp_size, '\0');
        file.read(file_stamp.data(), stamp_size);
        if (file_stamp != stamp) {
            return false;
        }
        get(index_offset);
        file.seekg(static_cast<std::streamoff>(index_offset));
        get(record_count);
        get(blob_count);
        precompiled_records.reserve(record_count);
        for (u32 i = 0; i < record_count; ++i) {
            u64 key_hash{};
            u8 env_count{};
            std::array<u8, 3> pad{};
            PrecompiledRecord record;
            get(key_hash);
            get(env_count);
            get(pad);
            get(record.envs);
            get(record.infos);
            get(record.spirv);
            record.env_count = env_count;
            if (pad[0] == 1) {
                precompiled_compute_records.emplace(key_hash, record);
            } else {
                precompiled_records.emplace(key_hash, record);
            }
        }
        precompiled_info_blobs.resize(blob_count);
        for (auto& [offset, size] : precompiled_info_blobs) {
            u32 pad{};
            get(offset);
            get(size);
            get(pad);
        }
        u32 stage_count{};
        get(stage_count);
        precompiled_stages.reserve(stage_count);
        for (u32 i = 0; i < stage_count; ++i) {
            u64 program_hash{};
            std::array<u8, 4> stage{};
            PrecompiledStage entry;
            get(program_hash);
            get(stage);
            get(entry.env);
            get(entry.info);
            entry.stage = stage[0];
            if (entry.stage < 5 && entry.info < blob_count) {
                precompiled_stages.emplace(program_hash, entry);
            }
        }

        std::ifstream prep(precompiled, std::ios::binary);
        prep.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        std::array<u32, 4> header{};
        prep.read(magic.data(), magic.size());
        prep.read(reinterpret_cast<char*>(header.data()), sizeof(header));
        precompiled_env_index.resize(header[1]);
        prep.read(reinterpret_cast<char*>(precompiled_env_index.data()),
                  precompiled_env_index.size() * sizeof(precompiled_env_index[0]));
    } catch (const std::exception&) {
        // Unreadable, or a damaged count (a reserve that cannot be met): prepared again.
        precompiled_records.clear();
        precompiled_compute_records.clear();
        precompiled_stages.clear();
        return false;
    }
    precompiled_path = precompiled;
    precompiled_infos_path = infos;
    LOG_INFO(Render_Vulkan,
             "Precompiled shader infos for {} graphics and {} compute pipelines, and {} graphics "
             "stages by program",
             precompiled_records.size(), precompiled_compute_records.size(),
             precompiled_stages.size());
    return true;
}

/// A kept info blob, on the GPU thread; false when it is unreadable or needs layer emulation.
static bool ReadKeptInfo(std::ifstream& file, std::pair<u64, u32> blob, Shader::Info& info) {
    std::vector<u8> bytes(blob.second);
    file.clear();
    file.seekg(static_cast<std::streamoff>(blob.first));
    file.read(reinterpret_cast<char*>(bytes.data()), blob.second);
    std::span<const u8> in{bytes};
    info = {};
    return Shader::DeserializeInfo(in, info) && !info.requires_layer_emulation;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreatePrecompiledGraphicsPipeline(
    GraphicsEnvironments& environments, std::optional<PrecompiledRecord>* matched) {
    if ((precompiled_records.empty() && precompiled_stages.empty()) || !PrecompiledFastPath()) {
        return nullptr;
    }
    const u64 hash = graphics_key.Hash();
    // A vertex A program is merged into vertex B: not one of these.
    if (precompiled_rejected.contains(hash) || graphics_key.unique_hashes[0] != 0) {
        return nullptr;
    }
    // EDEN_NX2_PRECOMPILED_PER_STAGE=1: every pipeline is looked up per stage, to test it.
    static const bool per_stage_only = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_PER_STAGE");
        return value != nullptr && value[0] == '1';
    }();
    const auto it = precompiled_records.find(hash);
    if (it == precompiled_records.end() || per_stage_only) {
        std::string_view reason;
        auto pipeline = CreatePerStageGraphicsPipeline(environments, matched, reason);
        if (!pipeline && !reason.empty()) {
            CountPrecompiledReason(reason);
        }
        return pipeline;
    }
    const PrecompiledRecord& record = it->second;
    const auto live = environments.Span();
    if (live.size() != record.env_count) {
        return nullptr;
    }
    auto recorded = std::make_shared<std::vector<FileEnvironment>>(record.env_count);
    std::array<Shader::Info, 5> infos{};
    std::array<bool, 5> present{};
    try {
        OpenPrecompiledFiles();
        for (size_t i = 0; i < record.env_count; ++i) {
            precompiled_file.clear();
            const auto [offset, size] = precompiled_env_index.at(record.envs[i]);
            precompiled_file.seekg(static_cast<std::streamoff>(offset));
            (*recorded)[i].Deserialize(precompiled_file,
                                       static_cast<std::streamoff>(offset + size));
        }
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (record.infos[stage] == NoInfo) {
                continue;
            }
            if (!ReadKeptInfo(precompiled_infos_file,
                              precompiled_info_blobs.at(record.infos[stage]), infos[stage])) {
                CountPrecompiledReason("infos unreadable or with layer emulation");
                return nullptr;
            }
            present[stage] = true;
        }
    } catch (const std::exception&) {
        CountPrecompiledReason("files unreadable");
        precompiled_file.close();
        precompiled_infos_file.close();
        return nullptr;
    }
    // Each environment with the format traits its own stage's translation used.
    size_t position = 0;
    for (size_t program = 1; program < Maxwell::MaxShaderProgram; ++program) {
        if (graphics_key.unique_hashes[program] == 0) {
            continue;
        }
        if (position >= live.size()) {
            return nullptr;
        }
        const size_t stage = program - 1;
        const u32 traits = present[stage] ? UsedFormatTraits(infos[stage])
                                          : FileEnvironment::FormatTraitInteger;
        const char* why = "";
        if (!(*recorded)[position].AnswersHoldFor(*live[position], &why, traits)) {
            // The program may be kept in other environments, from other pipelines. The reason
            // counted is the predicted key's.
            std::string_view per_stage_reason;
            auto pipeline = CreatePerStageGraphicsPipeline(environments, matched, per_stage_reason);
            if (!pipeline) {
                CountPrecompiledReason(why);
            }
            return pipeline;
        }
        ++position;
    }
    // The worker builds the SPIR-V the preparation step kept: the translation of the recorded
    // environments by this build, which the check above found to give the game state's.
    // EDEN_NX2_PRECOMPILED_VERIFY=1 translates them again instead and compares the infos.
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    std::optional<std::array<std::pair<u64, u32>, 5>> spirv_blobs;
    if (!verify) {
        spirv_blobs.emplace();
        for (size_t stage = 0; stage < present.size(); ++stage) {
            if (!present[stage]) {
                continue;
            }
            if (record.spirv[stage] == NoInfo ||
                record.spirv[stage] >= precompiled_info_blobs.size()) {
                spirv_blobs.reset();
                break;
            }
            (*spirv_blobs)[stage] = precompiled_info_blobs[record.spirv[stage]];
        }
    }
    if (matched) {
        *matched = record;
    }
    return MakePrecompiledGraphicsPipeline(std::move(recorded), infos, present, spirv_blobs);
}

void PipelineCache::OpenPrecompiledFiles() {
    if (!precompiled_file.is_open()) {
        precompiled_file.open(precompiled_path, std::ios::binary);
        precompiled_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    }
    if (!precompiled_infos_file.is_open()) {
        precompiled_infos_file.open(precompiled_infos_path, std::ios::binary);
        precompiled_infos_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    }
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreatePerStageGraphicsPipeline(
    GraphicsEnvironments& environments, std::optional<PrecompiledRecord>* matched,
    std::string_view& reason) {
    if (precompiled_stages.empty()) {
        return nullptr;
    }
    const auto live = environments.Span();
    auto recorded = std::make_shared<std::vector<FileEnvironment>>(live.size());
    std::array<Shader::Info, 5> infos{};
    std::array<bool, 5> present{};
    PrecompiledRecord found;
    found.envs.fill(NoInfo);
    found.infos.fill(NoInfo);
    found.spirv.fill(NoInfo);
    size_t position = 0;
    try {
        OpenPrecompiledFiles();
        for (size_t program = 1; program < Maxwell::MaxShaderProgram; ++program) {
            const u64 program_hash = graphics_key.unique_hashes[program];
            if (program_hash == 0) {
                continue;
            }
            if (position >= live.size()) {
                return nullptr;
            }
            const size_t stage = program - 1;
            // Each stage from the first of its program's recorded environments whose answers the
            // game's state gives.
            reason = "per stage: a program not kept";
            size_t tried = 0;
            const auto [first, last] = precompiled_stages.equal_range(program_hash);
            for (auto entry = first; entry != last && tried < MaxStageCandidates; ++entry) {
                const PrecompiledStage& candidate = entry->second;
                if (candidate.stage != stage || candidate.env >= precompiled_env_index.size()) {
                    continue;
                }
                ++tried;
                if (!ReadKeptInfo(precompiled_infos_file,
                                  precompiled_info_blobs[candidate.info], infos[stage])) {
                    continue;
                }
                FileEnvironment env;
                const auto [offset, size] = precompiled_env_index[candidate.env];
                precompiled_file.clear();
                precompiled_file.seekg(static_cast<std::streamoff>(offset));
                env.Deserialize(precompiled_file, static_cast<std::streamoff>(offset + size));
                const char* why = "";
                if (!env.AnswersHoldFor(*live[position], &why, UsedFormatTraits(infos[stage]))) {
                    reason = "per stage: a program's answers differ";
                    continue;
                }
                (*recorded)[position] = std::move(env);
                found.envs[position] = candidate.env;
                found.infos[stage] = candidate.info;
                present[stage] = true;
                break;
            }
            if (!present[stage]) {
                return nullptr;
            }
            ++position;
        }
    } catch (const std::exception&) {
        reason = "files unreadable";
        precompiled_file.close();
        precompiled_infos_file.close();
        return nullptr;
    }
    reason = {};
    if (position != live.size()) {
        return nullptr;
    }
    found.env_count = static_cast<u8>(position);
    if (matched) {
        *matched = found;
    }
    precompiled_per_stage.fetch_add(1, std::memory_order_relaxed);
    return MakePrecompiledGraphicsPipeline(std::move(recorded), infos, present, std::nullopt);
}

std::unique_ptr<GraphicsPipeline> PipelineCache::MakePrecompiledGraphicsPipeline(
    std::shared_ptr<std::vector<FileEnvironment>> recorded,
    const std::array<Shader::Info, 5>& infos, const std::array<bool, 5>& present,
    const std::optional<std::array<std::pair<u64, u32>, 5>>& spirv_blobs) {
    std::array<const Shader::Info*, 5> info_ptrs{};
    for (size_t stage = 0; stage < info_ptrs.size(); ++stage) {
        info_ptrs[stage] = present[stage] ? &infos[stage] : nullptr;
    }
    if (spirv_blobs) {
        auto build_kept = [this, key = graphics_key, path = precompiled_infos_path,
                           generation = kept_infos_generation.load(std::memory_order_acquire),
                           spirv_blobs = *spirv_blobs, present] {
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
            for (size_t stage = 0; stage < modules.size(); ++stage) {
                if (present[stage]) {
                    modules[stage] = BuildKeptSpirv(device, path, generation, spirv_blobs[stage],
                                                    key.Hash());
                }
            }
            precompiled_from_spirv.fetch_add(1, std::memory_order_relaxed);
            return modules;
        };
        // The job reads and decompresses the kept SPIR-V, holding little.
        auto pipeline = std::make_unique<GraphicsPipeline>(
            scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, pipeline_binaries.get(),
            &shader_notify, device, descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
            BuildQueue(true), nullptr, render_pass_cache, graphics_key,
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage>{}, info_ptrs,
            std::move(build_kept), std::nullopt);
        pipeline->MarkPrecompiled();
        return pipeline;
    }
    // The worker translates the recorded environments, which give the same programs as the
    // game's state does; the infos drawn with must be the ones that translation gives.
    auto emit_stages = [this, key = graphics_key, recorded, infos, present] {
        ShaderPools pools;
        boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> env_ptrs;
        for (auto& env : *recorded) {
            env_ptrs.push_back(&env);
        }
        GraphicsStages stages;
        if (!TranslateGraphicsStages(pools, key, MakeSpan(env_ptrs), stages, false) ||
            stages.mesh_emulation) {
            throw Shader::LogicError("Precompiled pipeline {:016x} is not drawn natively",
                                     key.Hash());
        }
        std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
        Shader::Backend::Bindings binding;
        for (size_t stage = 0; stage < modules.size(); ++stage) {
            if (stages.runtime_infos[stage].has_value() != present[stage]) {
                throw Shader::LogicError("Precompiled pipeline {:016x}: other stages", key.Hash());
            }
            if (!present[stage]) {
                continue;
            }
            modules[stage] = EmitShaderModule(device, profile, *stages.runtime_infos[stage],
                                              stages.programs[stage + 1], binding,
                                              key.unique_hashes[stage + 1], stage);
            std::vector<u8> translated, stored;
            Shader::SerializeInfo(stages.programs[stage + 1].info, translated);
            Shader::SerializeInfo(infos[stage], stored);
            if (translated != stored) {
                throw Shader::LogicError(
                    "Precompiled pipeline {:016x}: stage {} infos differ from its translation",
                    key.Hash(), stage);
            }
        }
        precompiled_drawn.fetch_add(1, std::memory_order_relaxed);
        return modules;
    };
    auto pipeline = std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, pipeline_binaries.get(),
        &shader_notify, device, descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
        BuildQueue(true, PrecompiledTranslationBytes), nullptr, render_pass_cache, graphics_key,
        std::array<vk::ShaderModule, Maxwell::MaxShaderStage>{}, info_ptrs,
        std::move(emit_stages), std::nullopt);
    pipeline->MarkPrecompiled();
    return pipeline;
}

ComputePipelineCacheEntry PipelineCache::CreatePrecompiledComputePipeline(
    const ComputePipelineCacheKey& key, ComputeEnvironment& env,
    const PrecompiledRecord** matched) {
    if (precompiled_compute_records.empty() || !PrecompiledFastPath()) {
        return {};
    }
    const u64 hash = key.Hash();
    if (precompiled_rejected.contains(hash)) {
        return {};
    }
    // One record for each storage pointer state the program was prepared in.
    const auto [first, last] = precompiled_compute_records.equal_range(hash);
    const char* first_reason = nullptr;
    for (auto it = first; it != last; ++it) {
        const PrecompiledRecord& record = it->second;
        if (record.env_count != 1 || record.infos[0] == NoInfo || record.spirv[0] == NoInfo ||
            record.spirv[0] >= precompiled_info_blobs.size()) {
            continue;
        }
        FileEnvironment recorded;
        Shader::Info info{};
        try {
            OpenPrecompiledFiles();
            precompiled_file.clear();
            const auto [env_offset, env_size] = precompiled_env_index.at(record.envs[0]);
            precompiled_file.seekg(static_cast<std::streamoff>(env_offset));
            recorded.Deserialize(precompiled_file,
                                 static_cast<std::streamoff>(env_offset + env_size));
            if (!ReadKeptInfo(precompiled_infos_file, precompiled_info_blobs.at(record.infos[0]),
                              info)) {
                if (!first_reason) {
                    first_reason = "infos unreadable or with layer emulation";
                }
                continue;
            }
        } catch (const std::exception&) {
            CountPrecompiledReason("files unreadable");
            precompiled_file.close();
            precompiled_infos_file.close();
            return {};
        }
        u32 format_traits = FileEnvironment::FormatTraitInteger;
        if (std::ranges::any_of(info.image_descriptors,
                                [](const auto& desc) { return desc.is_byte_addressed; })) {
            format_traits |= FileEnvironment::FormatTraitByteAddressed;
        }
        if (!info.texture_buffer_descriptors.empty()) {
            format_traits |= FileEnvironment::FormatTraitSnorm;
        }
        const char* why = "";
        if (!recorded.AnswersHoldFor(env, &why, format_traits)) {
            if (!first_reason) {
                first_reason = why;
            }
            continue;
        }
        auto emit_module = [this, path = precompiled_infos_path,
                            generation = kept_infos_generation.load(std::memory_order_acquire),
                            blob = precompiled_info_blobs[record.spirv[0]], hash] {
            auto module = BuildKeptSpirv(device, path, generation, blob, hash);
            precompiled_from_spirv.fetch_add(1, std::memory_order_relaxed);
            return module;
        };
        ComputePipelineCacheEntry entry{
            .pipeline = std::make_unique<ComputePipeline>(
                device, scheduler, vulkan_pipeline_cache, pipeline_binaries.get(), key.Hash(),
                descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, BuildQueue(true),
                nullptr, &shader_notify, info, vk::ShaderModule{}, key.unique_hash,
                std::move(emit_module)),
            // The live environment answered the same questions, as the translation would have.
            .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                        : std::vector<std::pair<u32, u32>>{},
            .mapped_pointers =
                env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{}};
        entry.pipeline->MarkPrecompiled();
        if (matched) {
            *matched = &record;
        }
        return entry;
    }
    if (first_reason) {
        CountPrecompiledReason(first_reason);
    }
    return {};
}

void PipelineCache::AuditPrecompiled(std::span<Shader::Environment* const> envs, bool compute,
                                     u64 key_hash, const PrecompiledRecord* given) {
    const auto it = precompiled_records.find(key_hash);
    if (!given && (compute || it == precompiled_records.end())) {
        return;
    }
    const PrecompiledRecord& record = given ? *given : it->second;
    const auto differ = [&](size_t stage, const char* what) {
        if (audit_differ.fetch_add(1, std::memory_order_relaxed) < 32) {
            LOG_ERROR(Render_Vulkan, "Precompiled audit: {} pipeline {:016x} stage {}: {} differ",
                      compute ? "compute" : "graphics", key_hash, stage, what);
        }
    };
    try {
        std::array<std::vector<u32>, Maxwell::MaxShaderStage> live_code;
        std::array<std::vector<u8>, Maxwell::MaxShaderStage> live_info;
        std::array<bool, Maxwell::MaxShaderStage> present{};
        main_pools.ReleaseContents();
        if (compute) {
            Shader::Maxwell::Flow::CFG cfg{*envs[0], main_pools.flow_block,
                                           envs[0]->StartAddress()};
            auto program{TranslateProgram(main_pools.inst, main_pools.block, *envs[0], cfg,
                                          host_info)};
            live_code[0] = EmitSPIRV(profile, program);
            Shader::SerializeInfo(program.info, live_info[0]);
            present[0] = true;
        } else {
            GraphicsStages stages;
            if (!TranslateGraphicsStages(main_pools, graphics_key, envs, stages, false)) {
                differ(0, "the pipeline (skipped by its translation)");
                return;
            }
            Shader::Backend::Bindings binding;
            for (size_t stage = 0; stage < live_code.size(); ++stage) {
                if (!stages.runtime_infos[stage]) {
                    continue;
                }
                live_code[stage] =
                    EmitPipelineShader(profile, *stages.runtime_infos[stage],
                                       stages.programs[stage + 1], binding,
                                       graphics_key.unique_hashes[stage + 1], {});
                Shader::SerializeInfo(stages.programs[stage + 1].info, live_info[stage]);
                present[stage] = true;
            }
        }
        bool same = true;
        for (size_t stage = 0; stage < present.size(); ++stage) {
            if (present[stage] != (record.infos[stage] != NoInfo)) {
                differ(stage, "stages");
                same = false;
                continue;
            }
            if (!present[stage]) {
                continue;
            }
            const auto [offset, size] = precompiled_info_blobs.at(record.infos[stage]);
            std::vector<u8> kept_info(size);
            precompiled_infos_file.clear();
            precompiled_infos_file.seekg(static_cast<std::streamoff>(offset));
            precompiled_infos_file.read(reinterpret_cast<char*>(kept_info.data()), size);
            if (kept_info != live_info[stage]) {
                differ(stage, "infos");
                same = false;
            }
            if (record.spirv[stage] != NoInfo &&
                ReadKeptSpirv(precompiled_infos_file,
                              precompiled_info_blobs.at(record.spirv[stage])) != live_code[stage]) {
                differ(stage, "SPIR-V words");
                same = false;
            }
        }
        if (same) {
            audit_same.fetch_add(1, std::memory_order_relaxed);
        }
    } catch (const std::exception& exception) {
        differ(0, exception.what());
    }
}

GraphicsPipeline* PipelineCache::ReplaceFailedPrecompiled() {
    precompiled_rejected.insert(graphics_key.Hash());
    precompiled_failed.fetch_add(1, std::memory_order_relaxed);
    if (const auto it = graphics_cache.find(graphics_key); it != graphics_cache.end()) {
        used_this_session.erase(it->second.get());
        evicted_pipelines.emplace_back(scheduler.CurrentTick(), std::move(it->second));
        graphics_cache.erase(it);
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    current_pipeline = nullptr;
    return CurrentGraphicsPipelineSlowPath();
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        flush_in_flight.store(false, std::memory_order_release);
    });
}

void PipelineCache::QueueVulkanPipelineCacheSave() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    // Not skipped when a timed flush is in flight: that one may have read the cache before the
    // stage's last builds. The serialization thread runs the two one after the other.
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "Driver pipeline cache saved after the shader precache, {} bytes",
                 size);
    });
}

namespace {
// EDEN_NX2_PIPELINE_DIFF=1: for a new graphics pipeline whose shaders an earlier one uses too,
// log which parts of the fixed state differ from that one (finds state that splits the same
// draws into many pipelines, each translated on the GPU thread).
void ReportNewPipelineKey(const GraphicsPipelineCacheKey& key) {
    static const bool enabled = std::getenv("EDEN_NX2_PIPELINE_DIFF") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    static std::unordered_map<u64, GraphicsPipelineCacheKey> last_by_shaders;
    static u32 logged{};
    const u64 shaders = Common::CityHash64(reinterpret_cast<const char*>(key.unique_hashes.data()),
                                           sizeof(key.unique_hashes));
    const auto [it, first] = last_by_shaders.try_emplace(shaders, key);
    if (first) {
        return;
    }
    const FixedPipelineState& now = key.state;
    const FixedPipelineState& before = it->second.state;
    std::string diffs;
    if (now.raw1 != before.raw1) diffs += fmt::format(" raw1^{:#x}", now.raw1 ^ before.raw1);
    if (now.raw2 != before.raw2) diffs += fmt::format(" raw2^{:#x}", now.raw2 ^ before.raw2);
    if (now.color_formats != before.color_formats) {
        diffs += fmt::format(" color_formats {:02x} -> {:02x}", fmt::join(before.color_formats, ""),
                             fmt::join(now.color_formats, ""));
    }
    if (now.alpha_test_ref != before.alpha_test_ref) {
        diffs += fmt::format(" alpha_test_ref {:#x} -> {:#x} (func {})", before.alpha_test_ref,
                             now.alpha_test_ref, now.alpha_test_func.Value());
    }
    if (now.point_size != before.point_size) {
        diffs += fmt::format(" point_size {:#x} -> {:#x}", before.point_size, now.point_size);
    }
    if (now.viewport_swizzles != before.viewport_swizzles) diffs += " viewport_swizzles";
    if (now.attribute_types != before.attribute_types) {
        diffs += fmt::format(" attribute_types^{:#x}", now.attribute_types ^ before.attribute_types);
    }
    if (std::memcmp(&now.dynamic_state, &before.dynamic_state, sizeof(now.dynamic_state)) != 0) {
        diffs += " dynamic_state";
    }
    if (now.driver_id != before.driver_id || now.driver_version != before.driver_version) {
        diffs += " driver";
    }
    const size_t size = now.Size();
    if (diffs.empty() && size > offsetof(FixedPipelineState, attachments)) {
        diffs = fmt::format(" past attachments (key size {})", size);
    }
    it->second = key;
    if (logged++ < 400) {
        LOG_INFO(Render_Vulkan, "New pipeline for known shaders {:016x} differs in:{}", shaders,
                 diffs.empty() ? std::string{" nothing in the key prefix"} : diffs);
    }
}

// Without VK_EXT_mesh_shader: logs, once per shader set, what a skipped task/mesh pipeline would
// need from an emulation (task stage, per-primitive outputs, topology, output sizes).
template <typename Programs>
void ReportUnsupportedMeshPipeline(const GraphicsPipelineCacheKey& key, const Programs& programs) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    static u32 with_task{}, per_primitive{}, non_triangle{}, total{};
    u64 shaders{};
    for (const u64 hash : key.unique_hashes) {
        shaders = shaders * 31 + hash;
    }
    std::scoped_lock lock{mutex};
    if (!seen.insert(shaders).second) {
        return;
    }
    const Shader::IR::Program* task{};
    const Shader::IR::Program* mesh{};
    bool has_pixel{};
    for (size_t index = 0; index < programs.size(); ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        const auto& program{programs[index]};
        if (program.stage == Shader::Stage::Task) {
            task = &program;
        } else if (program.stage == Shader::Stage::Mesh) {
            mesh = &program;
        } else if (program.stage == Shader::Stage::Fragment) {
            has_pixel = true;
        }
    }
    ++total;
    if (mesh == nullptr) {
        LOG_WARNING(Render_Vulkan, "Mesh emulation survey: pipeline {:016x} without a mesh stage",
                    shaders);
        return;
    }
    const auto& info{mesh->mesh};
    u32 position{}, generic{}, other{};
    std::string others;
    for (const u32 line : info.vertex_lines) {
        const auto attribute{static_cast<Shader::IR::Attribute>(line)};
        if (attribute >= Shader::IR::Attribute::PositionX &&
            attribute <= Shader::IR::Attribute::PositionW) {
            ++position;
        } else if (Shader::IR::IsGeneric(attribute)) {
            ++generic;
        } else {
            ++other;
            others += fmt::format(" {}", line);
        }
    }
    with_task += task != nullptr;
    per_primitive += !info.primitive_lines.empty();
    non_triangle += info.topology != 4;
    LOG_WARNING(Render_Vulkan,
                "Mesh emulation survey #{}: task={} (payload {} words, wg {}x{}x{}) mesh wg "
                "{}x{}x{} max_vertices={} max_primitives={} topology={} vertex_lines={} "
                "(position {} generic {} other {}:{}) primitive_lines={} pixel={} | totals: "
                "task {} per-primitive {} non-triangle {}",
                total, task != nullptr, info.payload_words, task ? task->workgroup_size[0] : 0,
                task ? task->workgroup_size[1] : 0, task ? task->workgroup_size[2] : 0,
                mesh->workgroup_size[0], mesh->workgroup_size[1], mesh->workgroup_size[2],
                info.max_vertices, info.max_primitives, info.topology, info.vertex_lines.size(),
                position, generic, other, others, info.primitive_lines.size(), has_pixel,
                with_task, per_primitive, non_triangle);
}
} // Anonymous namespace

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    namespace StutterTrace = VideoCommon::StutterTrace;
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        ReportNewPipelineKey(graphics_key);
        if (scheduled_prebuild) {
            pipeline = scheduled_prebuild->Take(graphics_key);
        }
        if (StutterTrace::Enabled()) {
            StutterTrace::CountLookup(pipeline ? StutterTrace::Lookup::GraphicsBackground
                                      : disk_pipelines.contains(graphics_key.Hash())
                                          ? StutterTrace::Lookup::GraphicsRecorded
                                          : StutterTrace::Lookup::GraphicsNew);
        }
        if (!pipeline) {
            const auto keep = evicted_pointer_slots.Take(graphics_key.Hash());
            const u32 unmasked_before = unmasked_generation.load(std::memory_order_acquire);
            pipeline = CreateGraphicsPipeline(keep ? &*keep : nullptr);
            // Dropped because its vertex program loads attributes its header does not list: the
            // program's keys keep every attribute type from now on. When that is this same key
            // (the attributes it dropped had type 0), the next draw must build it again instead
            // of finding this empty entry.
            if (!pipeline && unmasked_generation.load(std::memory_order_acquire) != unmasked_before) {
                graphics_cache.erase(pair);
                return nullptr;
            }
        }
    } else if (StutterTrace::Enabled() && pipeline && !pipeline->WasUsed()) {
        StutterTrace::CountLookup(StutterTrace::Lookup::GraphicsPrebuilt);
    }
    if (!pipeline) {
        return nullptr;
    }
    constexpr auto save_interval = std::chrono::seconds{15};
    if (used_this_session.insert(pipeline.get()).second &&
        std::chrono::steady_clock::now() - used_pipelines_saved > save_interval) {
        SaveUsedPipelines();
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    if (current_pipeline->IsPrecompiled() && current_pipeline->Failed()) {
        return ReplaceFailedPrecompiled();
    }
    current_pipeline->MarkUsed(frame_count);
    return BuiltPipeline(current_pipeline);
}

void PipelineCache::TickFrame() {
    ++frame_count;
    native_wait_budget_us =
        static_cast<s64>(Settings::values.native_pipeline_wait_ms.GetValue()) * 1000;
    while (!evicted_pipelines.empty() && scheduler.IsFree(evicted_pipelines.front().first)) {
        evicted_pipelines.pop_front();
    }
    while (!retired_compute_pipelines.empty() &&
           scheduler.IsFree(retired_compute_pipelines.front().first) &&
           retired_compute_pipelines.front().second->IsBuildTaskDone()) {
        retired_compute_pipelines.pop_front();
    }
    constexpr u64 EvictCheckFrames = 600;
    if (frame_count % EvictCheckFrames == 0) {
        EvictUnusedPipelines();
    }
}

void PipelineCache::EvictUnusedPipelines() {
    // Building a pipeline again on MoltenVK means a Metal compile mid-game, far dearer than the
    // rebuild cost below, so none are evicted there.
    if (device.IsMoltenVK()) {
        return;
    }
    // EDEN_NX2_PRELOAD_ALL_PIPELINES asked for every cached pipeline built: evicting the ones not
    // used yet threw the preload away after 9000 frames (Ocarina: 22,510 of 24,285, then rebuilt
    // one by one in new places).
    if (PreloadAllPipelines()) {
        return;
    }
    // Each built pipeline holds ~0.45 MB of driver memory and none was ever destroyed, so it grew
    // with every place a session went (title 1: 1,500 on its title screen alone). One unused for
    // this long is destroyed; rebuilding it later requires a full GPU-thread shader translation
    // (~8.5 ms each) plus a driver pipeline cache hit.
    // EDEN_NX2_EVICT_AFTER_FRAMES=<frames> changes the limit; 0 keeps every pipeline.
    static const u64 EvictAfterFrames = [] {
        const char* value = std::getenv("EDEN_NX2_EVICT_AFTER_FRAMES");
        return value ? std::strtoull(value, nullptr, 10) : u64{9000};
    }();
    if (EvictAfterFrames == 0 || frame_count <= EvictAfterFrames) {
        return;
    }
    const u64 cutoff = frame_count - EvictAfterFrames;
    const u64 tick = scheduler.CurrentTick();
    // ~0.9 GB of driver memory at ~0.45 MB each.
    constexpr size_t MaxUnusedPrebuiltPipelines = 2000;
    size_t kept_prebuilt = 0;
    size_t evicted = 0;
    for (auto it = graphics_cache.begin(); it != graphics_cache.end();) {
        GraphicsPipeline* const pipeline = it->second.get();
        if (pipeline == nullptr || pipeline == current_pipeline || !pipeline->IsBuilt() ||
            pipeline->LastUse() >= cutoff) {
            ++it;
            continue;
        }
        // A pipeline built from the disk cache at boot that no draw has used yet stays, up to a
        // budget: evicting it throws the prebuild away, and the game translates it again on the
        // GPU thread when it reaches that content (OoT: 1174 of 1346 evicted on the title screen,
        // 684 rebuilt in 10 s on entering the field). Past the budget they go like the rest, so a
        // large cache can't hold their driver memory for the whole session.
        if (!pipeline->WasUsed() && kept_prebuilt < MaxUnusedPrebuiltPipelines) {
            ++kept_prebuilt;
            ++it;
            continue;
        }
        if (used_this_session.erase(pipeline) != 0) {
            used_and_evicted.insert(it->first.Hash());
        }
        evicted_pointer_slots.Remember(it->first.Hash(), pipeline->MappedPointers());
        evicted_pipelines.emplace_back(tick, std::move(it->second));
        it = graphics_cache.erase(it);
        ++evicted;
    }
    if (evicted == 0) {
        return;
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    LOG_INFO(Render_Vulkan, "Evicted {} graphics pipelines unused for {} frames, {} left", evicted,
             EvictAfterFrames, graphics_cache.size());
}

/// Returns `pipeline` when it is built or the current draw must wait for it, and nullptr when the
/// draw may be skipped while it builds (asynchronous shaders; native draws only with
/// EDEN_NX2_ASYNC_DISCARD=1). Each nullptr returned for an unbuilt pipeline is counted for the log.
GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    // EDEN_NX2_ASYNC_DISCARD=1: skip native draws whose pipeline is still building instead of
    // waiting for it (experimental; the draw may be missing for a frame or more).
    /// Reads EDEN_NX2_ASYNC_DISCARD once and logs when it is on.
    static const bool discard_native = [] {
        const char* value = std::getenv("EDEN_NX2_ASYNC_DISCARD");
        const bool enabled = value != nullptr && value[0] == '1';
        if (enabled) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_ASYNC_DISCARD: native draws skip unbuilt pipelines");
        }
        return enabled;
    }();
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    // The environment variable wins over the setting; the setting is read per call.
    NativePendingMode mode = NativePendingMode::Wait;
    if (native) {
        switch (Settings::values.native_async_draws.GetValue()) {
        case Settings::NativeAsyncDraws::Bounded:
            mode = NativePendingMode::Bounded;
            break;
        case Settings::NativeAsyncDraws::Skip:
            mode = NativePendingMode::Skip;
            break;
        default:
            break;
        }
        if (discard_native) {
            mode = NativePendingMode::Skip;
        }
    }
    // Native draws are judged by the heuristic only when a mode other than Wait is chosen.
    const bool can_discard = CanDiscardPendingDraw(
        native, maxwell3d->regs.zeta_enable != 0, draw_state.draw_element_count,
        draw_state.draw_element_count, native && mode != NativePendingMode::Wait);
    bool discard = can_discard;
    if (native) {
        switch (DecideNativePending(mode, can_discard, native_wait_budget_us)) {
        case NativeDrawAction::Wait:
            discard = false;
            break;
        case NativeDrawAction::Skip:
            discard = true;
            break;
        case NativeDrawAction::WaitTimed: {
            const auto begin = std::chrono::steady_clock::now();
            const bool built =
                pipeline->WaitBuilt(std::chrono::microseconds{native_wait_budget_us});
            const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            native_wait_budget_us -= spent;
            native_wait_used_us += spent;
            static auto window = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() - window > std::chrono::seconds{10}) {
                LOG_INFO(Render_Vulkan,
                         "Bounded pipeline wait: {:.1f} ms spent in the last 10 s (budget {} ms "
                         "per frame)",
                         static_cast<double>(native_wait_used_us) / 1000.0,
                         Settings::values.native_pipeline_wait_ms.GetValue());
                native_wait_used_us = 0;
                window = std::chrono::steady_clock::now();
            }
            if (built) {
                return pipeline;
            }
            discard = true;
            break;
        }
        }
    }
    if (discard) {
        static std::atomic<u64> skipped{};
        if ((skipped.fetch_add(1, std::memory_order_relaxed) & 255) == 0) {
            // Counts lookups, not draws: a draw state can look its pipeline up more than once.
            LOG_INFO(Render_Vulkan,
                     "Async shaders: {} pipeline lookups skipped for unbuilt pipelines",
                     skipped.load(std::memory_order_relaxed));
        }
    }
    return discard ? nullptr : pipeline;
}

bool PipelineCache::TranslateGraphicsStages(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                            std::span<Shader::Environment* const> envs,
                                            GraphicsStages& out, bool reference_spirv,
                                            ParallelTranslation* parallel,
                                            std::vector<std::shared_ptr<ShaderPools>>* stage_pools) {
    auto hash = key.Hash();
    std::vector<VideoCommon::StageTranslation> translations;
    std::array<VideoCommon::StageTranslation*, Maxwell::MaxShaderProgram> translated{};
    if (parallel && stage_pools) {
        translations.reserve(envs.size());
        size_t stage_env{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            Shader::Environment& env{*envs[stage_env++]};
            if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
                !device.IsGeometryShaderSupported() && env.IsNativeProgram()) {
                continue; // The pipeline is skipped below before this stage is needed
            }
            translations.push_back({
                .env = &env,
                .pools = stage_pools->emplace_back(TakeSparePools()).get(),
                .cfg_offset = static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader)),
                .exits_to_dispatcher = index == 0,
            });
            translated[index] = &translations.back();
        }
        const auto begin = std::chrono::steady_clock::now();
        VideoCommon::TranslateStages(*shader_jobs, translations, host_info);
        parallel->wait += std::chrono::steady_clock::now() - begin;
        for (const VideoCommon::StageTranslation& translation : translations) {
            parallel->stages += translation.time;
        }
        parallel->count = translations.size();
    }
    size_t env_index{0};
    auto& programs{out.programs};
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};
    bool mesh_unsupported{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;
        if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
            !device.IsGeometryShaderSupported() && env.IsNativeProgram()) {
            // A native (Switch 2) geometry program on a device without geometry shaders: the
            // pipeline cannot run it, so its draws are skipped.
            static std::atomic<bool> logged{};
            if (!logged.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "Skipping native geometry shader pipelines: the device has no geometry shaders");
            }
            return false;
        }
        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline; the draw is
        // skipped (see the pipeline check in the rasterizer). The stages are still translated
        // so the skipped pipeline's needs can be reported below.
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "{} has no VK_EXT_mesh_shader: pipelines with task/mesh shaders are "
                            "skipped, so the geometry they draw (such as foliage and props) will "
                            "be missing",
                            device.GetModelName());
            }
            mesh_unsupported = true;
        }

        const auto translate = [&] {
            if (translated[index]) {
                return translated[index]->Take();
            }
            const u32 cfg_offset{
                static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
            return TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        };
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = translate();
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{translate()};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env, host_info);
        }

        // Canonical native keys keep the attribute types the header's input map lists. A program
        // that loads another one would read it as disabled: this pipeline is dropped and the
        // program's keys keep every attribute type from now on.
        if (index == 1 && key.state.dynamic_vertex_input && env.IsNativeProgram() &&
            env.ShaderStage() == Shader::Stage::VertexB) {
            u32 loads = 0;
            for (u32 attribute = 0; attribute < 32; ++attribute) {
                if (programs[index].info.loads.Generic(attribute)) {
                    loads |= 1U << attribute;
                }
            }
            const u32 listed = VideoCommon::GenericReadMask(env.SPH());
            if ((loads & ~listed) != 0 && !IsUnmaskedVertexProgram(key.unique_hashes[1])) {
                {
                    std::scoped_lock lock{unmasked_mutex};
                    unmasked_vertex_programs.insert(key.unique_hashes[1]);
                    any_unmasked_vertex_program.store(true, std::memory_order_release);
                    unmasked_generation.fetch_add(1, std::memory_order_acq_rel);
                }
                LOG_ERROR(Render_Vulkan,
                          "Vertex program {:016x} loads attributes {:#010x} its header does not "
                          "list ({:#010x}); its pipelines now keep every attribute type",
                          key.unique_hashes[1], loads, listed);
                return false;
            }
        }

        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return false;
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }
    // Mesh emulation without VK_EXT_mesh_shader (prototype): a mesh program, after an optional
    // task program, drawing triangles without per-primitive outputs, before a pixel program only.
    auto& mesh_emulation{out.mesh_emulation};
    size_t mesh_index{};
    if (mesh_unsupported) {
        ReportUnsupportedMeshPipeline(key, programs);
        const bool has_task{key.unique_hashes[1] != 0 &&
                            programs[1].stage == Shader::Stage::Task};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0 && programs[index].stage == Shader::Stage::Mesh) {
                mesh_index = index;
                break;
            }
        }
        bool eligible{mesh_index != 0 && (mesh_index == 1 || has_task) &&
                      layer_source_program == nullptr};
        const Shader::IR::Program& mesh{programs[mesh_index]};
        eligible = eligible && mesh.mesh.topology == 4 && mesh.mesh.primitive_lines.empty() &&
                   !mesh.mesh.vertex_lines.empty() && mesh.mesh.max_vertices != 0 &&
                   mesh.mesh.max_primitives != 0;
        for (size_t index = 1; index < Maxwell::MaxShaderProgram && eligible; ++index) {
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            eligible = emulated || key.unique_hashes[index] == 0 ||
                       programs[index].stage == Shader::Stage::Fragment;
        }
        if (!eligible) {
            return false;
        }
        // The output buffer's binding follows every program's (unified bindings, one per
        // descriptor element, as the SPIR-V backend numbers them). The task and mesh programs
        // share the compute stage, so their descriptors and the emulation's own buffers count
        // against one set of per-stage limits.
        u32 output_binding{};
        u32 compute_storage_buffers{has_task ? 4U : 1U};
        u32 compute_sampled_images{};
        u32 compute_resources{compute_storage_buffers};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            const Shader::Info& info{programs[index].info};
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            const auto add = [&](const auto& descriptors, u32* category) {
                const u32 count{Shader::NumDescriptors(descriptors)};
                output_binding += count;
                if (emulated) {
                    compute_resources += count;
                    if (category) {
                        *category += count;
                    }
                }
            };
            add(info.constant_buffer_descriptors, nullptr);
            add(info.storage_buffers_descriptors, &compute_storage_buffers);
            add(info.texture_buffer_descriptors, &compute_sampled_images);
            add(info.image_buffer_descriptors, nullptr);
            add(info.texture_descriptors, &compute_sampled_images);
            add(info.image_descriptors, nullptr);
        }
        if (compute_storage_buffers > device.GetMaxPerStageDescriptorStorageBuffers() ||
            compute_sampled_images > device.GetMaxPerStageDescriptorSampledImages() ||
            compute_resources > device.GetMaxPerStageResources()) {
            LOG_WARNING(Render_Vulkan,
                        "Mesh emulation skipped: {} storage buffers, {} sampled images and {} "
                        "resources exceed the compute stage limits",
                        compute_storage_buffers, compute_sampled_images, compute_resources);
            return false;
        }
        mesh_emulation.emplace();
        mesh_emulation->output_binding = output_binding;
        if (has_task) {
            // After the output: the task payload, offset and argument buffers.
            mesh_emulation->has_task = true;
            mesh_emulation->mesh_stage = static_cast<u32>(mesh_index - 1);
            mesh_emulation->payload_stride = (std::max)(programs[1].mesh.payload_words, 2U);
            mesh_emulation->payload_binding = output_binding + 1;
            mesh_emulation->offsets_binding = output_binding + 2;
            mesh_emulation->args_binding = output_binding + 3;
            mesh_emulation->distribution_spirv =
                Shader::Backend::SPIRV::EmitMeshEmulationDistribution(profile);
        }
        mesh_emulation->block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh.mesh);
        mesh_emulation->max_primitives = mesh.mesh.max_primitives;
        mesh_emulation->max_vertices = mesh.mesh.max_vertices;
        mesh_emulation->vertex_lines = static_cast<u32>(mesh.mesh.vertex_lines.size());
        for (u32 line = 0; line < mesh.mesh.vertex_lines.size(); ++line) {
            const u32 attribute{mesh.mesh.vertex_lines[line]};
            const u32 position_x{static_cast<u32>(Shader::IR::Attribute::PositionX)};
            if (attribute >= position_x && attribute < position_x + 4) {
                mesh_emulation->position_lines[attribute - position_x] = line;
            }
        }
        static std::atomic<u32> emulated{};
        if (const u32 count{++emulated}; count <= 64) {
            // The input region addresses the mesh program reads (0 is its workgroup index).
            std::string reads;
            for (const Shader::IR::Block* block : mesh.blocks) {
                for (const Shader::IR::Inst& inst : block->Instructions()) {
                    if (inst.GetOpcode() != Shader::IR::Opcode::IsbeRead) {
                        continue;
                    }
                    const Shader::IR::Value address{inst.Arg(0)};
                    reads += address.IsImmediate() ? fmt::format(" {:#x}", address.U32())
                                                   : std::string{" dynamic"};
                }
            }
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation #{}: pipeline {:016x}, {} vertices x {} lines, {} triangles "
                     "per workgroup, output binding {}, task {} (payload {} words), input "
                     "reads:{}",
                     count, hash, mesh.mesh.max_vertices, mesh.mesh.vertex_lines.size(),
                     mesh.mesh.max_primitives, output_binding, has_task,
                     mesh_emulation->payload_stride, reads.empty() ? " none" : reads);
        }
    }
    // A tessellation pipeline the driver would refuse is skipped like a missing mesh pipeline:
    // the draw is dropped (see the pipeline check in the rasterizer) instead of failing the build.
    {
        const Shader::IR::Program* control{};
        const Shader::IR::Program* evaluation{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            if (programs[index].stage == Shader::Stage::TessellationControl) {
                control = &programs[index];
            } else if (programs[index].stage == Shader::Stage::TessellationEval) {
                evaluation = &programs[index];
            }
        }
        if (control != nullptr || evaluation != nullptr) {
            const u32 max_patch_size = device.GetMaxTessellationPatchSize();
            const char* reason = nullptr;
            if (!device.IsTessellationSupported()) {
                reason = "the device has no tessellation shaders";
            } else if (control == nullptr || evaluation == nullptr) {
                reason = "it has only one of the control and evaluation stages";
            } else if (control->invocations == 0 || control->invocations > max_patch_size) {
                reason = "the control shader's output vertex count is out of range";
            }
            if (reason != nullptr) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping tessellation pipeline: {} (tcs={} tes={}, tcs output "
                            "vertices {}, patch control points {}, device max patch size {})",
                            reason, control != nullptr, evaluation != nullptr,
                            control != nullptr ? control->invocations : 0,
                            key.state.patch_control_points_minus_one.Value() + 1, max_patch_size);
                return false;
            }
        }
    }
    // Each stage stays within the limit on its own; together they share one descriptor set.
    if (const u32 set_limit = host_info.max_descriptor_set_storage_buffers; set_limit != 0) {
        size_t storage_buffers{};
        for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
             index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                storage_buffers +=
                    Shader::NumDescriptors(programs[index].info.storage_buffers_descriptors);
            }
        }
        // The fallback binds its output buffer (and, for task draws, payload,
        // offsets and indirect arguments) into the same descriptor set.
        const size_t auxiliary_buffers{
            mesh_emulation ? (mesh_emulation->has_task ? 4U : 1U) : 0U};
        if (!FitsStorageBufferSetLimit(storage_buffers, auxiliary_buffers, set_limit)) {
            throw Shader::NotImplementedException(
                "Binding {} storage buffers ({} for mesh fallback) to a descriptor set limited to {}",
                storage_buffers + auxiliary_buffers, auxiliary_buffers, set_limit);
        }
    }
    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};

        auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        if (mesh_emulation && (program.stage == Shader::Stage::Mesh ||
                               program.stage == Shader::Stage::Task)) {
            runtime_info.mesh_as_compute = true;
            runtime_info.mesh_output_binding = mesh_emulation->output_binding;
            if (mesh_emulation->has_task) {
                runtime_info.mesh_task_payload_binding = mesh_emulation->payload_binding;
                runtime_info.mesh_task_offsets_binding = mesh_emulation->offsets_binding;
                runtime_info.mesh_task_args_binding = mesh_emulation->args_binding;
                runtime_info.mesh_task_payload_stride = mesh_emulation->payload_stride;
            }
            if (program.stage == Shader::Stage::Mesh) {
                mesh_emulation->vertex_spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
                    profile, program.mesh, mesh_emulation->output_binding,
                    mesh_emulation->has_task ? mesh_emulation->args_binding : ~0U,
                    runtime_info.convert_depth_mode && !profile.support_native_ndc);
            }
        }
        ConvertLegacyToGeneric(program, runtime_info);
        // EDEN_NX2_MESH_EMULATION_DUMP=<dir>: write the emulation's vertex shader for spirv-val.
        // (Emitting the mesh program a second time here to dump its compute shader crashed the
        // build that followed, so only the vertex shader is written.)
        if (mesh_emulation && program.stage == Shader::Stage::Mesh) {
            if (const char* const dump{std::getenv("EDEN_NX2_MESH_EMULATION_DUMP")}) {
                const auto& code{mesh_emulation->vertex_spirv};
                std::ofstream file{fmt::format("{}/{:016x}.vert.spv", dump, hash),
                                   std::ios::binary};
                file.write(reinterpret_cast<const char*>(code.data()),
                           static_cast<std::streamsize>(code.size() * sizeof(u32)));
            }
        }
        out.runtime_infos[stage_index] = runtime_info;
        if (reference_spirv && VideoCommon::Nx2Perf::Verify()) {
            out.reference_spirv[stage_index] = EmitSPIRV(profile, runtime_info, program, binding);
        }
        previous_stage = &program;
    }
    return true;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools,
    ParallelTranslation* parallel,
    std::array<std::vector<u32>, Maxwell::MaxShaderStage>* spirv_out) try {
    LOG_DEBUG(Render_Vulkan, "{:#016x}", key.Hash());
    // Declared before the programs, whose IR they hold
    std::vector<std::shared_ptr<ShaderPools>> stage_pools;
    SCOPE_EXIT {
        for (std::shared_ptr<ShaderPools>& stage : stage_pools) {
            ReturnPools(std::move(stage));
        }
    };
    GraphicsStages stages;
    if (!TranslateGraphicsStages(pools, key, envs, stages, emit_pools != nullptr, parallel,
                                 &stage_pools)) {
        return nullptr;
    }
    auto& programs{stages.programs};
    u64 deferred_ir_bytes = 0;
    std::unique_ptr<QueuedIrBudget::Reservation> ir_reservation;
    if (emit_pools) {
        deferred_ir_bytes = IrBytes(programs);
        ir_reservation = queued_ir.TryReserve(deferred_ir_bytes);
    }
    const bool defer_emission = ir_reservation != nullptr;
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    for (size_t stage = 0; stage < infos.size(); ++stage) {
        if (stages.runtime_infos[stage]) {
            infos[stage] = &programs[stage + 1].info;
        }
    }
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    std::function<std::array<vk::ShaderModule, Maxwell::MaxShaderStage>()> emit_stages;
    if (defer_emission) {
        auto job = std::make_shared<DeferredStages>();
        job->reservation = std::move(ir_reservation);
        job->pools = emit_pools;
        job->stage_pools = std::move(stage_pools);
        stage_pools.clear();
        job->programs = std::move(programs);
        job->runtime_infos = std::move(stages.runtime_infos);
        job->reference_spirv = std::move(stages.reference_spirv);
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage]) {
                infos[stage] = &job->programs[stage + 1].info;
            }
        }
        emit_stages = [this, job = std::move(job), hashes = key.unique_hashes]() mutable {
            const auto begin = std::chrono::steady_clock::now();
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> stage_modules;
            Shader::Backend::Bindings stage_binding;
            for (size_t stage = 0; stage < stage_modules.size(); ++stage) {
                if (job->runtime_infos[stage]) {
                    stage_modules[stage] =
                        EmitShaderModule(device, profile, *job->runtime_infos[stage],
                                         job->programs[stage + 1], stage_binding,
                                         hashes[stage + 1], stage, job->reference_spirv[stage]);
                }
            }
            auto completed_pools = std::move(job->pools);
            auto completed_stages = std::move(job->stage_pools);
            job.reset();
            ReturnPools(std::move(completed_pools));
            for (std::shared_ptr<ShaderPools>& stage : completed_stages) {
                ReturnPools(std::move(stage));
            }
            AddRuntimeSpirvTime(1, begin);
            return stage_modules;
        };
    } else {
        Shader::Backend::Bindings binding;
        for (size_t stage = 0; stage < modules.size(); ++stage) {
            if (!stages.runtime_infos[stage]) {
                continue;
            }
            const auto begin = std::chrono::steady_clock::now();
            modules[stage] = EmitShaderModule(device, profile, *stages.runtime_infos[stage],
                                              programs[stage + 1], binding,
                                              key.unique_hashes[stage + 1], stage, {},
                                              spirv_out ? &(*spirv_out)[stage] : nullptr);
            if (build_in_parallel) {
                AddRuntimeSpirvTime(0, begin);
            }
        }
    }
    static_assert(VideoCommon::EvictedPointerSlots::NumPrograms == Maxwell::MaxShaderProgram);
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    for (size_t index = 0, env_index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] != 0) {
            mapped_pointers[index] = envs[env_index++]->MappedStoragePointerKeys();
        }
    }
    auto pipeline = std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, pipeline_binaries.get(),
        &shader_notify, device, descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
        BuildQueue(build_in_parallel, defer_emission ? deferred_ir_bytes : 0), statistics,
        render_pass_cache, key, std::move(modules), infos, std::move(emit_stages),
        std::move(stages.mesh_emulation));
    pipeline->SetMappedPointers(std::move(mapped_pointers));
    return pipeline;

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        try {
            // Walks the program for the dump; the one that failed may fail here again, and an
            // exception leaving this handler would end the process.
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        } catch (const Shader::Exception&) {
        }
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

namespace {
// Pipelines built on the GPU thread at run time (shader translation, and the SPIR-V unless a worker
// emits it; the driver compile is on workers): the first 400 over 2 ms and every one over 100 ms
// are logged, and the totals every 10 s, to tell how much of the stutter on new content is
// translation.
void ReportRuntimePipeline(const char* kind, std::chrono::steady_clock::time_point begin,
                           const std::string& shaders,
                           std::chrono::nanoseconds translation_wait = {},
                           std::chrono::nanoseconds stage_translation = {}) {
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    static std::chrono::steady_clock::duration total{};
    static std::chrono::nanoseconds total_translation_wait{}, total_stage_translation{};
    static u32 count{}, logged{};
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - begin;
    total += elapsed;
    total_translation_wait += translation_wait;
    total_stage_translation += stage_translation;
    ++count;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    // Slow builds are always logged with their shaders, to find the programs that are slow to
    // translate. EDEN_NX2_PIPELINE_LOG_LIMIT=<n> changes how many of the others are.
    static const u32 limit = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_LOG_LIMIT");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 400U;
    }();
    if (elapsed > std::chrono::milliseconds{100} ||
        (elapsed > std::chrono::milliseconds{2} && logged < limit)) {
        ++logged;
        LOG_INFO(Render_Vulkan, "{} pipeline built on the GPU thread in {:.1f} ms: {}", kind,
                 ms(elapsed), shaders);
    }
    if (now - window > std::chrono::seconds{10}) {
        const auto take = [](auto& counter) {
            return std::chrono::nanoseconds{counter.exchange(0, std::memory_order::relaxed)};
        };
        // Shader cache v2: the GPU thread's wait for graphics stages translated at once on
        // workers, the translation time they took together, and compute SPIR-V on workers
        const std::string v2 = fmt::format(
            ", {:.1f} ms on the GPU thread waiting for {:.1f} ms of graphics "
            "stage translation on workers, {:.1f} ms compute SPIR-V on workers",
            ms(total_translation_wait), ms(total_stage_translation), ms(take(runtime_spirv_ns[2])));
        LOG_INFO(Render_Vulkan,
                 "Runtime pipelines in the last 10 s: {} built, {:.1f} ms on the GPU thread "
                 "({:.1f} ms of it graphics SPIR-V), {:.1f} ms graphics SPIR-V on workers, "
                 "{:.1f} ms of draws held for unbuilt pipelines{}",
                 count, ms(total), ms(take(runtime_spirv_ns[0])), ms(take(runtime_spirv_ns[1])),
                 ms(GraphicsPipeline::TakeHoldTime()), v2);
        std::string reasons;
        {
            std::scoped_lock reasons_lock{precompiled_reasons_mutex};
            for (const auto& [reason, n] : precompiled_reasons) {
                reasons += fmt::format(", {} {}", n, reason);
            }
            precompiled_reasons.clear();
        }
        LOG_INFO(Render_Vulkan,
                 "Precompiled pipelines in the last 10 s: {} created from kept infos ({} of them "
                 "from stage infos, {:.1f} ms on the GPU thread), {} built from kept SPIR-V, {} "
                 "translated on workers from their recorded environments, {} failed and "
                 "replaced{}",
                 precompiled_created.exchange(0, std::memory_order::relaxed),
                 precompiled_per_stage.exchange(0, std::memory_order::relaxed),
                 ms(std::chrono::nanoseconds{
                     precompiled_gpu_ns.exchange(0, std::memory_order::relaxed)}),
                 precompiled_from_spirv.exchange(0, std::memory_order::relaxed),
                 precompiled_drawn.exchange(0, std::memory_order::relaxed),
                 precompiled_failed.exchange(0, std::memory_order::relaxed), reasons);
        if (PrecompiledAudit()) {
            LOG_INFO(Render_Vulkan,
                     "Precompiled audit in the last 10 s: {} pipelines translated from the game's "
                     "state gave the kept SPIR-V and infos, {} differed",
                     audit_same.exchange(0, std::memory_order::relaxed),
                     audit_differ.exchange(0, std::memory_order::relaxed));
        }
        total = {};
        total_translation_wait = {};
        total_stage_translation = {};
        count = 0;
        window = now;
    }
}
} // Anonymous namespace

void PipelineCache::WarnKeptPointer(size_t program, u64 slot) const {
    if ((slot >> 63) != 0) {
        return;
    }
    const u32 index = static_cast<u32>(slot >> 32);
    const u32 offset = static_cast<u32>(slot);
    const size_t stage = program == 0 ? 0 : program - 1;
    const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
    GPUVAddr value = 0;
    const char* reason = nullptr;
    if (index >= cbufs.size() || !cbufs[index].enabled ||
        offset + sizeof(u64) > cbufs[index].size ||
        !gpu_memory->IsFullyMappedRange(cbufs[index].address + offset, sizeof(u64))) {
        reason = "constant buffer not bound";
    } else {
        value = gpu_memory->Read<u64>(cbufs[index].address + offset);
        if (value == 0) {
            reason = "null";
        } else if (!gpu_memory->IsWithinGPUAddressRange(value)) {
            reason = "outside the GPU address range";
        } else if (!gpu_memory->IsFullyMappedRange(value, 1)) {
            reason = "unmapped";
        }
    }
    if (!reason) {
        return;
    }
    static std::atomic<u64> kept{};
    const u64 count = kept.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 32 || count % 1000 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "Graphics pipeline {:016x} translated again after eviction: s{}:c{}[{:#x}] "
                    "holds {:#x} ({}), kept as the storage buffer pointer it was ({} so far)",
                    graphics_key.Hash(), program, index, offset, value, reason, count);
    }
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    const VideoCommon::EvictedPointerSlots::Slots* keep_pointers) {
    const VideoCommon::Nx2Profile::Scope profile_scope{
        VideoCommon::Nx2Profile::Zone::GraphicsPipelineMiss};
    const auto begin = std::chrono::steady_clock::now();
    std::optional<ParallelTranslation> parallel;
    SCOPE_EXIT {
        std::string shaders = fmt::format("pipeline={:016x}", graphics_key.Hash());
        for (size_t stage = 0; stage < graphics_key.unique_hashes.size(); ++stage) {
            if (graphics_key.unique_hashes[stage] != 0) {
                shaders += fmt::format(" s{}:{:016x}", stage, graphics_key.unique_hashes[stage]);
            }
        }
        if (!parallel) {
            ReportRuntimePipeline("Graphics", begin, shaders);
            return;
        }
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        shaders += fmt::format(" translate_wait_ms={:.1f} stage_translate_ms={:.1f} stages={}",
                               ms(parallel->wait), ms(parallel->stages), parallel->count);
        ReportRuntimePipeline("Graphics", begin, shaders, parallel->wait, parallel->stages);
    };
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    if (keep_pointers) {
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (graphics_key.unique_hashes[index] == 0) {
                continue;
            }
            environments.envs[index].KeepStoragePointers((*keep_pointers)[index]);
            for (const u64 slot : (*keep_pointers)[index]) {
                WarnKeptPointer(index, slot);
            }
        }
    }

    const auto precompiled_begin = std::chrono::steady_clock::now();
    std::optional<PrecompiledRecord> precompiled_record;
    auto pipeline{CreatePrecompiledGraphicsPipeline(environments, &precompiled_record)};
    if (pipeline) {
        precompiled_created.fetch_add(1, std::memory_order_relaxed);
        precompiled_gpu_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                 precompiled_begin)
                .count(),
            std::memory_order_relaxed);
        if (PrecompiledAudit()) {
            AuditPrecompiled(environments.Span(), false, graphics_key.Hash(),
                             precompiled_record ? &*precompiled_record : nullptr);
        }
        // As for a translated pipeline: the storage pointers the game's state answered as mapped
        // (it answered every question the recorded environments were asked), kept if the
        // pipeline is evicted and translated again.
        VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (graphics_key.unique_hashes[index] != 0) {
                mapped_pointers[index] = environments.envs[index].MappedStoragePointerKeys();
            }
        }
        pipeline->SetMappedPointers(std::move(mapped_pointers));
    } else {
        // Shader cache v2: two or more stages translate at once on the shader jobs' CPU workers
        // while this thread waits. Only this thread may flush guest memory, as reading a texture
        // descriptor does, so the descriptor pool is flushed here first; with this thread waiting
        // nothing can make it dirty again. A pool too large to flush on every miss translates here
        // as before.
        if (std::ranges::count_if(graphics_key.unique_hashes,
                                  [](u64 hash) { return hash != 0; }) >= 2) {
            constexpr u64 MaxFlushedTexturePoolBytes = u64{1} << 20;
            const auto& tex_header = maxwell3d->regs.tex_header;
            const u64 pool_bytes =
                (u64{tex_header.limit} + 1) * sizeof(Tegra::Texture::TICEntry);
            if (pool_bytes <= MaxFlushedTexturePoolBytes) {
                gpu_memory->FlushRegion(tex_header.Address(), pool_bytes);
                for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
                    if (graphics_key.unique_hashes[index] != 0) {
                        environments.envs[index].SetTexturePoolFlushed();
                    }
                }
                parallel.emplace();
            } else {
                static std::atomic<bool> logged{};
                if (!logged.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Vulkan,
                             "Texture descriptor pool of {} bytes: pipelines that use it "
                             "translate on the GPU thread",
                             pool_bytes);
                }
            }
        }

        // Native programs emit their SPIR-V in the pipeline's job instead of here
        std::shared_ptr<ShaderPools> emit_pools;
        if (maxwell3d->GetNativeMacroEngine().native_turing) {
            emit_pools = TakeSparePools();
        } else {
            main_pools.ReleaseContents();
        }
        ShaderPools& pools = emit_pools ? *emit_pools : main_pools;
        pipeline = CreateGraphicsPipeline(pools, graphics_key, environments.Span(), nullptr, true,
                                          std::move(emit_pools), parallel ? &*parallel : nullptr);
    }
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    if (disk_pipelines.contains(graphics_key.Hash())) {
        // In the disk cache already, only not built at boot.
        QueueVulkanPipelineCacheFlush();
        return pipeline;
    }
    disk_pipelines.insert(graphics_key.Hash());
    serialization_thread.QueueWork([this, key = graphics_key, envs = std::move(environments.envs)] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        const std::vector<u8> payload{VideoCommon::SerializePipelinePayload(
            std::span(reinterpret_cast<const char*>(&key), sizeof(key)), env_ptrs,
            host_info.EnvironmentReadProperties())};
        if (!payload.empty()) {
            cache_v2->Writer().Append(VideoCommon::ShaderCacheV2::RecordKind::Graphics,
                                      GraphicsRecordKey(key), payload);
        }
    });
    QueueVulkanPipelineCacheFlush();
    return pipeline;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                               const ShaderInfo* shader,
                                                               std::span<const u64> keep_pointers) {
    const VideoCommon::Nx2Profile::Scope profile_scope{
        VideoCommon::Nx2Profile::Zone::ComputePipelineMiss};
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        ReportRuntimePipeline("Compute", begin,
                              fmt::format("pipeline={:016x} s6:{:016x} address={:#x}", key.Hash(),
                                          key.unique_hash,
                                          kepler_compute->regs.code_loc.Address() +
                                              kepler_compute->launch_description.program_start));
    };
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    env.KeepStoragePointers(keep_pointers);

    // A rebuild too (keep_pointers): its environment answers the kept pointers as pointers before
    // anything is asked, as its translation would see them.
    ComputePipelineCacheEntry entry;
    {
        const auto precompiled_begin = std::chrono::steady_clock::now();
        const PrecompiledRecord* matched = nullptr;
        entry = CreatePrecompiledComputePipeline(key, env, &matched);
        if (entry.pipeline && PrecompiledAudit()) {
            Shader::Environment* const env_ptr = &env;
            AuditPrecompiled(std::span(&env_ptr, 1), true, key.Hash(), matched);
        }
        if (entry.pipeline) {
            precompiled_created.fetch_add(1, std::memory_order_relaxed);
            precompiled_gpu_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - precompiled_begin)
                                             .count(),
                                         std::memory_order_relaxed);
        }
    }
    if (!entry.pipeline) {
        // The SPIR-V is emitted by the pipeline's job instead of here
        std::shared_ptr<ShaderPools> emit_pools = TakeSparePools();
        ShaderPools& pools = *emit_pools;
        entry = CreateComputePipeline(pools, key, env, nullptr, true, std::move(emit_pools));
    }
    if (const char* directory = std::getenv("EDEN_NX2_SHADER_DUMP");
        directory && !entry.pipeline && env.CanBeSerialized()) {
        const auto path = std::filesystem::path{directory} /
                          fmt::format("failed-compute-{:016x}-v{}.env", key.unique_hash, CACHE_VERSION);
        std::ofstream file{path, std::ios::binary};
        if (file) {
            env.Serialize(file, host_info.EnvironmentReadProperties());
            LOG_INFO(Render_Vulkan, "Captured failed compute environment: {} written={}",
                     path.string(), file.good());
        }
    }
    if (!entry.pipeline || pipeline_cache_filename.empty()) {
        return entry;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        const std::array<const GenericEnvironment*, 1> envs{&env_};
        const std::vector<u8> payload{VideoCommon::SerializePipelinePayload(
            std::span(reinterpret_cast<const char*>(&key), sizeof(key)), envs,
            host_info.EnvironmentReadProperties())};
        if (!payload.empty()) {
            cache_v2->Writer().Append(VideoCommon::ShaderCacheV2::RecordKind::Compute,
                                      ComputeRecordKey(payload), payload);
        }
    });
    QueueVulkanPipelineCacheFlush();
    return entry;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel,
    std::shared_ptr<ShaderPools> emit_pools, std::vector<u32>* spirv_out) try {
    SCOPE_EXIT {
        if (emit_pools) {
            ReturnPools(std::move(emit_pools));
        }
    };
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return {.failure = "Device reports broken compute support"};
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY || driver_id == VK_DRIVER_ID_MOLTENVK;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    std::unique_ptr<ComputePipeline> pipeline;
    const u64 ir_bytes = emit_pools ? IrBytes(std::span(&program, 1)) : 0;
    if (auto ir_reservation = emit_pools ? queued_ir.TryReserve(ir_bytes) : nullptr) {
        auto job = std::make_shared<DeferredCompute>();
        if (VideoCommon::Nx2Perf::Verify()) {
            job->reference_spirv = EmitComputeShader(profile, program, key.unique_hash);
        }
        job->reservation = std::move(ir_reservation);
        job->pools = std::move(emit_pools);
        job->program = std::move(program);
        // Copied by the pipeline before its job can run
        const Shader::Info& info = job->program.info;
        auto emit = [this, job, shader_hash = key.unique_hash]() mutable {
            const auto begin = std::chrono::steady_clock::now();
            vk::ShaderModule module{EmitComputeModule(device, profile, job->program, shader_hash,
                                                      job->reference_spirv)};
            auto completed_pools = std::move(job->pools);
            job.reset();
            ReturnPools(std::move(completed_pools));
            AddRuntimeSpirvTime(2, begin);
            return module;
        };
        pipeline = std::make_unique<ComputePipeline>(
            device, scheduler, vulkan_pipeline_cache, pipeline_binaries.get(), hash,
            descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
            BuildQueue(build_in_parallel, ir_bytes), statistics,
            &shader_notify, info, vk::ShaderModule{}, key.unique_hash, std::move(emit));
    } else {
        vk::ShaderModule module{
            EmitComputeModule(device, profile, program, key.unique_hash, {}, spirv_out)};
        const Shader::Info info{std::move(program.info)};
        if (!build_in_parallel) {
            // The driver compiles on this thread. Precache and boot workers would otherwise hold their
            // IR through the compile, up to hundreds of MiB for one large shader.
            program = {};
            pools.block.Free();
            pools.inst.Free();
        }
        pipeline = std::make_unique<ComputePipeline>(
            device, scheduler, vulkan_pipeline_cache, pipeline_binaries.get(), hash,
            descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring,
            BuildQueue(build_in_parallel), statistics, &shader_notify, info, std::move(module),
            key.unique_hash);
    }
    return {.pipeline = std::move(pipeline),
            .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                        : std::vector<std::pair<u32, u32>>{},
            .mapped_pointers =
                env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{}};

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Compute pipeline compilation failed shader={:#016x} pipeline={:#016x}: {}",
              key.unique_hash, key.Hash(), exception.what());
    return {.failure = exception.what()};
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) {
    // Written beside the cache and renamed over it: the emulator stopped mid-write left the cache
    // empty, and every later boot failed to read it.
    auto temporary = filename;
    temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(temporary));
        return;
    }
    try {
        file.exceptions(std::ifstream::failbit);
        file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
            .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

        size_t cache_size = 0;
        std::vector<char> cache_data;
        if (pipeline_cache) {
            pipeline_cache.Read(&cache_size, nullptr);
            cache_data.resize(cache_size);
            pipeline_cache.Read(&cache_size, cache_data.data());
        }
        file.write(cache_data.data(), cache_size);
        file.close();
    } catch (const std::ios_base::failure& e) {
        // The cache stays as it was: only the file being written goes.
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        file.exceptions(std::ios::goodbit);
        file.close();
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return;
    }
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        LOG_ERROR(Common_Filesystem, "Failed to replace Vulkan driver pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), error.message());
        std::filesystem::remove(temporary, error);
        return;
    }

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        // A file cut short (the emulator stopped while writing it) would throw from the reads
        // below, and macOS's libc++ throws an ios_base::failure the handler doesn't match.
        if (static_cast<size_t>(end) < magic_number.size() + sizeof(cache_version)) {
            file.close();
            LOG_ERROR(Common_Filesystem, "Truncated Vulkan driver pipeline cache file, deleting it");
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan,
                 "Loaded Vulkan driver pipeline cache: {}", Common::FS::PathToUTF8String(filename));

        try {
            return create_pipeline_cache(cache_size, cache_data.data());
        } catch (const vk::Exception& e) {
            // The blob came from another driver or driver version (e.g. switching between
            // MoltenVK and KosmicKrisp on macOS): the driver rejects it. Start empty instead of
            // aborting, and drop the stale file so the next run doesn't retry it.
            LOG_ERROR(Render_Vulkan,
                      "Driver rejected the Vulkan pipeline cache file ({}), starting with an empty "
                      "cache and deleting it",
                      e.what());
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
