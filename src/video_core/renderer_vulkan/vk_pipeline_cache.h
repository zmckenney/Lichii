// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_prebuild.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/shader_jobs.h"
#include "video_core/shader_stage_translation.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace VideoCommon::ShaderCacheV2 {
struct LogFile;
struct ProgramRecord;
} // namespace VideoCommon::ShaderCacheV2

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

using VideoCommon::ShaderPools;

struct ComputePipelineCacheEntry {
    std::unique_ptr<ComputePipeline> pipeline;
    std::string failure;
    /// Constant buffer slots that were null pointers at translation (see NullStoragePointerSlots)
    std::vector<std::pair<u32, u32>> null_pointer_slots;
    /// Constant buffer slots bound as storage buffer pointers (see MappedStoragePointerKeys)
    std::vector<u64> mapped_pointers;
    u32 rebuilds{};
    /// Whether a dispatch looked it up yet (kept only with EDEN_NX2_STUTTER_TRACE)
    bool used{};

    [[nodiscard]] ComputePipeline* Get(bool native, GPUVAddr address,
                                       const ComputePipelineCacheKey& key) const;
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    void TickFrame();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline(u64& native_key);

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

private:
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    /// Returns `pipeline` when it is built or its draw must wait for it, nullptr when the draw
    /// may be skipped while it builds (asynchronous shaders, CanDiscardPendingDraw).
    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) noexcept;

    /// With `keep_pointers`, the slots an evicted pipeline of this key bound stay storage buffer
    /// pointers whatever they hold now.
    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        const VideoCommon::EvictedPointerSlots::Slots* keep_pointers = nullptr);

    /// Logs a kept pointer slot (an index of keep_pointers) that this translation would otherwise
    /// have classified as no pointer: null, unmapped, or out of range right now.
    void WarnKeptPointer(size_t program, u64 slot) const;

    /// Shader cache v2, run-time misses: the stages translated at once on shader_jobs' CPU workers
    /// while the GPU thread waited (VideoCommon::TranslateStages)
    struct ParallelTranslation {
        std::chrono::nanoseconds wait{};   ///< On the GPU thread
        std::chrono::nanoseconds stages{}; ///< The stages' translations together
        size_t count{};
    };

    struct GraphicsStages;
    /// Translates a graphics pipeline's programs and works out each stage's runtime info (the
    /// work before SPIR-V emission); false when the pipeline is skipped, its draws dropped. With
    /// `parallel` and `stage_pools`, the stages translate at once on shader_jobs, each in IR pools
    /// added to `stage_pools`, which must outlive `out`.
    bool TranslateGraphicsStages(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                 std::span<Shader::Environment* const> envs, GraphicsStages& out,
                                 bool reference_spirv, ParallelTranslation* parallel = nullptr,
                                 std::vector<std::shared_ptr<ShaderPools>>* stage_pools = nullptr);

    /// Builds the pipeline as a job of shader_jobs with build_in_parallel, else leaves it for the
    /// caller's Build() (see BuildQueue). With `parallel` the stages translate at once on
    /// shader_jobs, each in IR pools of its own; their environments must be safe to read from those
    /// threads.
    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools = nullptr,
        ParallelTranslation* parallel = nullptr,
        std::array<std::vector<u32>, 5>* spirv_out = nullptr);

    /// Released IR pools for a build, or new ones
    [[nodiscard]] std::shared_ptr<ShaderPools> TakeSparePools();
    /// Releases the pools' contents and keeps a few for later builds
    void ReturnPools(std::shared_ptr<ShaderPools> pools);

    [[nodiscard]] bool StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const;

    /// Translates the current compute program, binding the pointers in keep_pointers whatever
    /// their slots hold now (see GenericEnvironment::KeepStoragePointers)
    ComputePipelineCacheEntry CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                    const ShaderInfo* shader,
                                                    std::span<const u64> keep_pointers = {});

    /// Builds the pipeline as CreateGraphicsPipeline does. With `emit_pools`, which `pools` must
    /// be, the SPIR-V is emitted by the CPU stage of the pipeline's job, which takes the pools and
    /// the program.
    ComputePipelineCacheEntry CreateComputePipeline(ShaderPools& pools,
                                                     const ComputePipelineCacheKey& key,
                                                     Shader::Environment& env,
                                                     PipelineStatistics* statistics,
                                                     bool build_in_parallel,
                                                     std::shared_ptr<ShaderPools> emit_pools = nullptr,
                                                     std::vector<u32>* spirv_out = nullptr);

    /// Where CreateGraphicsPipeline and CreateComputePipeline build; `cpu_bytes` estimates the
    /// IR a deferred emission holds
    [[nodiscard]] PipelineBuildQueue BuildQueue(bool build_in_parallel, u64 cpu_bytes = 0);

    /// Whether the device can build a cached graphics pipeline of this key: its dynamic state
    /// and provoking vertex must match the device's
    [[nodiscard]] bool KeyFitsDevice(const GraphicsPipelineCacheKey& key) const;

    /// The boot load of the v2 cache (LoadDiskResources): `log` read, its records built as jobs
    /// of shader_jobs
    void LoadPipelinesV2(u64 title_id, std::shared_ptr<const VideoCommon::ShaderCacheV2::LogFile> log,
                         std::stop_token stop_loading,
                         const VideoCore::DiskResourceLoadCallback& callback,
                         std::chrono::steady_clock::time_point load_begin, bool precache,
                         std::unordered_set<u64> precached_pipelines);

    /// Reads a v2 graphics record and translates its pipeline, left for Build(). Null when it
    /// doesn't translate, `incomplete` when the record lacks a value the translation reads.
    /// Throws std::ios_base::failure for a record it can't read.
    std::unique_ptr<GraphicsPipeline> TranslateCachedGraphics(std::span<const u8> payload,
                                                              const GraphicsPipelineCacheKey& key,
                                                              PipelineStatistics* statistics,
                                                              bool& incomplete);

    /// Cancels the v2 prebuild and waits for its running stages
    void StopBackgroundJobs();
    struct PrecompiledRecord;

    /// The compute counterpart of CreatePrecompiledGraphicsPipeline: an entry without a
    /// pipeline when the key is not precompiled or its answers differ. A key may have several
    /// records (storage pointer states): the first whose answers hold is used (set to `matched`).
    ComputePipelineCacheEntry CreatePrecompiledComputePipeline(
        const ComputePipelineCacheKey& key, VideoCommon::ComputeEnvironment& env,
        const PrecompiledRecord** matched = nullptr);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    void QueueVulkanPipelineCacheFlush();

    /// Queues one write of the driver pipeline cache file on the serialization thread, with no
    /// throttle. For the end of a precache: its builds never reach the timed flush, which counts
    /// only pipelines built during play.
    void QueueVulkanPipelineCacheSave();

    /// One build of the shader precache: a pipeline built and destroyed
    struct PrecacheJob {
        u64 graphics_hash{}; ///< The cached graphics pipeline it builds, 0 for a compute program
        /// Translates the pipeline and returns its build, which is true when it built (the job's
        /// CPU and driver stages)
        Common::UniqueFunction<Common::UniqueFunction<bool>> translate;
        u64 cpu_bytes{};
    };

    /// Shader precache before the game starts (precache_shaders), on every CPU thread: the cached
    /// graphics pipelines this boot skipped (`jobs`; `precached` holds those built by earlier
    /// precaches) and the compute programs of the title's precache file (made from the game files,
    /// missing reads guessed) are built, then thrown away. They only fill the driver's shader
    /// cache, so a later build of the same pipeline finds the driver's work done. Reports progress
    /// through `callback` and returns when it is done, skipped from the loading screen or the load
    /// is stopped.
    void PrecacheShaders(u64 title_id, std::vector<PrecacheJob> jobs,
                         std::unordered_set<u64> precached, std::stop_token stop_loading,
                         const VideoCore::DiskResourceLoadCallback& callback);

    /// Builds and drops every pipeline of the game's precompiled file (written offline by
    /// tools/nx2-precompile), so the driver's own cache holds them before they are first drawn.
    /// Draws never use these pipelines: they build their own from the guest's shaders and find the
    /// driver's compiled code. Runs once per precompiled file, driver and emulator build.
    void PreparePrecompiled(const std::filesystem::path& base_dir, std::stop_token stop_loading,
                            const VideoCore::DiskResourceLoadCallback& callback);

    /// The shader infos and SPIR-V the preparation step kept for the precompiled pipelines
    /// (precompiled_infos.fork-v<N>.bin, written for the file, driver and build of `stamp`), and
    /// the precompiled file's environment index; false when there are none for this stamp.
    bool LoadPrecompiledInfos(const std::filesystem::path& precompiled,
                              const std::filesystem::path& infos, const std::string& stamp);

    /// A precompiled pipeline for the current key, drawn with the shader infos the preparation
    /// step kept while a worker builds it from the SPIR-V it kept (or, without it, translates
    /// the recorded environments): the GPU thread checks that the game's state now answers the
    /// recorded environments' questions as they were answered (FileEnvironment::AnswersHoldFor),
    /// so the kept translation is the one the game's state would give, instead of translating.
    /// A key that was not precompiled, or whose answers differ, is looked up per stage
    /// (CreatePerStageGraphicsPipeline). Null when neither gives one. `matched` is set to the
    /// record the pipeline was made from.
    std::unique_ptr<GraphicsPipeline> CreatePrecompiledGraphicsPipeline(
        GraphicsEnvironments& environments,
        std::optional<PrecompiledRecord>* matched = nullptr);

    /// A pipeline whose key was not precompiled, from each stage's kept info: the first of the
    /// program's recorded environments (from any precompiled pipeline) whose answers the game's
    /// state gives. A stage's translation depends only on its program and environment (stages
    /// whose legacy attribute loads follow the previous stage are not kept per stage), so the
    /// infos hold; a worker translates the recorded environments with this key (the SPIR-V
    /// depends on the key), and a translation that gives other infos fails the build, which is
    /// then replaced (ReplaceFailedPrecompiled). Null unless every stage is found, with `reason`
    /// set to why (empty when found).
    std::unique_ptr<GraphicsPipeline> CreatePerStageGraphicsPipeline(
        GraphicsEnvironments& environments, std::optional<PrecompiledRecord>* matched,
        std::string_view& reason);

    /// The pipeline drawn with `infos`, built by a worker from the kept SPIR-V blobs or, without
    /// them, by translating the recorded environments.
    std::unique_ptr<GraphicsPipeline> MakePrecompiledGraphicsPipeline(
        std::shared_ptr<std::vector<VideoCommon::FileEnvironment>> recorded,
        const std::array<Shader::Info, 5>& infos, const std::array<bool, 5>& present,
        const std::optional<std::array<std::pair<u64, u32>, 5>>& spirv_blobs);

    /// Opens the precompiled file and the kept infos for reading, on the GPU thread.
    void OpenPrecompiledFiles();

    /// EDEN_NX2_PRECOMPILED_AUDIT=1: a precompiled pipeline just created is also translated from
    /// the game's state, on the GPU thread, and its SPIR-V and infos compared with the kept ones
    /// (counted in the 10 s report, the first differences logged). Slow: a check of the
    /// relational answers (FileEnvironment::AnswersHoldFor), not for play. A compute pipeline
    /// passes the record it was made from.
    void AuditPrecompiled(std::span<Shader::Environment* const> envs, bool compute,
                          u64 key_hash, const PrecompiledRecord* record = nullptr);

    /// The current pipeline was precompiled and its build failed (its translation did not give
    /// the infos it was drawn with): replaced by one translated from the game's state.
    GraphicsPipeline* ReplaceFailedPrecompiled();

    struct PrecompiledRecord {
        std::array<u32, 5> envs{};  ///< The precompiled file's environments, in stage order
        u8 env_count{};
        std::array<u32, 5> infos{}; ///< Per pipeline stage: an info blob, ~0U for none
        std::array<u32, 5> spirv{}; ///< Per pipeline stage: a SPIR-V blob (zstd), ~0U for none
    };
    std::unordered_map<u64, PrecompiledRecord> precompiled_records; ///< By key hash
    struct PrecompiledStage {
        u8 stage{};  ///< Pipeline stage
        u32 env{};   ///< The precompiled file's environment
        u32 info{};  ///< An info blob
    };
    /// Graphics stages by program hash, one for each recorded environment the program was kept in
    std::unordered_multimap<u64, PrecompiledStage> precompiled_stages;
    /// By key hash; a compute key has a record for each storage pointer state it was prepared in
    std::unordered_multimap<u64, PrecompiledRecord> precompiled_compute_records;
    std::vector<std::pair<u64, u64>> precompiled_env_index;         ///< Offset, size
    std::vector<std::pair<u64, u32>> precompiled_info_blobs;        ///< Offset, size (all blobs)
    std::filesystem::path precompiled_path;
    std::filesystem::path precompiled_infos_path;
    std::ifstream precompiled_file;       ///< GPU thread only
    std::ifstream precompiled_infos_file; ///< GPU thread only
    std::unordered_set<u64> precompiled_rejected; ///< Keys whose precompiled build failed

    /// Native pipelines: the key without state that cannot change the pipeline (see
    /// FixedPipelineState::CanonicalizeNative).
    void CanonicalizeNativeKey(GraphicsPipelineCacheKey& key,
                               std::span<const Shader::Environment* const> envs) const noexcept;

    /// Native vertex programs found to load attributes their header does not list: their keys
    /// keep every attribute type.
    bool IsUnmaskedVertexProgram(u64 hash) const {
        if (!any_unmasked_vertex_program.load(std::memory_order_acquire)) {
            return false;
        }
        std::scoped_lock lock{unmasked_mutex};
        return unmasked_vertex_programs.contains(hash);
    }
    mutable std::mutex unmasked_mutex;
    std::unordered_set<u64> unmasked_vertex_programs;
    std::atomic<bool> any_unmasked_vertex_program{};
    std::atomic<u32> unmasked_generation{}; ///< Programs added to unmasked_vertex_programs

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};
    /// The pipelines' builds, the prebuild and the precache. Declared before the pipelines, whose
    /// builds it runs.
    std::optional<VideoCommon::ShaderJobScheduler> shader_jobs;
    /// IR of the pipelines whose job emits their SPIR-V
    QueuedIrBudget queued_ir{VideoCommon::DefaultShaderJobBudget()};
    std::optional<ScheduledPrebuild<GraphicsPipelineCacheKey, GraphicsPipeline>> scheduled_prebuild;

    GraphicsPipelineCacheKey graphics_key{};
    /// Native titles: the state as the registers give it. FixedPipelineState refreshes some fields
    /// only when they are dirty, so graphics_key.state is a canonical copy made for each draw.
    FixedPipelineState native_state{};
    GraphicsPipeline* current_pipeline{};

    /// Destroys graphics pipelines unused for a while, once the GPU is done with them.
    void EvictUnusedPipelines();
    u64 frame_count{};
    /// Wait budget of native draws for unbuilt pipelines left in this frame, in microseconds
    /// (native_async_draws = Bounded). Only touched on the GPU thread; reset in TickFrame.
    s64 native_wait_budget_us{8000};
    s64 native_wait_used_us{}; ///< Budget spent since the last 10 s log line.
    std::deque<std::pair<u64, std::unique_ptr<GraphicsPipeline>>> evicted_pipelines; ///< (tick, pipeline)
    std::unordered_set<u64> used_and_evicted; ///< Key hashes of pipelines used this session, evicted since
    VideoCommon::EvictedPointerSlots evicted_pointer_slots;

    ::Common::unordered_map<ComputePipelineCacheKey, ComputePipelineCacheEntry> compute_cache;
    /// Replaced compute pipelines, destroyed once the GPU and their queued build are done with
    /// them (tick, pipeline)
    std::deque<std::pair<u64, std::unique_ptr<ComputePipeline>>> retired_compute_pipelines;
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    ShaderPools main_pools;
    std::mutex spare_pools_mutex;
    std::vector<std::shared_ptr<ShaderPools>> spare_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    /// Key hashes of the graphics pipelines in the disk cache. Only those a recent session used
    /// (used_pipelines) are built at boot; the rest are built when first needed, and not stored
    /// again. Each built pipeline holds driver memory (title 1: 6.5k cached, a quarter used).
    std::unordered_set<u64> disk_pipelines;
    std::unordered_map<u64, u8> used_pipelines; ///< Key hash -> sessions since it was last used
    std::unordered_set<const GraphicsPipeline*> used_this_session;
    std::chrono::steady_clock::time_point used_pipelines_saved{};
    void SaveUsedPipelines();

    /// The cache pipeline_cache_filename names, and this session's Used record
    std::unique_ptr<VideoCommon::ShaderCacheV2::Cache> cache_v2;
    /// Its driver layer, where the device has VK_KHR_pipeline_binary
    std::unique_ptr<PipelineBinaries> pipeline_binaries;
    u64 used_session_key{};
    u64 used_session_time{};

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;
    size_t pipelines_since_flush{};
    std::chrono::steady_clock::time_point last_flush{};
    std::atomic<size_t> last_cache_size{};
    std::atomic_bool flush_in_flight{};

    /// Requested first when the cache is destroyed, to stop a rewrite of the disk cache: Android
    /// and yuzu_cmd load with a stop token that never stops
    std::stop_source serialization_stop;
    Common::ThreadWorker serialization_thread;
    DynamicFeatures dynamic_features;
};

} // namespace Vulkan
