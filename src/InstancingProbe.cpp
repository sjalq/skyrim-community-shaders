#include "InstancingProbe.h"

#include "ConstantBufferRing.h"
#include "Globals.h"
#include "State.h"
#include "Utils/VersionedRelocation.h"

namespace InstancingProbe
{
	namespace
	{
		constexpr uint32_t kLogFrames = 600;

		// Per RenderBatches call tables (design 2.3: ~50-300 keys per call). Allocated once on first enable,
		// stamped with a per-call generation so nothing is cleared or allocated per call or per draw.
		constexpr uint32_t kTableSlots = 8192;  // power of two
		constexpr uint32_t kTableMask = kTableSlots - 1;
		constexpr uint32_t kMaxTableEntries = kTableSlots * 3 / 4;
		constexpr uint32_t kMaxMembersPerCall = 8192;
		constexpr uint32_t kMaxSnapshotsPerCall = 1024;
		constexpr uint32_t kCbRegisters = 32;  // PerGeometry (b2) registers compared per stage
		constexpr uint32_t kCbBytes = kCbRegisters * 16;
		constexpr uint32_t kMaxVsIds = 32;
		constexpr uint32_t kMaxMismatchExamples = 4;
		constexpr uint32_t kHistogramBins = 5;  // bucket sizes 1, 2-3, 4-7, 8-15, 16+

		// Technique ids at RenderPassImmediately are the shader's raw technique plus a per-shader base
		// (Skylighting.cpp ShaderTechnique: UtilityGeneralStart, LightingGeneralStart).
		constexpr uint32_t kUtilityTechniqueBase = 0x2B;
		constexpr uint32_t kLightingTechniqueBase = 0x4800002D;
		// ShaderCache.h UtilityShaderFlags: Skinned, LodLandscape, LodObject; TreeAnim.
		constexpr uint32_t kUtilityExcluded = (1u << 2) | (1u << 8) | (1u << 27);
		constexpr uint32_t kUtilityTreeAnim = 1u << 26;
		// ShaderCache.h LightingShaderFlags::Skinned; technique type in bits 24-29.
		constexpr uint32_t kLightingSkinned = 1u << 1;
		constexpr uint32_t kUtilityTreeParamsRegister = 7;  // Utility.hlsl PerGeometry TreeParams : packoffset(c7)

		enum class Window : uint8_t
		{
			None,
			Cascade0,
			Cascade1,
			Cascade2,
			Cascade3Plus,
			Prepass,
			GBuffer,
			Cubemap,
			Count
		};
		constexpr std::array<const char*, static_cast<size_t>(Window::Count)> kWindowNames{ "none", "cascade0", "cascade1",
			"cascade2", "cascade3+", "prepass", "gbuffer", "cubemap" };

		[[nodiscard]] constexpr bool IsUtilityWindow(Window a_window)
		{
			return a_window >= Window::Cascade0 && a_window <= Window::Prepass;
		}

		// Eligible(P), design 4.1, first failing rule. The interior gate is not applied: interior frames are
		// counted separately so the probe also measures interiors.
		enum Reason : uint8_t
		{
			kEligible,
			kShader,     // missing pass members, or not Utility (shadow/prepass) / Lighting (G-buffer/cubemap)
			kProperty,   // shader property is not exactly BSLightingShaderProperty
			kGeometry,   // geometry is not exactly BSTriShape, or has no renderer data
			kSkinned,    // skin instance present
			kTechnique,  // Skinned / LOD technique bits
			kFlags,      // skinned, decal, landscape, LOD or no-transparency-multisample property flags
			kTree,       // kTreeAnim without a TESObjectTREE base
			kFade,       // property alpha or parent BSFadeNode fade below 1
			kReasonCount
		};

		// Bucket key, design 2.3. Value-initialised so padding compares equal.
		struct Key
		{
			const void* shader;
			const void* rendererData;
			const void* material;
			const void* treeBase;
			uint64_t vertexDesc;
			uint64_t propertyFlags;
			uint32_t technique;
			uint32_t renderFlags;
			uint32_t triangleCount;
			uint16_t alphaFlags;
			uint8_t alphaThreshold;
			uint8_t alphaTest;
			uint8_t accumulationHint;
			uint8_t extraParam;
			uint8_t lodMode;
			uint8_t pad[5];
		};
		static_assert(sizeof(Key) == 72);

		struct Bucket
		{
			Key key;
			uint64_t hash;
			uint32_t count;
			int32_t snapshot;  // index of the first member's b2 snapshot, -1 when none
			uint16_t vsBytes;
			uint16_t psBytes;
			bool tree;
		};

		struct Member
		{
			RE::BSRenderPass* pass;
			RE::BSGeometry* geometry;
			RE::BSShaderProperty* property;
		};

		/** @brief Open-addressing pointer set, stamped with the per-call generation. */
		struct PointerSet
		{
			std::vector<const void*> keys;
			std::vector<uint32_t> stamps;
			uint32_t count = 0;

			void Allocate()
			{
				keys.assign(kTableSlots, nullptr);
				stamps.assign(kTableSlots, 0);
			}

			/** @return True when a_pointer was not in the set yet. */
			bool Insert(const void* a_pointer, uint32_t a_generation)
			{
				if (count >= kMaxTableEntries)
					return false;
				uint32_t i = static_cast<uint32_t>((reinterpret_cast<uintptr_t>(a_pointer) * 0x9E3779B97F4A7C15ull) >> 40) & kTableMask;
				while (stamps[i] == a_generation) {
					if (keys[i] == a_pointer)
						return false;
					i = (i + 1) & kTableMask;
				}
				stamps[i] = a_generation;
				keys[i] = a_pointer;
				++count;
				return true;
			}
		};

		struct WindowStats
		{
			uint64_t passes = 0;
			uint64_t eligible = 0;
			uint64_t buckets = 0;
			uint64_t batches = 0;    // RenderBatches calls with at least one eligible pass
			uint64_t mergeable = 0;  // eligible passes in buckets of two or more
			std::array<uint64_t, kHistogramBins> histogram{};
			std::array<uint64_t, kReasonCount> reasons{};
			uint64_t geometries = 0;  // distinct per call, summed
			uint64_t rendererData = 0;
			uint64_t materials = 0;
			uint64_t drawIndexed = 0;                      // DrawIndexed calls inside every pass of the window
			std::array<uint64_t, 3> eligibleDrawCounts{};  // eligible passes that issued 0, 1, 2+ DrawIndexed
			uint64_t passTicks = 0;
			uint64_t eligibleTicks = 0;
			uint64_t flushChecked = 0;
			uint64_t flushChanged = 0;  // pass->geometry or pass->shaderProperty changed by the time the call returned
			uint64_t vsIdMatch = 0;
			uint64_t vsIdMismatch = 0;
			uint32_t mismatchExamples = 0;
			std::array<std::pair<uint32_t, uint32_t>, kMaxMismatchExamples> mismatches{};
			uint64_t cbCompared = 0;
			uint64_t cbUnavailable = 0;
			std::array<uint64_t, kCbRegisters> vsDiff{};
			std::array<uint64_t, kCbRegisters> psDiff{};
			uint64_t treeMembers = 0;  // TREE_ANIM passes after the first of their bucket
			uint64_t treeCompared = 0;
			std::array<uint64_t, kCbRegisters> treeVsDiff{};
			std::array<uint64_t, kCbRegisters> treePsDiff{};
			uint32_t vsIdCount = 0;
			std::array<std::pair<uint32_t, uint64_t>, kMaxVsIds> vsIds{};
			uint64_t overflows = 0;
		};

		struct CurrentPass
		{
			Window window = Window::None;
			int32_t bucket = -1;
			uint32_t drawIndexed = 0;
			bool eligible = false;
			bool first = false;
			bool tree = false;
		};

		struct ProbeState
		{
			std::atomic_bool requested{ false };
			bool active = false;  // latched at the frame boundary; every hook reads only this
			bool allocated = false;
			bool drawIndexedHooked = false;
			std::atomic_bool armed{ false };  // inside a measured RenderPassImmediately: DrawIndexed counts
			ID3D11DeviceContext* context = nullptr;
			const RE::NiRTTI* triShapeRtti = nullptr;
			const RE::NiRTTI* lightingPropertyRtti = nullptr;
			int64_t qpcFrequency = 1;

			Window window = Window::None;
			uint32_t cascadeCalls = 0;  // RenderCascade calls this frame; the k-th call is cascade k
			uint32_t batchDepth = 0;

			// Current RenderBatches call.
			uint32_t generation = 0;
			Window callWindow = Window::None;
			bool callOverflow = false;
			std::vector<Bucket> buckets;
			std::vector<uint32_t> bucketStamps;
			std::vector<uint32_t> usedBuckets;
			uint32_t usedCount = 0;
			std::vector<Member> members;
			uint32_t memberCount = 0;
			PointerSet geometrySet;
			PointerSet rendererDataSet;
			PointerSet materialSet;
			std::vector<std::byte> snapshots;  // kMaxSnapshotsPerCall x (VS b2, PS b2)
			uint32_t snapshotsUsed = 0;

			CurrentPass current;

			// Current log period.
			uint32_t frames = 0;
			uint32_t interiorFrames = 0;
			uint64_t outsidePasses = 0;
			uint64_t unbatchedPasses = 0;
			uint64_t cullTicks = 0;
			uint64_t cullCalls = 0;
			std::array<WindowStats, static_cast<size_t>(Window::Count)> windows{};
		};

		// Leaked on purpose: the game can still call through these hooks after static destructors run.
		ProbeState& g = *new ProbeState();

		[[nodiscard]] int64_t Now()
		{
			LARGE_INTEGER counter;
			QueryPerformanceCounter(&counter);
			return counter.QuadPart;
		}

		[[nodiscard]] uint32_t HistogramBin(uint32_t a_size)
		{
			return a_size <= 1 ? 0u : a_size <= 3 ? 1u :
			                      a_size <= 7     ? 2u :
			                      a_size <= 15    ? 3u :
			                                        4u;
		}

		[[nodiscard]] uint64_t HashKey(const Key& a_key)
		{
			uint64_t words[sizeof(Key) / sizeof(uint64_t)];
			std::memcpy(words, &a_key, sizeof(Key));
			uint64_t hash = 0xCBF29CE484222325ull;
			for (const uint64_t word : words) {
				hash ^= word;
				hash *= 0x9E3779B97F4A7C15ull;
				hash ^= hash >> 29;
			}
			return hash;
		}

		/** @brief Base TESObjectTREE of the reference that owns a_geometry, or null (design 3.3 tree key). */
		[[nodiscard]] const void* TreeBase(RE::BSGeometry* a_geometry)
		{
			auto* reference = a_geometry->GetUserData();
			if (!reference)
				return nullptr;
			auto* base = reference->GetBaseObject();
			return base && base->GetFormType() == RE::FormType::Tree ? base : nullptr;
		}

		/** @brief currentFade of the nearest BSFadeNode above a_geometry, 1 when there is none. */
		[[nodiscard]] float ParentFade(RE::BSGeometry* a_geometry)
		{
			for (RE::NiNode* node = a_geometry->parent; node; node = node->parent) {
				if (auto* fadeNode = node->AsFadeNode())
					return fadeNode->GetRuntimeData().currentFade;
			}
			return 1.0f;
		}

		/** @brief Design 4.1 Eligible(P); fills the design 2.3 bucket key when eligible. */
		[[nodiscard]] Reason Classify(RE::BSRenderPass* a_pass, uint32_t a_technique, bool a_alphaTest, uint32_t a_renderFlags,
			bool a_utilityWindow, Key& a_key, bool& a_tree)
		{
			auto* shader = a_pass->shader;
			auto* property = a_pass->shaderProperty;
			auto* geometry = a_pass->geometry;
			if (!shader || !property || !geometry)
				return kShader;
			const auto type = shader->shaderType.get();
			if (a_utilityWindow ? type != RE::BSShader::Type::Utility : type != RE::BSShader::Type::Lighting)
				return kShader;
			if (property->GetRTTI() != g.lightingPropertyRtti)
				return kProperty;
			if (geometry->GetRTTI() != g.triShapeRtti)
				return kGeometry;
			auto& geometryData = geometry->GetGeometryRuntimeData();
			if (!geometryData.rendererData)
				return kGeometry;
			if (geometryData.skinInstance)
				return kSkinned;

			bool treeTechnique = false;
			if (a_utilityWindow) {
				const uint32_t raw = a_technique - kUtilityTechniqueBase;
				if (raw & kUtilityExcluded)
					return kTechnique;
				treeTechnique = (raw & kUtilityTreeAnim) != 0;
			} else {
				const uint32_t raw = a_technique - kLightingTechniqueBase;
				const uint32_t lightingType = (raw >> 24) & 0x3F;
				// LightingShaderTechniques: MTLand 8, LODLand 9, LODObjects 13, LODObjectHD 15, LODLandNoise 18, MTLandLODBlend 19.
				if ((raw & kLightingSkinned) || lightingType == 8 || lightingType == 9 || lightingType == 13 || lightingType == 15 ||
					lightingType == 18 || lightingType == 19)
					return kTechnique;
				treeTechnique = lightingType == 12;  // TreeAnim
			}

			using Flag = RE::BSShaderProperty::EShaderPropertyFlag;
			if (property->flags.any(Flag::kSkinned, Flag::kDecal, Flag::kDynamicDecal, Flag::kMultiTextureLandscape, Flag::kLODLandscape,
					Flag::kLODObjects, Flag::kNoTransparencyMultiSample))
				return kFlags;

			const void* treeBase = nullptr;
			a_tree = treeTechnique || property->flags.all(Flag::kTreeAnim);
			if (a_tree) {
				treeBase = TreeBase(geometry);
				if (!treeBase)
					return kTree;
			}

			if (property->alpha != 1.0f || ParentFade(geometry) != 1.0f)
				return kFade;

			a_key.shader = shader;
			a_key.rendererData = geometryData.rendererData;
			a_key.material = property->material;
			a_key.treeBase = treeBase;
			std::memcpy(&a_key.vertexDesc, &geometryData.vertexDesc, sizeof(uint64_t));
			a_key.propertyFlags = property->flags.underlying();
			a_key.technique = a_technique;
			a_key.renderFlags = a_renderFlags;
			a_key.triangleCount = static_cast<RE::BSTriShape*>(geometry)->GetTrishapeRuntimeData().triangleCount;
			if (auto* alphaProperty = geometryData.alphaProperty.get()) {
				a_key.alphaFlags = alphaProperty->alphaFlags;
				a_key.alphaThreshold = alphaProperty->alphaThreshold;
			}
			a_key.alphaTest = a_alphaTest ? 1 : 0;
			a_key.accumulationHint = a_pass->accumulationHint;
			a_key.extraParam = a_pass->extraParam;
			std::memcpy(&a_key.lodMode, &a_pass->LODMode, sizeof(uint8_t));
			return kEligible;
		}

		void BeginCall()
		{
			if (++g.generation == 0) {
				std::fill(g.bucketStamps.begin(), g.bucketStamps.end(), 0u);
				for (auto* set : { &g.geometrySet, &g.rendererDataSet, &g.materialSet })
					std::fill(set->stamps.begin(), set->stamps.end(), 0u);
				g.generation = 1;
			}
			g.callWindow = Window::None;
			g.callOverflow = false;
			g.usedCount = 0;
			g.memberCount = 0;
			g.snapshotsUsed = 0;
			g.geometrySet.count = 0;
			g.rendererDataSet.count = 0;
			g.materialSet.count = 0;
		}

		/** @brief Adds an eligible pass to its bucket of the current RenderBatches call (no draw is held back). */
		void Gather(RE::BSRenderPass* a_pass, const Key& a_key, bool a_tree, Window a_window, WindowStats& a_stats)
		{
			g.callWindow = a_window;
			const uint32_t generation = g.generation;
			if (g.geometrySet.Insert(a_pass->geometry, generation))
				++a_stats.geometries;
			if (g.rendererDataSet.Insert(a_key.rendererData, generation))
				++a_stats.rendererData;
			if (g.materialSet.Insert(a_key.material, generation))
				++a_stats.materials;
			if (g.memberCount < kMaxMembersPerCall)
				g.members[g.memberCount++] = { a_pass, a_pass->geometry, a_pass->shaderProperty };
			else
				g.callOverflow = true;

			const uint64_t hash = HashKey(a_key);
			uint32_t i = static_cast<uint32_t>(hash >> 32) & kTableMask;
			while (g.bucketStamps[i] == generation) {
				auto& bucket = g.buckets[i];
				if (bucket.hash == hash && std::memcmp(&bucket.key, &a_key, sizeof(Key)) == 0) {
					++bucket.count;
					g.current.bucket = static_cast<int32_t>(i);
					g.current.first = false;
					return;
				}
				i = (i + 1) & kTableMask;
			}
			if (g.usedCount >= kMaxTableEntries) {
				g.callOverflow = true;
				return;
			}
			g.bucketStamps[i] = generation;
			auto& bucket = g.buckets[i];
			bucket.key = a_key;
			bucket.hash = hash;
			bucket.count = 1;
			bucket.snapshot = -1;
			bucket.vsBytes = 0;
			bucket.psBytes = 0;
			bucket.tree = a_tree;
			g.usedBuckets[g.usedCount++] = i;
			g.current.bucket = static_cast<int32_t>(i);
			g.current.first = true;
		}

		/** @brief Counts the buckets of the RenderBatches call that just returned (the S1 flush point). */
		void FlushCall()
		{
			if (g.callWindow == Window::None)
				return;
			auto& stats = g.windows[static_cast<size_t>(g.callWindow)];
			++stats.batches;
			stats.buckets += g.usedCount;
			for (uint32_t u = 0; u < g.usedCount; ++u) {
				const auto& bucket = g.buckets[g.usedBuckets[u]];
				++stats.histogram[HistogramBin(bucket.count)];
				if (bucket.count >= 2) {
					stats.mergeable += bucket.count;
					if (bucket.tree)
						stats.treeMembers += bucket.count - 1;
				}
			}
			// Design 2.2: do the gathered passes still describe the same draw when the call returns?
			for (uint32_t m = 0; m < g.memberCount; ++m) {
				const auto& member = g.members[m];
				++stats.flushChecked;
				if (member.pass->geometry != member.geometry || member.pass->shaderProperty != member.property)
					++stats.flushChanged;
			}
			if (g.callOverflow)
				++stats.overflows;
		}

		/** @brief Latest contents of the PerGeometry (b2) buffer bound to one stage, while the CB ring holds them. */
		[[nodiscard]] bool ReadPerGeometry(ID3D11DeviceContext* a_context, bool a_vertex, const std::byte*& a_data, uint32_t& a_bytes)
		{
			ID3D11Buffer* buffer = nullptr;
			if (a_vertex)
				a_context->VSGetConstantBuffers(2, 1, &buffer);
			else
				a_context->PSGetConstantBuffers(2, 1, &buffer);
			if (!buffer)
				return false;
			const void* data = nullptr;
			UINT bytes = 0;
			const bool found = ConstantBufferRing::PeekLatest(buffer, data, bytes);
			buffer->Release();
			if (!found)
				return false;
			a_data = static_cast<const std::byte*>(data);
			a_bytes = std::min<uint32_t>(bytes, kCbBytes);
			return true;
		}

		void DiffRegisters(const std::byte* a_reference, const std::byte* a_data, uint32_t a_bytes,
			std::array<uint64_t, kCbRegisters>& a_diff, std::array<uint64_t, kCbRegisters>* a_treeDiff)
		{
			for (uint32_t r = 0; r < a_bytes / 16; ++r) {
				if (std::memcmp(a_reference + r * 16, a_data + r * 16, 16) != 0) {
					++a_diff[r];
					if (a_treeDiff)
						++(*a_treeDiff)[r];
				}
			}
		}

		/** @brief Design 3.3/3.4 cb2 variance: compares a member's PerGeometry registers with its bucket's first member. */
		void ComparePerGeometry(ID3D11DeviceContext* a_context, WindowStats& a_stats)
		{
			auto& current = g.current;
			auto& bucket = g.buckets[static_cast<uint32_t>(current.bucket)];
			const std::byte* vsData = nullptr;
			uint32_t vsBytes = 0;
			if (!ReadPerGeometry(a_context, true, vsData, vsBytes)) {
				++a_stats.cbUnavailable;
				return;
			}
			const std::byte* psData = nullptr;
			uint32_t psBytes = 0;
			if (!*globals::game::currentPixelShader || !ReadPerGeometry(a_context, false, psData, psBytes))
				psBytes = 0;

			if (current.first) {
				if (g.snapshotsUsed >= kMaxSnapshotsPerCall) {
					g.callOverflow = true;
					return;
				}
				bucket.snapshot = static_cast<int32_t>(g.snapshotsUsed++);
				std::byte* snapshot = g.snapshots.data() + static_cast<size_t>(bucket.snapshot) * 2 * kCbBytes;
				std::memcpy(snapshot, vsData, vsBytes);
				if (psBytes)
					std::memcpy(snapshot + kCbBytes, psData, psBytes);
				bucket.vsBytes = static_cast<uint16_t>(vsBytes);
				bucket.psBytes = static_cast<uint16_t>(psBytes);
				return;
			}
			if (bucket.snapshot < 0)
				return;
			const std::byte* snapshot = g.snapshots.data() + static_cast<size_t>(bucket.snapshot) * 2 * kCbBytes;
			++a_stats.cbCompared;
			if (current.tree)
				++a_stats.treeCompared;
			DiffRegisters(snapshot, vsData, std::min<uint32_t>(vsBytes, bucket.vsBytes), a_stats.vsDiff, current.tree ? &a_stats.treeVsDiff : nullptr);
			if (psBytes && bucket.psBytes)
				DiffRegisters(snapshot + kCbBytes, psData, std::min<uint32_t>(psBytes, bucket.psBytes), a_stats.psDiff,
					current.tree ? &a_stats.treePsDiff : nullptr);
		}

		void NoteVertexShader(WindowStats& a_stats)
		{
			auto* vertexShader = *globals::game::currentVertexShader;
			if (!vertexShader)
				return;
			// Design 3.2: is VertexShader::id the shader cache descriptor of the current technique?
			const uint32_t id = vertexShader->id;
			const uint32_t descriptor = globals::state->modifiedVertexDescriptor;
			if (id == descriptor) {
				++a_stats.vsIdMatch;
			} else {
				++a_stats.vsIdMismatch;
				if (a_stats.mismatchExamples < kMaxMismatchExamples)
					a_stats.mismatches[a_stats.mismatchExamples++] = { id, descriptor };
			}
			for (uint32_t i = 0; i < a_stats.vsIdCount; ++i) {
				if (a_stats.vsIds[i].first == id) {
					++a_stats.vsIds[i].second;
					return;
				}
			}
			if (a_stats.vsIdCount < kMaxVsIds)
				a_stats.vsIds[a_stats.vsIdCount++] = { id, 1 };
		}

		void OnArmedDrawIndexed(ID3D11DeviceContext* a_context)
		{
			auto& current = g.current;
			auto& stats = g.windows[static_cast<size_t>(current.window)];
			++stats.drawIndexed;
			if (++current.drawIndexed != 1 || !current.eligible)
				return;
			NoteVertexShader(stats);
			if (current.bucket >= 0)
				ComparePerGeometry(a_context, stats);
		}

		struct RenderPassImmediately
		{
			static void thunk(RE::BSRenderPass* a_pass, uint32_t a_technique, bool a_alphaTest, uint32_t a_renderFlags);
			static inline REL::Relocation<decltype(thunk)> func;
		};

		void MeasurePass(RE::BSRenderPass* a_pass, uint32_t a_technique, bool a_alphaTest, uint32_t a_renderFlags)
		{
			const Window window = g.window;
			if (window == Window::None || !a_pass) {
				++g.outsidePasses;
				RenderPassImmediately::func(a_pass, a_technique, a_alphaTest, a_renderFlags);
				return;
			}
			auto& stats = g.windows[static_cast<size_t>(window)];
			++stats.passes;
			auto& current = g.current;
			current = {};
			current.window = window;
			if (g.batchDepth == 0) {
				++g.unbatchedPasses;
			} else {
				Key key{};
				bool tree = false;
				const Reason reason = Classify(a_pass, a_technique, a_alphaTest, a_renderFlags, IsUtilityWindow(window), key, tree);
				if (reason == kEligible) {
					++stats.eligible;
					current.eligible = true;
					current.tree = tree;
					Gather(a_pass, key, tree, window, stats);
				} else {
					++stats.reasons[reason];
				}
			}

			g.armed.store(g.drawIndexedHooked, std::memory_order_relaxed);
			const int64_t start = Now();
			RenderPassImmediately::func(a_pass, a_technique, a_alphaTest, a_renderFlags);
			const uint64_t ticks = static_cast<uint64_t>(Now() - start);
			g.armed.store(false, std::memory_order_relaxed);

			stats.passTicks += ticks;
			if (current.eligible) {
				stats.eligibleTicks += ticks;
				if (g.drawIndexedHooked)
					++stats.eligibleDrawCounts[std::min<uint32_t>(current.drawIndexed, 2)];
			}
		}

		void RenderPassImmediately::thunk(RE::BSRenderPass* a_pass, uint32_t a_technique, bool a_alphaTest, uint32_t a_renderFlags)
		{
			if (!g.active) [[likely]] {
				func(a_pass, a_technique, a_alphaTest, a_renderFlags);
				return;
			}
			MeasurePass(a_pass, a_technique, a_alphaTest, a_renderFlags);
		}

		struct BSBatchRenderer_RenderBatches
		{
			static bool thunk(void* a_renderer, uint32_t* a_currentPass, uint32_t* a_bucketIndex, void* a_passIndexList, uint32_t a_renderFlags)
			{
				if (!g.active) [[likely]]
					return func(a_renderer, a_currentPass, a_bucketIndex, a_passIndexList, a_renderFlags);
				if (g.batchDepth++ == 0)
					BeginCall();
				const bool result = func(a_renderer, a_currentPass, a_bucketIndex, a_passIndexList, a_renderFlags);
				if (--g.batchDepth == 0)
					FlushCall();
				return result;
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct RenderCascade
		{
			static void thunk(void* a_light, void* a_arg1, void* a_arg2, uint32_t a_flags)
			{
				if (!g.active) [[likely]] {
					func(a_light, a_arg1, a_arg2, a_flags);
					return;
				}
				const Window previous = g.window;
				const uint32_t cascade = std::min(g.cascadeCalls++, 3u);
				g.window = static_cast<Window>(static_cast<uint32_t>(Window::Cascade0) + cascade);
				func(a_light, a_arg1, a_arg2, a_flags);
				g.window = previous;
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct Main_RenderDepth
		{
			static void thunk(bool a1, bool a2)
			{
				if (!g.active) [[likely]] {
					func(a1, a2);
					return;
				}
				const Window previous = g.window;
				g.window = Window::Prepass;
				func(a1, a2);
				g.window = previous;
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct Main_RenderWorld_Start
		{
			static void thunk(void* a_this, uint32_t a_startRange, uint32_t a_endRanges, uint32_t a_renderFlags, int a_geometryGroup)
			{
				// The G-buffer window runs from the first opaque batch to the blended decals (Deferred.h).
				if (g.active && g.window == Window::None && globals::state->inWorld)
					g.window = Window::GBuffer;
				func(a_this, a_startRange, a_endRanges, a_renderFlags, a_geometryGroup);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct Main_RenderWorld_BlendedDecals
		{
			static void thunk(void* a_this, uint32_t a_renderFlags)
			{
				if (g.window == Window::GBuffer)
					g.window = Window::None;
				func(a_this, a_renderFlags);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct BSCubeMapCamera_RenderCubemap
		{
			static void thunk(RE::NiAVObject* a_camera, int a2, bool a3, bool a4, bool a5)
			{
				if (!g.active) [[likely]] {
					func(a_camera, a2, a3, a4, a5);
					return;
				}
				const Window previous = g.window;
				g.window = Window::Cubemap;
				func(a_camera, a2, a3, a4, a5);
				g.window = previous;
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct DirShadowLightCulling
		{
			static void thunk(void* a_light, void* a_jobArrays, void* a_nodes)
			{
				if (!g.active) [[likely]] {
					func(a_light, a_jobArrays, a_nodes);
					return;
				}
				const int64_t start = Now();
				func(a_light, a_jobArrays, a_nodes);
				g.cullTicks += static_cast<uint64_t>(Now() - start);
				++g.cullCalls;
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct ID3D11DeviceContext_DrawIndexed
		{
			static void thunk(ID3D11DeviceContext* This, UINT a_indexCount, UINT a_startIndex, INT a_baseVertex)
			{
				if (g.armed.load(std::memory_order_relaxed) && This == g.context) [[unlikely]]
					OnArmedDrawIndexed(This);
				func(This, a_indexCount, a_startIndex, a_baseVertex);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		void Allocate()
		{
			g.buckets.resize(kTableSlots);
			g.bucketStamps.assign(kTableSlots, 0);
			g.usedBuckets.resize(kMaxTableEntries);
			g.members.resize(kMaxMembersPerCall);
			g.geometrySet.Allocate();
			g.rendererDataSet.Allocate();
			g.materialSet.Allocate();
			g.snapshots.resize(static_cast<size_t>(kMaxSnapshotsPerCall) * 2 * kCbBytes);
			g.generation = 0;
			LARGE_INTEGER frequency;
			QueryPerformanceFrequency(&frequency);
			g.qpcFrequency = std::max<int64_t>(frequency.QuadPart, 1);
			g.allocated = true;
		}

		void ResetPeriod()
		{
			g.frames = 0;
			g.interiorFrames = 0;
			g.outsidePasses = 0;
			g.unbatchedPasses = 0;
			g.cullTicks = 0;
			g.cullCalls = 0;
			g.windows = {};
		}

		[[nodiscard]] std::string RegisterList(const std::array<uint64_t, kCbRegisters>& a_diff, uint64_t a_compared)
		{
			std::string out;
			for (uint32_t r = 0; r < kCbRegisters; ++r) {
				if (a_diff[r])
					out += std::format("{}c{}:{:.1f}%", out.empty() ? "" : ",", r, 100.0 * static_cast<double>(a_diff[r]) / static_cast<double>(std::max<uint64_t>(a_compared, 1)));
			}
			return out.empty() ? "-" : out;
		}

		void LogPeriod()
		{
			const double frames = std::max<double>(g.frames, 1.0);
			const auto perFrame = [frames](uint64_t a_value) { return static_cast<double>(a_value) / frames; };
			const auto microseconds = [](uint64_t a_ticks, uint64_t a_count) {
				return a_count ? static_cast<double>(a_ticks) * 1e6 / static_cast<double>(g.qpcFrequency) / static_cast<double>(a_count) : 0.0;
			};

			for (size_t w = static_cast<size_t>(Window::Cascade0); w < static_cast<size_t>(Window::Count); ++w) {
				const auto& s = g.windows[w];
				const bool utility = IsUtilityWindow(static_cast<Window>(w));
				const double e = s.passes ? static_cast<double>(s.eligible) / static_cast<double>(s.passes) : 0.0;
				const double k = s.buckets ? static_cast<double>(s.eligible) / static_cast<double>(s.buckets) : 0.0;
				const std::string treeParams = utility && s.treeCompared ?
				                                   std::format("{:.1f}%", 100.0 * static_cast<double>(s.treeVsDiff[kUtilityTreeParamsRegister]) / static_cast<double>(s.treeCompared)) :
				                                   std::string("n/a");
				logger::info(
					"[InstancingProbe] {} frames={} passes={:.1f} eligible={:.1f} e={:.3f} buckets={:.1f} k={:.2f} drawsAfter={:.1f} mergeable={:.1f} "
					"hist=1:{:.1f}/2-3:{:.1f}/4-7:{:.1f}/8-15:{:.1f}/16+:{:.1f} batches={:.1f} geoms={:.1f} rdata={:.1f} mats={:.1f} "
					"di={:.1f} eligDi=0:{:.1f}/1:{:.1f}/2+:{:.1f} usPass={:.2f} usElig={:.2f} flushChanged={}/{} vsId=match:{}/mismatch:{} "
					"cb=cmp:{:.1f}/na:{:.1f} vsDiff={} psDiff={} treeMembers={:.1f} treeCmp={:.1f} treeParamsDiff={} treeVsDiff={} treePsDiff={} "
					"inelig=shader:{:.1f}/prop:{:.1f}/geom:{:.1f}/skin:{:.1f}/tech:{:.1f}/flags:{:.1f}/tree:{:.1f}/fade:{:.1f} overflows={}",
					kWindowNames[w], g.frames, perFrame(s.passes), perFrame(s.eligible), e, perFrame(s.buckets), k,
					perFrame(s.passes - s.eligible + s.buckets), perFrame(s.mergeable),
					perFrame(s.histogram[0]), perFrame(s.histogram[1]), perFrame(s.histogram[2]), perFrame(s.histogram[3]), perFrame(s.histogram[4]),
					perFrame(s.batches), perFrame(s.geometries), perFrame(s.rendererData), perFrame(s.materials),
					perFrame(s.drawIndexed), perFrame(s.eligibleDrawCounts[0]), perFrame(s.eligibleDrawCounts[1]), perFrame(s.eligibleDrawCounts[2]),
					microseconds(s.passTicks, s.passes), microseconds(s.eligibleTicks, s.eligible), s.flushChanged, s.flushChecked,
					s.vsIdMatch, s.vsIdMismatch, perFrame(s.cbCompared), perFrame(s.cbUnavailable),
					RegisterList(s.vsDiff, s.cbCompared), RegisterList(s.psDiff, s.cbCompared),
					perFrame(s.treeMembers), perFrame(s.treeCompared), treeParams,
					RegisterList(s.treeVsDiff, s.treeCompared), RegisterList(s.treePsDiff, s.treeCompared),
					perFrame(s.reasons[kShader]), perFrame(s.reasons[kProperty]), perFrame(s.reasons[kGeometry]), perFrame(s.reasons[kSkinned]),
					perFrame(s.reasons[kTechnique]), perFrame(s.reasons[kFlags]), perFrame(s.reasons[kTree]), perFrame(s.reasons[kFade]),
					s.overflows);

				if (s.vsIdCount) {
					std::string ids;
					for (uint32_t i = 0; i < s.vsIdCount; ++i)
						ids += std::format("{}0x{:X}:{:.1f}", i ? "," : "", s.vsIds[i].first, perFrame(s.vsIds[i].second));
					std::string mismatches;
					for (uint32_t i = 0; i < s.mismatchExamples; ++i)
						mismatches += std::format("{}id=0x{:X}/descriptor=0x{:X}", i ? "," : "", s.mismatches[i].first, s.mismatches[i].second);
					logger::info("[InstancingProbe] {} vsIds={} mismatchExamples={}", kWindowNames[w], ids, mismatches.empty() ? "-" : mismatches);
				}
			}

			const auto ringStatus = ConstantBufferRing::GetStatus();
			logger::info("[InstancingProbe] frame frames={} interiorFrames={} outsidePasses={:.1f} unbatchedPasses={:.1f} dirShadowCullUs={:.1f} cullCalls={:.1f} drawIndexedHook={} cbRing={}",
				g.frames, g.interiorFrames, perFrame(g.outsidePasses), perFrame(g.unbatchedPasses),
				static_cast<double>(g.cullTicks) * 1e6 / static_cast<double>(g.qpcFrequency) / frames, perFrame(g.cullCalls),
				g.drawIndexedHooked ? "on" : "off", ringStatus.active ? "active" : "inactive (no cb variance)");
		}
	}

	void SetEnabled(bool a_enabled)
	{
		g.requested.store(a_enabled, std::memory_order_relaxed);
	}

	bool IsEnabled()
	{
		return g.requested.load(std::memory_order_relaxed);
	}

	void Install()
	{
		g.triShapeRtti = REL::Relocation<const RE::NiRTTI*>{ RE::BSTriShape::Ni_RTTI }.get();
		g.lightingPropertyRtti = REL::Relocation<const RE::NiRTTI*>{ RE::BSLightingShaderProperty::Ni_RTTI }.get();

		// Installed after every feature's PostPostLoad, so these thunks are outermost on shared call sites
		// (InteriorSun, Light Limit Fix and Terrain Blending chain the same RenderPassImmediately call).
		stl::write_thunk_call<RenderPassImmediately>(REL::RelocationID(100852, 107642).address() + REL::Relocate(0x29E, 0x28F));
		stl::detour_thunk<BSBatchRenderer_RenderBatches>(REL::RelocationID(100852, 107642));
		stl::write_thunk_call<RenderCascade>(REL::RelocationID(101495, 108489).address() + REL::Relocate(0xC6, 0xC6));
		stl::write_thunk_call<Main_RenderDepth>(REL::RelocationID(35560, 36559).address() + Util::VersionedRelocation::Select(0x395, 0x395, 0x3B3));
		stl::write_thunk_call<Main_RenderWorld_Start>(REL::RelocationID(99938, 106583).address() + REL::Relocate(0x8E, 0x84));
		stl::write_thunk_call<Main_RenderWorld_BlendedDecals>(REL::RelocationID(99938, 106583).address() + REL::Relocate(0x319, 0x308));
		stl::write_vfunc<0x35, BSCubeMapCamera_RenderCubemap>(RE::VTABLE_BSCubeMapCamera[0]);
		stl::detour_thunk<DirShadowLightCulling>(REL::RelocationID(101498, 108492));

		logger::info("[InstancingProbe] Installed hooks (counters only; enable with Advanced > \"Instancing Probe\")");
	}

	void InstallD3DHooks(ID3D11DeviceContext* a_context)
	{
		g.context = a_context;
		g.drawIndexedHooked = ConstantBufferRing::DetourVFunc<12, ID3D11DeviceContext_DrawIndexed>(a_context);
		if (!g.drawIndexedHooked)
			logger::warn("[InstancingProbe] DrawIndexed hook is not live; draws per pass, VS id and cb2 variance will not be counted");
	}

	void OnFrameBoundary()
	{
		const bool wanted = g.requested.load(std::memory_order_relaxed);
		if (wanted && !g.allocated)
			Allocate();
		if (wanted != g.active) {
			g.active = wanted;
			ResetPeriod();
			logger::info("[InstancingProbe] {}", wanted ? "Enabled: logging per-window counters every 600 frames" : "Disabled");
		}
		g.window = Window::None;
		g.cascadeCalls = 0;
		g.batchDepth = 0;
		g.armed.store(false, std::memory_order_relaxed);
		if (!g.active)
			return;

		if (auto* tes = RE::TES::GetSingleton(); tes && tes->interiorCell)
			++g.interiorFrames;
		if (++g.frames >= kLogFrames) {
			LogPeriod();
			ResetPeriod();
		}
	}
}
