#include "ConstantBufferRing.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <d3d11_1.h>
#include <functional>

#include "Utils/D3D.h"

namespace ConstantBufferRing
{
	namespace
	{
		constexpr UINT kSlotCount = D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT;
		constexpr UINT kConstantBytes = 16;
		constexpr UINT kMaxWindowConstants = D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT;
		// *SetConstantBuffers1 offsets and sizes must be multiples of 16 constants (256 bytes).
		constexpr UINT kAlignmentBytes = 256;
		// Largest engine buffer taken over: the most a single binding can expose (4096 constants).
		constexpr UINT kMaxManagedBytes = kMaxWindowConstants * kConstantBytes;
		constexpr UINT kRingBytes = 32u << 20;
		constexpr UINT kInvalid = UINT_MAX;
		// Frames after activation at which the redirection statistics are logged once.
		constexpr uint32_t kStatsLogFrame = 600;

		// Order matches the *SetConstantBuffers1 block of the ID3D11DeviceContext1 vtable.
		enum Stage : uint32_t
		{
			kVS,
			kHS,
			kDS,
			kGS,
			kPS,
			kCS,
			kStageCount
		};

		// A wrap re-uploads every bound buffer before finishing the upload that triggered it, so the
		// ring must hold one maximum-size buffer for every slot of every stage, plus that upload.
		static_assert(kRingBytes >= (kStageCount * kSlotCount + 1) * kMaxManagedBytes);
		static_assert(kSlotCount <= 16, "slot masks are 16 bits wide");

		// Shadow storage unit. Cache-line alignment exceeds the 16 bytes D3D11 guarantees for mapped
		// pointers, so the engine's vector stores and the copy into the ring never split a line.
		struct alignas(64) ShadowBlock
		{
			std::byte bytes[64];
		};
		static_assert(kAlignmentBytes % sizeof(ShadowBlock) == 0);

		using SetFunc = void (*)(ID3D11DeviceContext*, UINT, UINT, ID3D11Buffer* const*);
		using Set1Func = void (*)(ID3D11DeviceContext1*, UINT, UINT, ID3D11Buffer* const*, const UINT*, const UINT*);
		using GetFunc = void (*)(ID3D11DeviceContext*, UINT, UINT, ID3D11Buffer**);

		/** @brief A constant buffer whose contents live in the ring (an engine buffer, or one CS publishes via PublishOwned). */
		struct Record
		{
			ID3D11Buffer* buffer = nullptr;         // one reference is held so the address cannot be reused
			std::unique_ptr<ShadowBlock[]> shadow;  // latest contents; the engine writes here between Map and Unmap
			// byteWidth bytes of the latest upload when the ring cannot be read cheaply (WC VRAM / non-DXVK).
			std::unique_ptr<ShadowBlock[]> lastUpload;
			UINT byteWidth = 0;
			UINT allocBytes = 0;                             // byteWidth rounded up to kAlignmentBytes
			UINT numConstants = 0;                           // allocBytes / 16, so a multiple of 16 and at most 4096
			UINT firstConstant = 0;                          // ring offset of the latest upload, in constants
			uint32_t generation = 0;                         // ring generation of that upload; 0 = never uploaded
			std::array<uint16_t, kStageCount> boundSlots{};  // slots that logically hold this buffer
			uint32_t changeStreak = 0;                       // consecutive changed uploads; >= 16 enables direct ring writes
			uint32_t directUses = 0;                         // periodic shadow probe counter for direct-write buffers
			UINT directOffset = 0;                           // reserved ring byte offset while directMapped is true
			std::byte* directPtr = nullptr;                  // pointer returned to engine while directMapped is true
			bool mapped = false;
			bool directMapped = false;  // engine is writing directly into ringData at directOffset
			bool shadowStale = false;   // latest bytes live in ringData rather than shadow; synced lazily on wrap/deactivate
			bool owned = false;         // taken over by PublishOwned rather than by an engine map
			// Every slot in boundSlots has the record's latest upload bound (boundFirst/boundCount equal
			// RingWindow), so RebindRecord would bind nothing. Set by RebindRecord; cleared by CopyToRing
			// (new range) and ForgetBindings, false for a new record. ApplyBinds binds every slot it tracks
			// for a record right away, so binds keep it valid; Wrap rebinds every slot it invalidates.
			bool bindingsCurrent = false;
		};

		/** @brief What the application has bound to one stage slot. */
		struct Slot
		{
			ID3D11Buffer* logical = nullptr;  // the buffer the application bound (never the ring)
			Record* record = nullptr;         // set while `logical` is ring-managed
			UINT windowFirst = 0;             // FirstConstant passed to *SetConstantBuffers1 (0 for plain binds)
			UINT windowCount = kInvalid;      // NumConstants passed to *SetConstantBuffers1; kInvalid = whole buffer
			UINT boundFirst = kInvalid;       // ring window currently bound for `record`
			UINT boundCount = 0;
		};

		struct RingState
		{
			ID3D11DeviceContext* context = nullptr;
			winrt::com_ptr<ID3D11DeviceContext1> context1;
			winrt::com_ptr<ID3D11Buffer> ring;
			MapFunc map = nullptr;
			UnmapFunc unmap = nullptr;
			std::array<SetFunc, kStageCount> set{};
			std::array<Set1Func, kStageCount> set1{};
			std::array<GetFunc, kStageCount> get{};
			std::atomic<DWORD> renderThread{ 0 };
			std::uintptr_t engineTextBegin = 0;
			std::size_t engineTextSize = 0;

			bool supported = false;
			bool disabledForSession = false;
			bool uploadFailed = false;
			bool warnedThread = false;
			bool persistentMapping = false;  // DXVK: uploads write through ringData (see Wrap)
			bool ringReadProbed = false;
			bool ringReadable = false;  // true when ringData reads at host-cached RAM speed (skip lastUpload copy)
			std::atomic_bool active{ false };
			std::atomic_bool requested{ true };

			UINT cursor = kRingBytes;  // forces a discard before the first upload
			uint32_t generation = 0;
			std::byte* ringData = nullptr;  // pointer the latest discard returned while persistentMapping is on

			std::vector<std::unique_ptr<Record>> records;
			ankerl::unordered_dense::map<const void*, Record*> managed;
			// 128-entry direct-mapped L1 lookup cache (2 KiB) avoiding hash probes when alternating CBs.
			struct FindEntry
			{
				const void* key = nullptr;
				Record* record = nullptr;
			};
			static constexpr size_t kFindCacheMask = 127u;
			std::array<FindEntry, kFindCacheMask + 1u> findCache{};
			std::vector<std::unique_ptr<Record>> draining;  // still mapped by the engine when the ring went off
			uint32_t mappedCount = 0;
			std::array<std::array<Slot, kSlotCount>, kStageCount> slots{};

			uint32_t uploadsThisFrame = 0;
			uint32_t uploadsLastFrame = 0;
			uint32_t directUploadsThisFrame = 0;
			uint32_t dedupedThisFrame = 0;
			uint32_t rebindsSkippedThisFrame = 0;
			uint32_t ownedPublishesThisFrame = 0;
			uint64_t uploadsSinceActivation = 0;
			uint64_t directUploadsSinceActivation = 0;
			uint64_t dedupedSinceActivation = 0;
			uint64_t rebindsSkippedSinceActivation = 0;
			uint64_t ownedPublishesSinceActivation = 0;
			uint64_t wrapsSinceActivation = 0;
			uint32_t framesSinceActivation = 0;
			bool loggedStats = false;
		};

		// Leaked on purpose: the game can still call through these hooks after static destructors run.
		RingState& g = *new RingState();

		[[nodiscard]] bool OnRenderThread()
		{
			return GetCurrentThreadId() == g.renderThread.load(std::memory_order_relaxed);
		}

		/** @brief True when a call on `a_context` must go through the ring bookkeeping. */
		[[nodiscard]] bool Owns(const void* a_context)
		{
			// The thread check comes first: everything else is only touched on the render thread.
			return OnRenderThread() && g.active.load(std::memory_order_relaxed) &&
			       (a_context == g.context || a_context == g.context1.get());
		}

		[[nodiscard]] bool IsEngineCode(const void* a_address)
		{
			return reinterpret_cast<std::uintptr_t>(a_address) - g.engineTextBegin < g.engineTextSize;
		}

		/** @brief True on DXVK, recognised by its ID3D11VkExtDevice interface, which native drivers do not implement. */
		[[nodiscard]] bool IsDXVK(ID3D11Device* a_device)
		{
			// IID of ID3D11VkExtDevice, from DXVK's src/d3d11/d3d11_interfaces.h.
			constexpr GUID kVkExtDeviceIID{ 0x8a6e3c42, 0xf74c, 0x45b7, { 0x82, 0x65, 0xa2, 0x31, 0xb6, 0x77, 0xca, 0x17 } };
			void* extension = nullptr;
			if (FAILED(a_device->QueryInterface(kVkExtDeviceIID, &extension)) || !extension)
				return false;
			static_cast<IUnknown*>(extension)->Release();
			return true;
		}

		[[nodiscard]] Record* Find(const void* a_resource)
		{
			if (!a_resource)
				return nullptr;
			const uintptr_t h = reinterpret_cast<uintptr_t>(a_resource);
			auto& entry = g.findCache[((h >> 4) ^ (h >> 11)) & RingState::kFindCacheMask];
			if (entry.key == a_resource)
				return entry.record;
			const auto it = g.managed.find(a_resource);
			Record* rec = it != g.managed.end() ? it->second : nullptr;
			entry = { a_resource, rec };
			return rec;
		}

		void ForgetFindCache()
		{
			g.findCache.fill({});
		}

		[[nodiscard]] uint16_t SlotBit(UINT a_index)
		{
			return static_cast<uint16_t>(1u << a_index);
		}

		struct Window
		{
			UINT first;
			UINT count;
		};

		/** @brief The ring range that shows the slot's buffer as the application bound it. */
		[[nodiscard]] Window RingWindow(const Slot& a_slot)
		{
			const Record& record = *a_slot.record;
			const UINT offset = std::min(a_slot.windowFirst, record.numConstants);
			const UINT available = record.numConstants - offset;
			const UINT count = a_slot.windowCount == kInvalid ? available : std::min(a_slot.windowCount, available);
			return { record.firstConstant + offset, count };
		}

		void BindRingSlot(Stage a_stage, UINT a_index)
		{
			Slot& slot = g.slots[a_stage][a_index];
			const Window window = RingWindow(slot);
			if (slot.boundFirst == window.first && slot.boundCount == window.count)
				return;
			ID3D11Buffer* ring = g.ring.get();
			g.set1[a_stage](g.context1.get(), a_index, 1, &ring, &window.first, &window.count);
			slot.boundFirst = window.first;
			slot.boundCount = window.count;
		}

		void RebindRecord(Record& a_record)
		{
			for (uint32_t stage = 0; stage < kStageCount; ++stage) {
				for (uint32_t bits = a_record.boundSlots[stage]; bits != 0; bits &= bits - 1)
					BindRingSlot(static_cast<Stage>(stage), static_cast<UINT>(std::countr_zero(bits)));
			}
			a_record.bindingsCurrent = true;
		}

		void ProbeRingReadable(const std::byte* a_data)
		{
			g.ringReadProbed = true;
			LARGE_INTEGER freq{}, t0{}, t1{};
			if (!a_data || !QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) {
				g.ringReadable = false;
				return;
			}
			// Read 256 cache lines (16 KiB): host-cached RAM (=c) takes ~1-3 us; WC BAR/VRAM takes ~80-400 us.
			QueryPerformanceCounter(&t0);
			uint64_t acc = 0;
			const auto* p = reinterpret_cast<const volatile uint64_t*>(a_data);
			for (size_t i = 0; i < 256; ++i)
				acc += p[i * 8];
			QueryPerformanceCounter(&t1);
			(void)acc;
			const double us = static_cast<double>(t1.QuadPart - t0.QuadPart) * 1e6 / static_cast<double>(freq.QuadPart);
			g.ringReadable = us < 25.0;
			logger::info("[CB Ring] Ring read probe: {:.1f} us/16KiB -> ringReadable={} ({})",
				us, g.ringReadable, g.ringReadable ? "skip lastUpload copy" : "keep lastUpload copy");
		}

		void SyncStaleShadow(Record& a_record)
		{
			if (a_record.shadowStale && g.ringData && a_record.generation == g.generation) {
				std::memcpy(a_record.shadow.get(), g.ringData + a_record.firstConstant * kConstantBytes, a_record.byteWidth);
				a_record.shadowStale = false;
			}
		}

		/** @brief Copies the shadow into the ring at the cursor. Space must already be available. */
		void CopyToRing(Record& a_record)
		{
			if (g.cursor + a_record.allocBytes > kRingBytes) {
				g.uploadFailed = true;
				return;
			}
			if (g.ringData) {
				std::memcpy(g.ringData + g.cursor, a_record.shadow.get(), a_record.byteWidth);
			} else {
				D3D11_MAPPED_SUBRESOURCE mapped{};
				if (FAILED(g.map(g.context, g.ring.get(), 0, D3D11_MAP_WRITE_NO_OVERWRITE, 0, &mapped))) {
					g.uploadFailed = true;
					return;
				}
				if (mapped.pData)
					std::memcpy(static_cast<std::byte*>(mapped.pData) + g.cursor, a_record.shadow.get(), a_record.byteWidth);
				g.unmap(g.context, g.ring.get(), 0);
				if (!mapped.pData) {
					g.uploadFailed = true;
					return;
				}
			}
			// When ringData is host-cached RAM, dedupe compares directly against ringData[firstConstant] with no second copy.
			if (!g.ringReadable || !g.ringData)
				std::memcpy(a_record.lastUpload.get(), a_record.shadow.get(), a_record.byteWidth);
			a_record.shadowStale = false;
			a_record.firstConstant = g.cursor / kConstantBytes;
			a_record.generation = g.generation;
			a_record.bindingsCurrent = false;  // its slots still show the previous range
			g.cursor += a_record.allocBytes;
			++g.uploadsThisFrame;
		}

		/** @brief Discards the ring and re-uploads and rebinds every buffer that is still bound. */
		void Wrap()
		{
			// Before discarding the old ring storage, bring every direct-written record's shadow up to date so
			// wrap re-uploads and later binds after this wrap still have its latest bytes.
			if (g.ringData) {
				for (auto& record : g.records) {
					if (!record->mapped)
						SyncStaleShadow(*record);
				}
			}

			D3D11_MAPPED_SUBRESOURCE mapped{};
			if (FAILED(g.map(g.context, g.ring.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
				g.ringData = nullptr;
				g.uploadFailed = true;
				return;
			}
			// DXVK's buffer Unmap does nothing and its Map(WRITE_NO_OVERWRITE) returns the pointer of the latest
			// discard, so until the next wrap uploads write through this pointer instead of mapping per upload.
			// The ring is still unmapped right away, so no draw ever uses it while it is mapped.
			g.ringData = g.persistentMapping ? static_cast<std::byte*>(mapped.pData) : nullptr;
			g.unmap(g.context, g.ring.get(), 0);
			if (g.ringData && !g.ringReadProbed)
				ProbeRingReadable(g.ringData);
			g.cursor = 0;
			if (++g.generation == 0)
				g.generation = 1;
			++g.wrapsSinceActivation;

			// The discard gave the ring new storage, so every window bound from the previous storage now
			// reads undefined data. Bound buffers are uploaded again from their shadows and rebound.
			for (uint32_t stage = 0; stage < kStageCount; ++stage) {
				for (UINT index = 0; index < kSlotCount; ++index) {
					Slot& slot = g.slots[stage][index];
					if (!slot.record)
						continue;
					if (slot.record->generation != g.generation)
						CopyToRing(*slot.record);
					slot.boundFirst = kInvalid;
					BindRingSlot(static_cast<Stage>(stage), index);
				}
			}
		}

		/** @brief Makes the record's shadow visible to the GPU in a fresh ring range. */
		void Upload(Record& a_record)
		{
			if (g.cursor + a_record.allocBytes > kRingBytes) {
				Wrap();
				if (g.uploadFailed)
					return;
				// The wrap re-uploads every bound buffer from its current shadow, possibly this one.
				if (a_record.generation == g.generation)
					return;
			}
			CopyToRing(a_record);
		}

		/** @brief Gives the engine buffer itself the latest contents (used when the ring turns off). */
		void FlushToEngineBuffer(const Record& a_record)
		{
			D3D11_MAPPED_SUBRESOURCE mapped{};
			if (FAILED(g.map(g.context, a_record.buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
				logger::error("[CB Ring] Could not restore an engine constant buffer ({} bytes)", a_record.byteWidth);
				return;
			}
			const void* src = (a_record.directMapped && a_record.directPtr) ? static_cast<const void*>(a_record.directPtr) : static_cast<const void*>(a_record.shadow.get());
			if (mapped.pData)
				std::memcpy(mapped.pData, src, a_record.byteWidth);
			g.unmap(g.context, a_record.buffer, 0);
		}

		/** @brief Records a binding the application made and keeps the records' slot masks in sync. */
		void Track(Stage a_stage, UINT a_index, ID3D11Buffer* a_buffer, Record* a_record, UINT a_first, UINT a_count)
		{
			Slot& slot = g.slots[a_stage][a_index];
			const uint16_t bit = SlotBit(a_index);
			if (slot.record && slot.record != a_record)
				slot.record->boundSlots[a_stage] = static_cast<uint16_t>(slot.record->boundSlots[a_stage] & ~bit);
			if (a_record) {
				a_record->boundSlots[a_stage] = static_cast<uint16_t>(a_record->boundSlots[a_stage] | bit);
			} else {
				// The slot now holds an ordinary buffer, so no ring window is bound there.
				slot.boundFirst = kInvalid;
				slot.boundCount = 0;
			}
			slot.logical = a_buffer;
			slot.record = a_record;
			slot.windowFirst = a_first;
			slot.windowCount = a_count;
		}

		void ForgetBindings()
		{
			for (auto& stageSlots : g.slots)
				stageSlots.fill(Slot{});
			for (auto& record : g.records) {
				record->boundSlots.fill(0);
				record->bindingsCurrent = false;
			}
		}

		/** @brief Takes over an engine buffer the game executable is mapping with WRITE_DISCARD. */
		Record* Manage(ID3D11Resource* a_resource)
		{
			D3D11_RESOURCE_DIMENSION dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN;
			a_resource->GetType(&dimension);
			if (dimension != D3D11_RESOURCE_DIMENSION_BUFFER)
				return nullptr;

			auto* buffer = static_cast<ID3D11Buffer*>(a_resource);
			// The per-frame buffer stays on the original path: CacheFramebuffer reads it through the Map hook.
			if (buffer == *globals::game::perFrame.get() || buffer == g.ring.get())
				return nullptr;

			D3D11_BUFFER_DESC desc{};
			buffer->GetDesc(&desc);
			if (desc.Usage != D3D11_USAGE_DYNAMIC || desc.BindFlags != static_cast<UINT>(D3D11_BIND_CONSTANT_BUFFER) ||
				(desc.CPUAccessFlags & static_cast<UINT>(D3D11_CPU_ACCESS_WRITE)) == 0 || desc.MiscFlags != 0 ||
				desc.ByteWidth == 0 || desc.ByteWidth > kMaxManagedBytes)
				return nullptr;

			auto record = std::make_unique<Record>();
			record->byteWidth = desc.ByteWidth;
			record->allocBytes = (desc.ByteWidth + kAlignmentBytes - 1) & ~(kAlignmentBytes - 1);
			record->numConstants = record->allocBytes / kConstantBytes;
			record->shadow = std::make_unique<ShadowBlock[]>(record->allocBytes / sizeof(ShadowBlock));
			record->lastUpload = std::make_unique<ShadowBlock[]>(record->allocBytes / sizeof(ShadowBlock));
			buffer->AddRef();
			record->buffer = buffer;

			// Slots that already hold this buffer from earlier binds must now follow its ring uploads.
			for (uint32_t stage = 0; stage < kStageCount; ++stage) {
				for (UINT index = 0; index < kSlotCount; ++index) {
					Slot& slot = g.slots[stage][index];
					if (slot.logical != buffer)
						continue;
					slot.record = record.get();
					slot.boundFirst = kInvalid;
					slot.boundCount = 0;
					record->boundSlots[stage] = static_cast<uint16_t>(record->boundSlots[stage] | SlotBit(index));
				}
			}

			Record* raw = record.get();
			g.managed.emplace(static_cast<const void*>(buffer), raw);
			ForgetFindCache();  // it may hold this buffer as unmanaged
			g.records.push_back(std::move(record));
			return raw;
		}

		void Activate()
		{
			// Start from the bindings the application already made; buffers bound before activation are
			// picked up by Manage() when the game next maps them.
			for (uint32_t stage = 0; stage < kStageCount; ++stage) {
				std::array<ID3D11Buffer*, kSlotCount> bound{};
				g.get[stage](g.context, 0, kSlotCount, bound.data());
				for (UINT index = 0; index < kSlotCount; ++index) {
					Slot& slot = g.slots[stage][index];
					slot = Slot{};
					slot.logical = bound[index];
					if (bound[index])
						bound[index]->Release();  // the context keeps the buffer alive while it is bound
				}
			}
			g.cursor = kRingBytes;  // the first upload discards, so no earlier storage is reused under NO_OVERWRITE
			g.ringData = nullptr;
			g.uploadFailed = false;
			g.uploadsSinceActivation = 0;
			g.directUploadsSinceActivation = 0;
			g.dedupedSinceActivation = 0;
			g.rebindsSkippedSinceActivation = 0;
			g.ownedPublishesSinceActivation = 0;
			g.wrapsSinceActivation = 0;
			g.framesSinceActivation = 0;
			g.active.store(true, std::memory_order_relaxed);
			logger::info("[CB Ring] Engine constant buffer ring active ({} MiB)", kRingBytes >> 20);
		}

		void Deactivate(const char* a_reason)
		{
			// Point every ring-managed binding back at its engine buffer, as the application bound it...
			for (uint32_t stage = 0; stage < kStageCount; ++stage) {
				for (UINT index = 0; index < kSlotCount; ++index) {
					Slot& slot = g.slots[stage][index];
					if (!slot.record)
						continue;
					ID3D11Buffer* buffer = slot.logical;
					if (slot.windowCount == kInvalid)
						g.set[stage](g.context, index, 1, &buffer);
					else
						g.set1[stage](g.context1.get(), index, 1, &buffer, &slot.windowFirst, &slot.windowCount);
					slot.record = nullptr;
					slot.boundFirst = kInvalid;
					slot.boundCount = 0;
				}
			}

			// ...and give each engine buffer its latest contents, because buffers such as the alpha-test
			// reference or per-technique groups are bound again without being mapped first.
			for (auto& record : g.records) {
				if (record->mapped) {
					g.draining.push_back(std::move(record));  // finished by its Unmap
					continue;
				}
				SyncStaleShadow(*record);
				FlushToEngineBuffer(*record);
				record->buffer->Release();
			}
			g.records.clear();
			g.managed.clear();
			ForgetFindCache();
			g.mappedCount = 0;
			g.ringData = nullptr;
			g.active.store(false, std::memory_order_relaxed);
			logger::info("[CB Ring] Engine constant buffer ring off ({})", a_reason);
		}

		void DisableForSession(const char* a_reason)
		{
			g.disabledForSession = true;
			if (g.active.load(std::memory_order_relaxed))
				Deactivate(a_reason);
			else
				logger::info("[CB Ring] Engine constant buffer ring disabled for this session ({})", a_reason);
		}

		/**
		 * @brief Makes the record's shadow current on every slot that holds it, as the Unmap of a redirected map does.
		 * @return False when a ring map failed. The ring is then off for the session (the record is destroyed)
		 *         and deactivation has written the shadow into the buffer itself.
		 */
		[[nodiscard]] bool Publish(Record& a_record)
		{
			// Contents equal to the latest upload, whose range no wrap has discarded since, reuse that range.
			const void* prevData = (g.ringReadable && g.ringData) ?
			                           static_cast<const void*>(g.ringData + a_record.firstConstant * kConstantBytes) :
			                           static_cast<const void*>(a_record.lastUpload.get());
			if (a_record.generation != 0 && a_record.generation == g.generation && !a_record.shadowStale &&
				std::memcmp(a_record.shadow.get(), prevData, a_record.byteWidth) == 0) {
				a_record.changeStreak = 0;
				++g.dedupedThisFrame;
				// Nothing changed since every slot of the record was last bound to that range.
				if (a_record.bindingsCurrent) {
					++g.rebindsSkippedThisFrame;
					return true;
				}
			} else {
				++a_record.changeStreak;
				Upload(a_record);
				if (g.uploadFailed) {
					DisableForSession("a ring map failed");
					return false;
				}
			}
			RebindRecord(a_record);
			return true;
		}

		void CallOriginal(Stage a_stage, UINT a_start, UINT a_count, ID3D11Buffer* const* a_buffers, const UINT* a_first,
			const UINT* a_num, bool a_viaSet1)
		{
			if (a_viaSet1)
				g.set1[a_stage](g.context1.get(), a_start, a_count, a_buffers, a_first, a_num);
			else
				g.set[a_stage](g.context, a_start, a_count, a_buffers);
		}

		/** @brief Shared body of the *SetConstantBuffers and *SetConstantBuffers1 hooks. */
		void ApplyBinds(Stage a_stage, UINT a_start, UINT a_count, ID3D11Buffer* const* a_buffers, const UINT* a_first,
			const UINT* a_num, bool a_viaSet1)
		{
			// Without both arrays the runtime ignores the offsets, as for a plain bind.
			const bool hasWindows = a_viaSet1 && a_first && a_num;
			std::array<Record*, kSlotCount> records{};
			bool anyManaged = false;
			for (UINT i = 0; i < a_count; ++i) {
				records[i] = Find(a_buffers[i]);
				anyManaged = anyManaged || records[i] != nullptr;
			}

			if (!anyManaged) {
				// Common case: nothing ring-managed, so forward the call unchanged and only record it.
				CallOriginal(a_stage, a_start, a_count, a_buffers, a_first, a_num, a_viaSet1);
				for (UINT i = 0; i < a_count; ++i) {
					if (hasWindows && a_num[i] > kMaxWindowConstants)
						continue;  // the runtime skips such slots and keeps their previous binding
					Track(a_stage, a_start + i, a_buffers[i], nullptr, hasWindows ? a_first[i] : 0, hasWindows ? a_num[i] : kInvalid);
				}
				return;
			}

			// A buffer whose latest upload predates a ring wrap must be uploaded again before it is bound.
			for (UINT i = 0; i < a_count; ++i) {
				if (records[i] && records[i]->generation != g.generation)
					Upload(*records[i]);
			}
			if (g.uploadFailed) {
				DisableForSession("a ring map failed");
				CallOriginal(a_stage, a_start, a_count, a_buffers, a_first, a_num, a_viaSet1);
				return;
			}

			for (UINT i = 0; i < a_count; ++i) {
				if (hasWindows && a_num[i] > kMaxWindowConstants)
					continue;
				const UINT index = a_start + i;
				const UINT first = hasWindows ? a_first[i] : 0;
				const UINT count = hasWindows ? a_num[i] : kInvalid;
				if (Record* record = records[i]) {
					// The slot is bound to the record's latest upload right away and no other slot of the
					// record changes, so record->bindingsCurrent stays valid.
					Track(a_stage, index, a_buffers[i], record, first, count);
					BindRingSlot(a_stage, index);
				} else {
					CallOriginal(a_stage, index, 1, &a_buffers[i], hasWindows ? &a_first[i] : nullptr, hasWindows ? &a_num[i] : nullptr, a_viaSet1);
					Track(a_stage, index, a_buffers[i], nullptr, first, count);
				}
			}
		}

		/**
		 * @brief Replaces ring windows in a *GetConstantBuffers result with the buffers the application bound,
		 * so state save/restore code (ImGui, Effects11) restores those buffers rather than the ring.
		 */
		void ReportApplicationBindings(Stage a_stage, UINT a_start, UINT a_count, ID3D11Buffer** a_buffers, UINT* a_first, UINT* a_num)
		{
			if (a_start >= kSlotCount)
				return;
			const UINT count = std::min(a_count, kSlotCount - a_start);
			for (UINT i = 0; i < count; ++i) {
				const Slot& slot = g.slots[a_stage][a_start + i];
				if (!slot.record)
					continue;
				if (a_buffers) {
					if (a_buffers[i] != g.ring.get())
						continue;
					a_buffers[i]->Release();
					a_buffers[i] = slot.logical;
					slot.logical->AddRef();
				}
				if (a_first)
					a_first[i] = slot.windowFirst;
				if (a_num)
					a_num[i] = slot.windowCount == kInvalid ? std::min(slot.record->byteWidth / kConstantBytes, kMaxWindowConstants) : slot.windowCount;
			}
		}

		[[nodiscard]] bool ValidRange(UINT a_start, UINT a_count)
		{
			return a_start < kSlotCount && a_count <= kSlotCount - a_start;
		}

		template <Stage S>
		struct SetConstantBuffers
		{
			static void thunk(ID3D11DeviceContext* This, UINT StartSlot, UINT NumBuffers, ID3D11Buffer* const* ppConstantBuffers)
			{
				if (!ppConstantBuffers || !ValidRange(StartSlot, NumBuffers) || !Owns(This)) {
					func(This, StartSlot, NumBuffers, ppConstantBuffers);
					return;
				}
				if (NumBuffers == 1) {
					ID3D11Buffer* buf = ppConstantBuffers[0];
					Slot& slot = g.slots[S][StartSlot];
					if (slot.logical == buf && slot.record && slot.windowFirst == 0 && slot.windowCount == kInvalid) {
						Record* rec = slot.record;
						if (rec->generation == g.generation && slot.boundFirst == rec->firstConstant && slot.boundCount == rec->numConstants)
							return;
					}
					Record* rec = Find(buf);
					if (!rec) {
						func(This, StartSlot, 1, ppConstantBuffers);
						if (slot.record) {
							slot.record->boundSlots[S] = static_cast<uint16_t>(slot.record->boundSlots[S] & ~SlotBit(StartSlot));
							slot.record = nullptr;
							slot.boundFirst = kInvalid;
							slot.boundCount = 0;
						}
						slot.logical = buf;
						slot.windowFirst = 0;
						slot.windowCount = kInvalid;
						return;
					}
					if (rec->generation != g.generation) {
						Upload(*rec);
						if (g.uploadFailed) {
							DisableForSession("a ring map failed");
							func(This, StartSlot, 1, ppConstantBuffers);
							return;
						}
					}
					Track(S, StartSlot, buf, rec, 0, kInvalid);
					BindRingSlot(S, StartSlot);
					return;
				}
				ApplyBinds(S, StartSlot, NumBuffers, ppConstantBuffers, nullptr, nullptr, false);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		template <Stage S>
		struct SetConstantBuffers1
		{
			static void thunk(ID3D11DeviceContext1* This, UINT StartSlot, UINT NumBuffers, ID3D11Buffer* const* ppConstantBuffers,
				const UINT* pFirstConstant, const UINT* pNumConstants)
			{
				if (!ppConstantBuffers || !ValidRange(StartSlot, NumBuffers) || !Owns(This)) {
					func(This, StartSlot, NumBuffers, ppConstantBuffers, pFirstConstant, pNumConstants);
					return;
				}
				ApplyBinds(S, StartSlot, NumBuffers, ppConstantBuffers, pFirstConstant, pNumConstants, true);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		template <Stage S>
		struct GetConstantBuffers
		{
			static void thunk(ID3D11DeviceContext* This, UINT StartSlot, UINT NumBuffers, ID3D11Buffer** ppConstantBuffers)
			{
				func(This, StartSlot, NumBuffers, ppConstantBuffers);
				if (ppConstantBuffers && Owns(This))
					ReportApplicationBindings(S, StartSlot, NumBuffers, ppConstantBuffers, nullptr, nullptr);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		template <Stage S>
		struct GetConstantBuffers1
		{
			static void thunk(ID3D11DeviceContext1* This, UINT StartSlot, UINT NumBuffers, ID3D11Buffer** ppConstantBuffers,
				UINT* pFirstConstant, UINT* pNumConstants)
			{
				func(This, StartSlot, NumBuffers, ppConstantBuffers, pFirstConstant, pNumConstants);
				if (Owns(This))
					ReportApplicationBindings(S, StartSlot, NumBuffers, ppConstantBuffers, pFirstConstant, pNumConstants);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct CopyResourceHook
		{
			static void thunk(ID3D11DeviceContext* This, ID3D11Resource* pDstResource, ID3D11Resource* pSrcResource)
			{
				// Vanilla Skyrim performs several full-resolution CopyResource calls on kMAIN depth
				// (kMAIN -> kPOST_ZPREPASS_COPY / kMAIN_COPY) before upscaling while dynamic resolution
				// only renders the top-left viewport. On DXVK, sub-region D24S8 copies work cleanly and
				// cut copy bandwidth by ~4x at 50% scale.
				if (g.persistentMapping && This == g.context && OnRenderThread() && pDstResource && pSrcResource && pDstResource != pSrcResource) {
					auto* viewport = globals::game::graphicsState;
					auto* renderer = globals::game::renderer;
					if (viewport && renderer) {
						const auto& rt = viewport->GetRuntimeData();
						if (!rt.dynamicResolutionLock && rt.dynamicResolutionWidthRatio > 0.0f && rt.dynamicResolutionHeightRatio > 0.0f &&
							(rt.dynamicResolutionWidthRatio < 0.995f || rt.dynamicResolutionHeightRatio < 0.995f)) {
							auto& ds = renderer->GetDepthStencilData().depthStencils;
							if (pSrcResource == ds[RE::RENDER_TARGETS_DEPTHSTENCIL::kMAIN].texture &&
								(pDstResource == ds[RE::RENDER_TARGETS_DEPTHSTENCIL::kPOST_ZPREPASS_COPY].texture ||
									pDstResource == ds[RE::RENDER_TARGETS_DEPTHSTENCIL::kMAIN_COPY].texture)) {
								const UINT w = static_cast<UINT>(viewport->screenWidth);
								const UINT h = static_cast<UINT>(viewport->screenHeight);
								if (w > 0 && h > 0) {
									D3D11_BOX box{};
									box.left = 0;
									box.top = 0;
									box.front = 0;
									box.right = std::min(w, static_cast<UINT>(std::ceil(static_cast<float>(w) * rt.dynamicResolutionWidthRatio)) + 4u);
									box.bottom = std::min(h, static_cast<UINT>(std::ceil(static_cast<float>(h) * rt.dynamicResolutionHeightRatio)) + 4u);
									box.back = 1;
									This->CopySubresourceRegion(pDstResource, 0, 0, 0, 0, pSrcResource, 0, &box);
									return;
								}
							}
						}
					}
				}
				func(This, pDstResource, pSrcResource);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct ClearState
		{
			static void thunk(ID3D11DeviceContext* This)
			{
				func(This);
				if (Owns(This))
					ForgetBindings();
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct ExecuteCommandList
		{
			static void thunk(ID3D11DeviceContext* This, ID3D11CommandList* pCommandList, BOOL RestoreContextState)
			{
				func(This, pCommandList, RestoreContextState);
				// Without RestoreContextState the immediate context is left with nothing bound.
				if (!RestoreContextState && Owns(This))
					ForgetBindings();
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		struct SwapDeviceContextState
		{
			static void thunk(ID3D11DeviceContext1* This, ID3DDeviceContextState* pState, ID3DDeviceContextState** ppPreviousState)
			{
				// Bindings live in the swapped state objects, which the tracking cannot follow.
				if (OnRenderThread() && This == g.context1.get() && !g.disabledForSession)
					DisableForSession("SwapDeviceContextState is in use");
				func(This, pState, ppPreviousState);
			}
			static inline REL::Relocation<decltype(thunk)> func;
		};

		[[nodiscard]] bool InstallHooks(ID3D11DeviceContext* a_context, ID3D11DeviceContext1* a_context1)
		{
			// ID3D11DeviceContext / ID3D11DeviceContext1 vtable slots, as laid out in d3d11.h and d3d11_1.h.
			const std::array results{
				DetourVFunc<7, SetConstantBuffers<kVS>>(a_context),
				DetourVFunc<16, SetConstantBuffers<kPS>>(a_context),
				DetourVFunc<22, SetConstantBuffers<kGS>>(a_context),
				DetourVFunc<62, SetConstantBuffers<kHS>>(a_context),
				DetourVFunc<66, SetConstantBuffers<kDS>>(a_context),
				DetourVFunc<71, SetConstantBuffers<kCS>>(a_context),
				DetourVFunc<72, GetConstantBuffers<kVS>>(a_context),
				DetourVFunc<77, GetConstantBuffers<kPS>>(a_context),
				DetourVFunc<81, GetConstantBuffers<kGS>>(a_context),
				DetourVFunc<100, GetConstantBuffers<kHS>>(a_context),
				DetourVFunc<104, GetConstantBuffers<kDS>>(a_context),
				DetourVFunc<109, GetConstantBuffers<kCS>>(a_context),
				DetourVFunc<47, CopyResourceHook>(a_context),
				DetourVFunc<58, ExecuteCommandList>(a_context),
				DetourVFunc<110, ClearState>(a_context),
				DetourVFunc<119, SetConstantBuffers1<kVS>>(a_context1),
				DetourVFunc<120, SetConstantBuffers1<kHS>>(a_context1),
				DetourVFunc<121, SetConstantBuffers1<kDS>>(a_context1),
				DetourVFunc<122, SetConstantBuffers1<kGS>>(a_context1),
				DetourVFunc<123, SetConstantBuffers1<kPS>>(a_context1),
				DetourVFunc<124, SetConstantBuffers1<kCS>>(a_context1),
				DetourVFunc<125, GetConstantBuffers1<kVS>>(a_context1),
				DetourVFunc<126, GetConstantBuffers1<kHS>>(a_context1),
				DetourVFunc<127, GetConstantBuffers1<kDS>>(a_context1),
				DetourVFunc<128, GetConstantBuffers1<kGS>>(a_context1),
				DetourVFunc<129, GetConstantBuffers1<kPS>>(a_context1),
				DetourVFunc<130, GetConstantBuffers1<kCS>>(a_context1),
				DetourVFunc<131, SwapDeviceContextState>(a_context1),
			};

			// Trampolines to the original implementations, used for every call the ring makes itself.
			g.set = { SetConstantBuffers<kVS>::func.get(), SetConstantBuffers<kHS>::func.get(), SetConstantBuffers<kDS>::func.get(),
				SetConstantBuffers<kGS>::func.get(), SetConstantBuffers<kPS>::func.get(), SetConstantBuffers<kCS>::func.get() };
			g.set1 = { SetConstantBuffers1<kVS>::func.get(), SetConstantBuffers1<kHS>::func.get(), SetConstantBuffers1<kDS>::func.get(),
				SetConstantBuffers1<kGS>::func.get(), SetConstantBuffers1<kPS>::func.get(), SetConstantBuffers1<kCS>::func.get() };
			g.get = { GetConstantBuffers<kVS>::func.get(), GetConstantBuffers<kHS>::func.get(), GetConstantBuffers<kDS>::func.get(),
				GetConstantBuffers<kGS>::func.get(), GetConstantBuffers<kPS>::func.get(), GetConstantBuffers<kCS>::func.get() };

			return std::ranges::all_of(results, std::identity{});
		}
	}

	void Install(ID3D11Device* a_device, ID3D11DeviceContext* a_context, MapFunc a_originalMap, UnmapFunc a_originalUnmap)
	{
		if (g.context || !a_device || !a_context || !a_originalMap || !a_originalUnmap)
			return;

		g.context = a_context;
		g.map = a_originalMap;
		g.unmap = a_originalUnmap;
		g.renderThread.store(GetCurrentThreadId(), std::memory_order_relaxed);
		const auto text = REL::Module::get().segment(REL::Segment::textx);
		g.engineTextBegin = text.address();
		g.engineTextSize = text.size();

		D3D11_FEATURE_DATA_D3D11_OPTIONS options{};
		if (FAILED(a_device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &options, sizeof(options))) ||
			!options.ConstantBufferOffsetting || !options.MapNoOverwriteOnDynamicConstantBuffer) {
			logger::info("[CB Ring] Unavailable: the device does not report ConstantBufferOffsetting and MapNoOverwriteOnDynamicConstantBuffer; engine constant buffers stay on WRITE_DISCARD");
			return;
		}
		if (FAILED(a_context->QueryInterface(IID_PPV_ARGS(g.context1.put())))) {
			logger::info("[CB Ring] Unavailable: ID3D11DeviceContext1 is not supported; engine constant buffers stay on WRITE_DISCARD");
			return;
		}
		if (g.engineTextSize == 0) {
			logger::error("[CB Ring] Unavailable: the game executable's code segment is unknown");
			return;
		}

		D3D11_BUFFER_DESC desc{};
		desc.ByteWidth = kRingBytes;
		desc.Usage = D3D11_USAGE_DYNAMIC;
		desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		if (FAILED(a_device->CreateBuffer(&desc, nullptr, g.ring.put()))) {
			logger::error("[CB Ring] Unavailable: could not create the {} MiB ring buffer", kRingBytes >> 20);
			return;
		}
		Util::SetResourceName(g.ring.get(), "ConstantBufferRing::Ring");

		if (!InstallHooks(a_context, g.context1.get())) {
			logger::error("[CB Ring] Unavailable: a constant buffer binding hook could not be installed; engine constant buffers stay on WRITE_DISCARD");
			return;
		}

		g.persistentMapping = IsDXVK(a_device);
		g.supported = true;
		logger::info("[CB Ring] {} MiB engine constant buffer ring ready (ConstantBufferOffsetting, MapNoOverwriteOnDynamicConstantBuffer{}); {} by the \"Constant Buffer Ring\" setting",
			kRingBytes >> 20, g.persistentMapping ? ", DXVK: uploads write through the pointer of each ring discard" : "",
			g.requested.load(std::memory_order_relaxed) ? "enabled" : "disabled");
	}

	void SetEnabled(bool a_enabled)
	{
		g.requested.store(a_enabled, std::memory_order_relaxed);
	}

	bool IsEnabled()
	{
		return g.requested.load(std::memory_order_relaxed);
	}

	Status GetStatus()
	{
		Status status;
		status.supported = g.supported && !g.disabledForSession;
		status.active = g.active.load(std::memory_order_relaxed);
		if (OnRenderThread()) {
			status.managedBuffers = static_cast<uint32_t>(g.records.size());
			status.uploadsLastFrame = g.uploadsLastFrame;
		}
		return status;
	}

	bool IsDXVK()
	{
		return g.persistentMapping;
	}

	void OnFrameBoundary()
	{
		if (!g.supported)
			return;
		if (!OnRenderThread()) {
			if (!g.warnedThread) {
				g.warnedThread = true;
				logger::warn("[CB Ring] Present runs on thread {} but the device was created on thread {}; the ring stays off",
					GetCurrentThreadId(), g.renderThread.load(std::memory_order_relaxed));
			}
			return;
		}

		g.uploadsLastFrame = std::exchange(g.uploadsThisFrame, 0);
		g.uploadsSinceActivation += g.uploadsLastFrame;
		g.directUploadsSinceActivation += std::exchange(g.directUploadsThisFrame, 0);
		g.dedupedSinceActivation += std::exchange(g.dedupedThisFrame, 0);
		g.rebindsSkippedSinceActivation += std::exchange(g.rebindsSkippedThisFrame, 0);
		g.ownedPublishesSinceActivation += std::exchange(g.ownedPublishesThisFrame, 0);

		if (g.uploadFailed && g.active.load(std::memory_order_relaxed))
			DisableForSession("a ring map failed");

		const bool wanted = g.requested.load(std::memory_order_relaxed) && !g.disabledForSession;
		const bool active = g.active.load(std::memory_order_relaxed);
		if (wanted && !active)
			Activate();
		else if (!wanted && active)
			Deactivate("disabled by the \"Constant Buffer Ring\" setting");

		if (g.active.load(std::memory_order_relaxed) && !g.loggedStats && ++g.framesSinceActivation == kStatsLogFrame) {
			g.loggedStats = true;
			const auto ownedBuffers = std::ranges::count_if(g.records, [](const auto& a_record) { return a_record->owned; });
			logger::info("[CB Ring] After {} frames: {} engine CBs, {:.0f} uploads/frame ({:.0f} direct zero-copy), {:.0f} deduped/frame, {:.0f} rebinds skipped/frame, {} wraps, {:.0f} owned publishes/frame ({} CS CBs), ringReadable={}",
				kStatsLogFrame, g.records.size() - static_cast<std::size_t>(ownedBuffers), static_cast<double>(g.uploadsSinceActivation) / kStatsLogFrame,
				static_cast<double>(g.directUploadsSinceActivation) / kStatsLogFrame,
				static_cast<double>(g.dedupedSinceActivation) / kStatsLogFrame,
				static_cast<double>(g.rebindsSkippedSinceActivation) / kStatsLogFrame, g.wrapsSinceActivation,
				static_cast<double>(g.ownedPublishesSinceActivation) / kStatsLogFrame, ownedBuffers, g.ringReadable);
		}
	}

	bool TryMap(ID3D11DeviceContext* a_context, ID3D11Resource* a_resource, UINT a_subresource, D3D11_MAP a_mapType,
		D3D11_MAPPED_SUBRESOURCE* a_mapped, const void* a_caller)
	{
		if (!Owns(a_context) || !a_resource || !a_mapped || a_subresource != 0)
			return false;
		if (a_mapType != D3D11_MAP_WRITE_DISCARD && a_mapType != D3D11_MAP_WRITE_NO_OVERWRITE)
			return false;

		Record* record = Find(a_resource);
		if (!record) {
			// Only buffers the game executable maps with WRITE_DISCARD are taken over.
			if (a_mapType != D3D11_MAP_WRITE_DISCARD || !IsEngineCode(a_caller))
				return false;
			record = Manage(a_resource);
			if (!record)
				return false;
		}

		// High-churn engine CBs (per-object transforms that rarely repeat): hand the engine a fresh slice of
		// ringData directly on WRITE_DISCARD so Map+Unmap does zero memcmp and zero memcpy. Keep 128 KiB of
		// wrap-free headroom while the pointer is out, and probe through shadow every 256th map so a buffer
		// that stops churning (e.g. interior/menu) re-enables dedupe cleanly.
		if (a_mapType == D3D11_MAP_WRITE_DISCARD && !record->owned && !record->mapped &&
			record->changeStreak >= 16 && g.ringData != nullptr && g.mappedCount == 0 &&
			(g.cursor + record->allocBytes + (kMaxManagedBytes * 2) <= kRingBytes)) {
			if (++record->directUses >= 256) {
				record->directUses = 0;
				record->changeStreak = 14;  // two shadow cycles: first refreshes baseline, second checks dedupe
			} else {
				record->directOffset = g.cursor;
				g.cursor += record->allocBytes;
				record->directPtr = g.ringData + record->directOffset;
				record->directMapped = true;
				record->mapped = true;
				++g.mappedCount;
				a_mapped->pData = record->directPtr;
				a_mapped->RowPitch = record->byteWidth;
				a_mapped->DepthPitch = record->byteWidth;
				return true;
			}
		}

		// WRITE_NO_OVERWRITE expects the latest contents to still be in the mapped buffer.
		if (a_mapType == D3D11_MAP_WRITE_NO_OVERWRITE && record->shadowStale)
			SyncStaleShadow(*record);

		if (!record->mapped) {
			record->mapped = true;
			++g.mappedCount;
		}
		a_mapped->pData = record->shadow.get();
		a_mapped->RowPitch = record->byteWidth;
		a_mapped->DepthPitch = record->byteWidth;
		return true;
	}

	bool TryUnmap(ID3D11DeviceContext* a_context, ID3D11Resource* a_resource, UINT a_subresource)
	{
		if (!OnRenderThread() || a_context != g.context || a_subresource != 0)
			return false;
		if (g.mappedCount == 0 && g.draining.empty())
			return false;

		if (Record* record = Find(a_resource); record && record->mapped) {
			record->mapped = false;
			--g.mappedCount;
			if (record->directMapped) {
				record->directMapped = false;
				record->directPtr = nullptr;
				record->firstConstant = record->directOffset / kConstantBytes;
				record->generation = g.generation;
				record->bindingsCurrent = false;
				record->shadowStale = true;
				++g.uploadsThisFrame;
				++g.directUploadsThisFrame;
				RebindRecord(*record);
				return true;
			}
			// When a ring map fails, deactivation writes the shadow into the engine buffer itself, so this
			// update still lands either way.
			(void)Publish(*record);
			return true;
		}

		// Buffers that were mapped when the ring went off get their contents at their Unmap.
		const auto draining = std::ranges::find_if(g.draining, [&](const auto& a_record) { return a_record->buffer == a_resource; });
		if (draining == g.draining.end())
			return false;
		FlushToEngineBuffer(**draining);
		(*draining)->buffer->Release();
		g.draining.erase(draining);
		return true;
	}

	bool PublishOwned(ID3D11Buffer* a_buffer, const void* a_data, UINT a_size)
	{
		// The thread check comes first: the ring state is only touched on the render thread.
		if (!OnRenderThread() || !g.active.load(std::memory_order_relaxed) || !a_buffer || !a_data || a_size == 0)
			return false;

		Record* record = Find(a_buffer);
		if (!record) {
			// Manage() also picks up the slots this buffer is already bound to, so the publish below rebinds them.
			record = Manage(a_buffer);
			if (!record)
				return false;
			record->owned = true;
		}
		// A map of the buffer through the Map hook is in flight (its Unmap publishes), or the data does not fit.
		if (record->mapped || a_size > record->byteWidth)
			return false;

		std::memcpy(record->shadow.get(), a_data, a_size);
		++g.ownedPublishesThisFrame;
		// On failure the ring is off and the record gone; the caller's own update then reaches the buffer directly.
		return Publish(*record);
	}
}
