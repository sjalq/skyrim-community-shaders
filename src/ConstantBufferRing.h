#pragma once

#include <d3d11.h>

/**
 * @brief Redirects the engine's per-draw constant buffer updates into one large dynamic ring buffer.
 *
 * Skyrim refreshes its ConstantGroup buffers with Map(WRITE_DISCARD) before nearly every draw. Under
 * DXVK each discard renames the buffer (a fresh slice from DxvkLocalAllocationCache plus reference
 * count churn when the renamed buffer is rebound), which costs the main thread milliseconds per frame.
 *
 * While the ring is active, a Map(WRITE_DISCARD) that the game executable issues on a dynamic constant
 * buffer returns a CPU shadow copy of that buffer instead. Unmap copies the shadow into the next
 * 256-byte-aligned range of the ring (mapped with WRITE_NO_OVERWRITE) and binds that range with
 * *SetConstantBuffers1 to every stage/slot where the engine buffer is bound. The ring itself is only
 * discarded when it wraps; every buffer still bound at that point is uploaded again and rebound.
 * An Unmap whose shadow equals the buffer's latest upload, while no wrap has discarded that range, binds
 * the existing range instead of copying again, and binds nothing when every slot already shows it.
 * Under DXVK, whose buffer Unmap does nothing and whose WRITE_NO_OVERWRITE maps return the pointer of
 * the latest discard, uploads write through the pointer the wrap's discard returned instead of mapping
 * the ring for every upload.
 *
 * Every constant buffer binding entry point of the immediate context is hooked so D3D11 semantics hold:
 * binds of ring-managed buffers become ring ranges, *GetConstantBuffers report the buffers the
 * application bound (never the ring), and ClearState / ExecuteCommandList(FALSE) reset the tracking.
 * SwapDeviceContextState turns the ring off for the session because the tracking cannot follow
 * swapped state objects.
 *
 * Scope and thread model, matching the assumptions the Map/Unmap hooks already made:
 * - Only the thread that created the device (the thread the game renders on) is redirected; other
 *   threads and deferred contexts pass straight through.
 * - The engine's per-frame buffer stays on the original path so CacheFramebuffer keeps working.
 * - Buffers are only taken over when the game executable maps them first, so Community Shaders'
 *   own buffers, ImGui's and other plugins' are never managed.
 */
namespace ConstantBufferRing
{
	using MapFunc = HRESULT (*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT, D3D11_MAPPED_SUBRESOURCE*);
	using UnmapFunc = void (*)(ID3D11DeviceContext*, ID3D11Resource*, UINT);

	/** @brief Snapshot shown next to the setting in the menu. */
	struct Status
	{
		bool supported = false;
		bool active = false;
		uint32_t managedBuffers = 0;
		uint32_t uploadsLastFrame = 0;
	};

	/**
	 * @brief stl::detour_vfunc that also reports whether calls through the slot now reach Hook::thunk.
	 *
	 * On success Detours leaves the vtable slot alone and points Hook::func at a trampoline; the
	 * fallback instead rewrites the slot (or a clone of the vtable). Neither happening means the
	 * slot is not hooked, and the ring must not rely on it.
	 */
	template <std::size_t Index, class Hook>
	bool DetourVFunc(void* a_object)
	{
		const auto original = (*static_cast<std::uintptr_t**>(a_object))[Index];
		stl::detour_vfunc<Index, Hook>(a_object);
		const auto current = (*static_cast<std::uintptr_t**>(a_object))[Index];
		return Hook::func.address() != original || current == reinterpret_cast<std::uintptr_t>(&Hook::thunk);
	}

	/**
	 * @brief Checks device support, creates the ring and hooks the constant buffer binding entry points.
	 *
	 * Call once on the render thread right after the Map/Unmap detours are live. The ring stays off
	 * (and every hook passes through) when the device lacks ConstantBufferOffsetting or
	 * MapNoOverwriteOnDynamicConstantBuffer, or when any hook fails to install.
	 * @param a_originalMap Trampoline to the original ID3D11DeviceContext::Map.
	 * @param a_originalUnmap Trampoline to the original ID3D11DeviceContext::Unmap.
	 */
	void Install(ID3D11Device* a_device, ID3D11DeviceContext* a_context, MapFunc a_originalMap, UnmapFunc a_originalUnmap);

	/** @brief Requests the ring on or off (live A/B toggle); applied at the next frame boundary. */
	void SetEnabled(bool a_enabled);

	/** @brief The requested state, as saved in the settings. */
	[[nodiscard]] bool IsEnabled();

	/** @brief Current support/activity for display. Call on the render thread. */
	[[nodiscard]] Status GetStatus();

	/** @brief Applies a pending toggle or a failure fallback. Call once per frame at Present. */
	void OnFrameBoundary();

	/**
	 * @brief Map interception, called first by the Map hook.
	 * @param a_caller Return address of the Map call, used to recognise calls from the game executable.
	 * @return True when the map was redirected to a shadow copy; the hook then returns S_OK.
	 */
	bool TryMap(ID3D11DeviceContext* a_context, ID3D11Resource* a_resource, UINT a_subresource, D3D11_MAP a_mapType,
		D3D11_MAPPED_SUBRESOURCE* a_mapped, const void* a_caller);

	/**
	 * @brief Unmap interception, called first by the Unmap hook.
	 * @return True when the unmap completed a redirected map; the original Unmap must then not run.
	 */
	bool TryUnmap(ID3D11DeviceContext* a_context, ID3D11Resource* a_resource, UINT a_subresource);
}
