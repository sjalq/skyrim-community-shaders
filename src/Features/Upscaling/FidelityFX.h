#pragma once

#include <atomic>
#include <d3d12.h>
#include <winrt/base.h>

#include <FidelityFX/host/backends/dx11/ffx_dx11.h>
#include <FidelityFX/host/ffx_fsr3.h>
#include <FidelityFX/host/ffx_interface.h>

#include <FidelityFX/api/include/dx12/ffx_api_dx12.hpp>

#include <FidelityFX/api/include/ffx_api.hpp>
#include <FidelityFX/api/include/ffx_api_loader.h>
#include <FidelityFX/framegeneration/include/dx12/ffx_api_framegeneration_dx12.hpp>
#include <FidelityFX/framegeneration/include/ffx_framegeneration.hpp>

#include "../../Buffer.h"
#include "../../State.h"

/** @brief Manages AMD FidelityFX Super Resolution 3 upscaling and frame generation. */
class FidelityFX
{
public:
	static constexpr const wchar_t* PluginDir = L"Data\\Shaders\\Upscaling\\FidelityFX";

	HMODULE module = nullptr;

	ffx::Context swapChainContext{};
	ffx::Context frameGenContext;
	FfxFsr3Context fsrContext[2];

	bool featureFSR3FG = false;

	// Track if FidelityFX is currently being used for frame generation
	bool isFrameGenActive = false;

	// Track HDR state for frame generation callback (needs to be accessible from static callback)
	// Using atomic for thread safety since async workloads may read this from different threads
	static inline std::atomic<bool> isHDRActive = false;
	static inline std::atomic<float> hdrPeakNits = 1000.0f;
	static inline std::atomic<bool> needsReset = false;

	// Track previous HDR parameters to detect changes that require FG reset
	bool prevHDRActive = false;
	float prevPeakNits = 1000.0f;

	// Cached DLL version info for FidelityFX plugin directory
	static std::vector<std::pair<std::string, std::string>> dllVersions;

	/** @brief Loads the FidelityFX loader and frame generation DLLs from the plugin directory. */
	void LoadFFX();
	/** @brief Creates the FidelityFX frame generation context for the current swap chain. */
	void SetupFrameGeneration();
	/**
	 * @brief Presents the current frame, optionally performing FidelityFX frame generation.
	 * @param a_useFrameGeneration Whether to enable frame generation for this present call.
	 * @param a_isHDR Whether HDR output is active, affecting color space configuration.
	 * @param a_depth Depth input on the D3D12 device (shared texture or input-ring slot).
	 * @param a_motionVectors Motion-vector input on the D3D12 device.
	 * @param a_ui Premultiplied UI buffer registered for FidelityFX UI composition.
	 */
	void Present(bool a_useFrameGeneration, bool a_isHDR, ID3D12Resource* a_depth, ID3D12Resource* a_motionVectors, ID3D12Resource* a_ui);

	/**
	 * @brief Applies FfxApiSwapchainFramePacingTuning to the frame-generation swap chain when it changed.
	 *
	 * Hybrid spin and WaitForSingleObject on fences stay on (they stop the FG threads from
	 * busy-spinning); safety margin and variance factor come from the settings.
	 * @param a_safetyMarginMs Pacing safety margin in milliseconds (AMD default 0.1).
	 * @param a_varianceFactor Pacing variance factor in [0, 1] (AMD default 0.1).
	 */
	void ApplyFramePacingTuning(float a_safetyMarginMs, float a_varianceFactor);

	FfxApiSwapchainFramePacingTuning framePacingTuning{ 0.1f, 0.1f, true, 2, true };
	bool framePacingTuningApplied = false;

	/** @brief Creates FSR3 upscaling resources including scratch buffers and context. */
	void CreateFSRResources();

	/** @brief Destroys FSR3 upscaling resources and frees the scratch buffer. */
	void DestroyFSRResources();

	/**
	 * @brief Dispatches FSR3 upscaling for the current frame.
	 * @param a_upscalingTexture The input color texture to upscale.
	 * @param a_reactiveMask Reactive mask identifying areas needing temporal stability.
	 * @param a_transparencyCompositionMask Mask for transparency and composition handling.
	 * @param a_motionVectors Per-pixel motion vectors for temporal reprojection.
	 * @param a_sharpness RCAS sharpening strength applied after upscaling.
	 */
	void Upscale(ID3D11Resource* a_upscalingTexture, ID3D11Resource* a_reactiveMask, ID3D11Resource* a_transparencyCompositionMask, ID3D11Resource* a_motionVectors, float a_sharpness);

private:
	// FSR scratch buffer - needs to be freed in DestroyFSRResources
	void* fsrScratchBuffer = nullptr;

	// Flag to prevent spamming the log with FSR3 dispatch crash messages
	bool fsrDispatchCrashLogged = false;
};
