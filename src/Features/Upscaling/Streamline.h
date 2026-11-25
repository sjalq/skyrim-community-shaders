#pragma once

#include "../../Buffer.h"
#include "../../State.h"

#include <cstdint>
#include <d3d11_4.h>
#include <d3d12.h>

#define NV_WINDOWS

#pragma warning(push)
#pragma warning(disable: 4471)
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_matrix_helpers.h>
#include <sl_reflex.h>
#include <sl_version.h>
#pragma warning(pop)

/** @brief Manages NVIDIA Streamline integration for DLSS upscaling and Reflex latency reduction. */
class Streamline
{
public:
	Streamline() = default;

	/** @brief Returns the short identifier used for logging. */
	inline std::string GetShortName() { return "Streamline"; }

	// Configure before calling LoadInterposer(). DX12 instance uses a separate plugin
	// directory and interposer DLL so the two SDK states are fully independent per-process.
	sl::RenderAPI renderAPI = sl::RenderAPI::eD3D11;
	std::wstring pluginDir = L"Data\\Shaders\\Upscaling\\Streamline";
	std::wstring interposerDllName = L"sl.interposer.dll";
	std::string instanceTag = "DX11";

	bool enabledAtBoot = false;
	bool initialized = false;
	bool triedInitialization = false;

	bool featureDLSS = false;
	bool featureDLSSG = false;
	bool featureReflex = false;
	bool featurePCL = false;
	bool reflexSupportedOnCurrentAdapter = false;

	sl::ViewportHandle viewport{ 0 };
	static constexpr uint32_t MAX_RESOLUTION = 8192;
	HMODULE interposer = NULL;

	// SL Interposer Functions
	PFun_slInit* slInit{};
	PFun_slShutdown* slShutdown{};
	PFun_slIsFeatureSupported* slIsFeatureSupported{};
	PFun_slIsFeatureLoaded* slIsFeatureLoaded{};
	PFun_slSetFeatureLoaded* slSetFeatureLoaded{};
	PFun_slEvaluateFeature* slEvaluateFeature{};
	PFun_slAllocateResources* slAllocateResources{};
	PFun_slFreeResources* slFreeResources{};
	PFun_slSetTag* slSetTag{};
	PFun_slSetTagForFrame* slSetTagForFrame{};
	PFun_slGetFeatureRequirements* slGetFeatureRequirements{};
	PFun_slGetFeatureVersion* slGetFeatureVersion{};
	PFun_slUpgradeInterface* slUpgradeInterface{};
	PFun_slSetConstants* slSetConstants{};
	PFun_slGetNativeInterface* slGetNativeInterface{};
	PFun_slGetFeatureFunction* slGetFeatureFunction{};
	PFun_slGetNewFrameToken* slGetNewFrameToken{};
	PFun_slSetD3DDevice* slSetD3DDevice{};

	// DLSS specific functions (DX11 instance)
	PFun_slDLSSGetOptimalSettings* slDLSSGetOptimalSettings{};
	PFun_slDLSSGetState* slDLSSGetState{};
	PFun_slDLSSSetOptions* slDLSSSetOptions{};

	// DLSS-G specific functions (DX12 instance)
	PFun_slDLSSGGetState* slDLSSGGetState{};
	PFun_slDLSSGSetOptions* slDLSSGSetOptions{};

	// Reflex/PCL functions (bound per instance)
	PFun_slReflexGetState* slReflexGetState{};
	PFun_slReflexSleep* slReflexSleep{};
	PFun_slReflexSetOptions* slReflexSetOptions{};
	PFun_slPCLSetMarker* slPCLSetMarker{};

	Util::FrameChecker frameChecker;
	sl::FrameToken* frameToken = nullptr;

	bool isRTXBelow40series = false;

	struct ReflexOptionsCache
	{
		bool valid = false;
		sl::ReflexMode mode = sl::ReflexMode::eOff;
		uint32_t frameLimitUs = 0;
		bool useMarkersToOptimize = false;
	};
	ReflexOptionsCache reflexOptionsCache{};
	uint32_t lastReflexSleepFrame = UINT32_MAX;

	// Per-instance camera history used to derive clipToPrevClip/prevClipToClip. The SL helper
	// recalculateCameraMatrices() keeps this in function statics that both Streamline instances
	// would share, so with DLSS (DX11) and DLSS-G (DX12) both active the second caller in a frame
	// would see zero camera motion.
	sl::float4x4 prevCameraViewToWorld{};
	sl::float4x4 prevCameraViewToClip{};
	bool hasPrevCameraMatrices = false;

	// DLSS-G runtime state (DX12 instance only)
	uint32_t dlssgFramesToGenerateMax = 0;     ///< Max generated frames per present reported by slDLSSGGetState (0 = not known yet)
	uint32_t dlssgFramesToGenerateActive = 0;  ///< Generated frames per present in the last applied options (0 = DLSS-G off)
	uint32_t dlssgFramesPresented = 0;         ///< Frames presented since the previous slDLSSGGetState call
	sl::DLSSGStatus dlssgStatus = sl::DLSSGStatus::eOk;
	bool dlssgVsyncSupportAvailable = false;
	bool dlssgEnabledLastFrame = false;

	/**
	 * @brief Executes DLSS evaluation for a single viewport with the given resources.
	 * @param vp The viewport handle identifying the DLSS instance.
	 * @param colorIn Input color texture to upscale.
	 * @param colorOut Output texture receiving the upscaled result.
	 * @param depth Depth buffer for temporal reprojection.
	 * @param mvec Per-pixel motion vectors.
	 * @param reactiveMask Reactive mask for temporal stability hints.
	 * @param transparencyMask Mask for transparency and composition handling.
	 * @param extentIn Input resolution extent.
	 * @param extentOut Output resolution extent.
	 * @param outputWidth Target output width for DLSS options.
	 */
	void EvaluateDLSS(sl::ViewportHandle vp,
		ID3D11Resource* colorIn, ID3D11Resource* colorOut, ID3D11Resource* depth,
		ID3D11Resource* mvec, ID3D11Resource* reactiveMask, ID3D11Resource* transparencyMask,
		const sl::Extent& extentIn, const sl::Extent& extentOut, uint32_t outputWidth);

	// Cached DLL version info for Streamline plugin directory
	static std::vector<std::pair<std::string, std::string>> dllVersions;

	/** @brief Loads the Streamline interposer DLL and initializes the SDK with feature preferences. */
	void LoadInterposer();

	/**
	 * @brief Queries available Streamline features (DLSS, Reflex, PCL) on the given adapter.
	 * @param a_adapter The DXGI adapter to check feature support against.
	 */
	void CheckFeatures(IDXGIAdapter* a_adapter);

	/**
	 * @brief Binds a D3D12 device to this Streamline instance (DX12 instance only).
	 *
	 * Must run before the frame-generation command queue and swap chain are created: binding the
	 * device initializes the SL plugins, which is when DLSS-G registers its swap-chain hooks.
	 * @param a_device The native D3D12 device used by the frame-generation swap chain.
	 * @return True if Streamline accepted the device.
	 */
	bool SetD3DDevice12(ID3D12Device* a_device);

	/** @brief Binds DLSS and Reflex feature functions after the D3D device is created. */
	void PostDevice();

	/**
	 * @brief Applies the DLSS-G options for the next present (DX12 instance only).
	 * @param enabled True to run DLSS frame generation for the next present.
	 * @param retainResources Keep DLSS-G resources allocated while it is temporarily off (menus).
	 * @param requestedFramesToGenerate Generated frames per rendered frame (1 = 2x, 2 = 3x, 3 = 4x);
	 *        clamped to the range the GPU reports via slDLSSGGetState.
	 */
	void ConfigureDLSSG(bool enabled, bool retainResources, uint32_t requestedFramesToGenerate);

	/**
	 * @brief Tags the DLSS-G inputs for a frame (DX12 instance only).
	 * @param token Frame token of the frame being presented.
	 * @param cmdList Command list that is executed on the presenting queue before the present.
	 * @param depth Shared depth buffer (render resolution data).
	 * @param mvec Shared motion-vector buffer (render resolution data).
	 * @param renderExtent Valid region of depth/mvec.
	 * @param hudLessColor Scene color without HUD, or nullptr when the UI is not separated this frame.
	 * @param uiColorAndAlpha Premultiplied UI color and alpha, or nullptr when the UI is not separated.
	 * @param backbufferExtent Full backbuffer extent (hudless/UI must match the backbuffer).
	 */
	void TagDLSSGResources(const sl::FrameToken& token, ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* depth, ID3D12Resource* mvec, const sl::Extent& renderExtent,
		ID3D12Resource* hudLessColor, ID3D12Resource* uiColorAndAlpha, const sl::Extent& backbufferExtent);

	/**
	 * @brief Reads the DLSS-G state after a present (DX12 instance only).
	 *
	 * Refreshes the supported multi-frame range and status, and makes the presenting queue wait
	 * on the DLSS-G input-consumption fence so the D3D11 side (which waits on that queue) cannot
	 * overwrite the shared inputs while DLSS-G still reads them.
	 * @param presentQueue Native presenting queue, or nullptr to skip the fence wait.
	 * @param waitForInputs True when DLSS-G was enabled for the present that just happened.
	 */
	void QueryDLSSGState(ID3D12CommandQueue* presentQueue, bool waitForInputs);

	/**
	 * @brief Returns the Streamline frame token for an explicit frame index.
	 * @param frameIndex Frame index shared by constants, tags and markers of one frame.
	 * @return The token, or nullptr if Streamline is unavailable.
	 */
	sl::FrameToken* GetFrameToken(uint32_t frameIndex);

	/**
	 * @brief Sends a PC Latency marker for an explicit frame index (no-op when PCL is unavailable).
	 * @param marker The PCL marker to send.
	 * @param frameIndex Frame index the marker belongs to.
	 */
	void SetPCLMarker(sl::PCLMarker marker, uint32_t frameIndex);

	/** @brief Acquires a new frame token from Streamline for the current frame. */
	bool EnsureFrameToken();
	/**
	 * @brief Sets camera and jitter constants on the Streamline viewport for the current frame.
	 * @param p_viewport The viewport handle to configure.
	 * @return True if constants were set successfully.
	 */
	bool CheckFrameConstants(sl::ViewportHandle p_viewport);

	/**
	 * @brief Sets camera and jitter constants for an explicit frame token.
	 * @param token Frame token the constants belong to.
	 * @param p_viewport The viewport handle to configure.
	 * @return True if constants were set successfully.
	 */
	bool SetFrameConstants(const sl::FrameToken& token, sl::ViewportHandle p_viewport);

	/**
	 * @brief Detects whether the GPU is an NVIDIA RTX card below the 40-series generation.
	 * @param a_adapter The DXGI adapter to inspect.
	 * @return True if the adapter is RTX 20xx or 30xx series.
	 */
	bool IsRTXAndBelow40Series(IDXGIAdapter* a_adapter);

	/**
	 * @brief Configures DLSS quality mode and resolution options for a viewport.
	 * @param p_viewport The viewport handle to configure.
	 * @param width The target output width.
	 */
	void SetDLSSOptions(sl::ViewportHandle p_viewport, uint32_t width);

	/**
	 * @brief Dispatches DLSS upscaling for the current frame.
	 * @param a_upscalingTexture The input color texture to upscale.
	 * @param a_reactiveMask Reactive mask for temporal stability hints.
	 * @param a_transparencyCompositionMask Mask for transparency handling.
	 * @param a_motionVectors Per-pixel motion vectors for temporal reprojection.
	 */
	void Upscale(ID3D11Resource* a_upscalingTexture, ID3D11Resource* a_reactiveMask, ID3D11Resource* a_transparencyCompositionMask, ID3D11Resource* a_motionVectors);
	/** @brief Updates Reflex latency reduction state and performs the Reflex sleep call. */
	void UpdateReflex();

	/** @brief Frees DLSS viewport resources through the Streamline SDK. */
	void DestroyDLSSResources();
};
