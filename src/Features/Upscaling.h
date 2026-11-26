#pragma once

#include "Feature.h"
#include "Upscaling/DX12SwapChain.h"
#include "Upscaling/FidelityFX.h"
#include "Upscaling/RCAS/RCAS.h"
#include "Upscaling/Streamline.h"
#include <d3d11_4.h>
#include <d3d12.h>
#include <winrt/base.h>

/**
 * @brief Provides upscaling functionality including DLSS, FSR and TAA.
 *
 * This feature handles various upscaling methods and frame generation technologies
 * to improve performance while maintaining visual quality.
 */
struct Upscaling : Feature
{
private:
	static constexpr std::string_view MOD_ID = "156952";

public:
	// Feature interface
	virtual inline std::string GetName() override { return "Upscaling"; }
	virtual std::string GetDisplayName() override { return T("feature.upscaling.name", "Upscaling"); }
	virtual inline std::string GetShortName() override { return "Upscaling"; }
	virtual inline std::string GetFeatureModLink() override { return MakeNexusModURL(MOD_ID); }
	virtual inline bool IsCore() const override { return false; }
	virtual inline std::string_view GetCategory() const override { return FeatureCategories::kDisplay; }

	virtual std::pair<std::string, std::vector<std::string>> GetFeatureSummary() override
	{
		return { T("feature.upscaling.description", "Advanced upscaling and frame generation technologies for improved performance"),
			{ T("feature.upscaling.key_feature_1", "DLSS (Deep Learning Super Sampling) support"),
				T("feature.upscaling.key_feature_2", "FSR (FidelityFX Super Resolution) support"),
				T("feature.upscaling.key_feature_3", "TAA (Temporal Anti-Aliasing) support"),
				T("feature.upscaling.key_feature_4", "Frame generation for supported systems") } };
	};

	float2 jitter = { 0, 0 };

	enum class FrameGenMethod
	{
		kNone,
		kFSR,
		kDLSSG
	};

	enum class UpscaleMethod
	{
		kNONE,
		kTAA,
		kFSR,
		kDLSS
	};

	struct Settings
	{
		uint upscaleMethod = (uint)UpscaleMethod::kDLSS;
		uint upscaleMethodNoDLSS = (uint)UpscaleMethod::kFSR;
		uint qualityMode = 1;  // Default to Quality (1=Quality, 2=Balanced, 3=Performance, 4=Ultra Performance, 0=Native AA)
		uint frameLimitMode = 1;
		uint frameGenerationMode = 1;
		uint frameGenerationForceEnable = 0;
		bool frameGenerationAllowInMenus = false;
		uint streamlineLogLevel = 0;  // 0=Off, 1=Default, 2=Verbose
		float sharpnessFSR = 0.0f;
		bool sharpnessEnabledDLSS = false;
		float sharpnessDLSS = 0.0f;
		uint presetDLSS = 0;  // 0=Default, 1=J, 2=K, 3=L, 4=M
		bool reflexLowLatencyMode = false;
		bool reflexLowLatencyBoost = false;
		bool reflexUseMarkersToOptimize = false;
		bool reflexUseFPSLimit = false;
		float reflexFPSLimit = 60.0f;
		// NVIDIA DLSS Frame Generation (DX12 Streamline). Generated frames per rendered frame:
		// 1 = 2x, 2 = 3x, 3 = 4x, 4 = 5x, 5 = 6x (multi-frame generation, RTX 50 series). Clamped at
		// runtime to what slDLSSGGetState reports as supported (numFramesToGenerateMax).
		uint dlssgFramesToGenerate = 3;
		// Use DLSS-G instead of FSR frame generation when the GPU supports it (takes effect at restart).
		bool frameGenerationPreferDLSSG = true;
		// DLSS 4.5 Dynamic Multi Frame Generation (DLSSGMode::eDynamic): the multiplier adapts so the
		// presented rate tracks the target. Used only while DLSSGState::bIsDynamicMFGSupported is true;
		// dlssgFramesToGenerate is ignored while it runs. Applied live.
		bool dlssgDynamic = true;
		// Dynamic MFG target in presented frames per second; 0 = the display refresh rate (SL auto-detect).
		float dlssgDynamicTargetFps = 0.0f;
		// Presented-FPS lock (overrides Dynamic MFG while on, applied live, works with VSync on): CS picks
		// the fixed multiplier N itself, the smallest whose output reaches dlssgLockMinFps at the engine's
		// measured capability, and caps rendered frames at dlssgLockFps / N, so presented frames stay in
		// [dlssgLockMinFps, dlssgLockFps]. 0 = auto: lock = refresh - refresh^2 / 3600 (NVIDIA's G-SYNC +
		// Reflex cap, 157.4 at 165 Hz), minimum = 93% of the lock.
		bool dlssgLock = false;
		float dlssgLockFps = 0.0f;
		float dlssgLockMinFps = 0.0f;
		// DLSS-G User Interface Recomposition: interpolate the HUD-less scene and the UI separately.
		bool dlssgUIRecomposition = false;
		// Reflex mode for the DX12 (DLSS-G) Streamline instance: -1 = default (forced on while frame
		// generation runs, otherwise the Low Latency settings), 0 = off, 1 = on, 2 = on + boost. Applied live.
		int frameGenerationReflexMode = -1;
		// Pipelined frame-generation hand-off: the D3D12 side copies the D3D11 inputs into a 3-deep ring
		// and releases D3D11 right away instead of after the frame-generation present, and the proxy swap
		// chain uses 3 buffers (buffer count needs a restart; the hand-off itself switches live).
		bool frameGenerationPipelined = true;
		// FidelityFX frame-generation pacing tuning (FfxApiSwapchainFramePacingTuning), applied live.
		// AMD defaults are 0.1 ms / 0.1; AMD documents 0.01 ms / 0.3 ("tuning set B") for FG that gets
		// stuck at a low frame rate after heavy scenes.
		float fsrFGPacingSafetyMarginMs = 0.1f;
		float fsrFGPacingVarianceFactor = 0.1f;
	};

	static constexpr uint kDLSSGMaxFramesToGenerate = 5;  ///< 6x is the highest multiplier DLSS-G offers

	Settings settings;

	struct JitterCB
	{
		float2 jitter;
		float useWideKernel;
		float pad0;
	};

	struct UpscalingDataCB
	{
		float2 trueSamplingDim;
		float2 pad0;
	};

	ConstantBuffer* jitterCB = nullptr;
	ConstantBuffer* upscalingDataCB = nullptr;

	// Runtime state
	bool isWindowed = false;
	bool lowRefreshRate = false;
	bool fidelityFXMissing = false;
	bool d3d12SwapChainActive = false;

	// Timing and scaling
	double refreshRate = 0.0f;
	float2 resolutionScale = { 1.0f, 1.0f };
	LARGE_INTEGER qpf;

	// Presented-FPS lock runtime state (settings.dlssgLock)
	struct FrameGenLock
	{
		static constexpr uint32_t kWindow = 32;         ///< rendered frames in the work-time window
		static constexpr uint32_t kUpFrames = 3;        ///< frames a higher multiplier must be needed before switching up
		static constexpr uint32_t kMinDownHold = 60;    ///< frames a lower multiplier must be sustainable before switching down
		static constexpr uint32_t kMaxDownHold = 1920;  ///< hold after repeated down/up flapping
		std::array<float, kWindow> workMs{};
		uint32_t head = 0;
		uint32_t samples = 0;
		uint32_t multiplier = 0;     ///< presented frames per rendered frame requested from DLSS-G (0 = lock idle)
		uint32_t maxMultiplier = 0;  ///< highest multiplier the GPU reported when the lock (re)started
		uint32_t upFrames = 0;
		uint32_t downFrames = 0;
		uint32_t downHold = kMinDownHold;
		uint32_t framesSinceDown = 1u << 30;
		uint32_t framesSinceUp = 1u << 30;
		int64_t lastExitQPC = 0;
		float capableFps = 0.0f;  ///< engine rate at the 90th percentile frame work time
		float capFps = 0.0f;      ///< rendered-frame cap = lock / multiplier
	} fgLock;

	// FG FPS Measurement for Overlay
	bool IsFrameGenerationDx12PathActive() const;
	bool IsFrameGenerationActive() const;
	bool ShouldUseFrameGenerationThisFrame() const;
	float GetFrameGenerationFrameTime() const;
	bool IsUpscalingActive() const;

	// Feature interface overrides
	virtual void DrawSettings() override;
	virtual void SaveSettings(json& o_json) override;
	virtual void LoadSettings(json& o_json) override;
	virtual void RestoreDefaultSettings() override;
	virtual void DataLoaded() override;

	/**
	 * @brief Installs Direct3D-related hooks for device and factory creation.
	 *
	 * Loads FidelityFX support and patches the import address table (IAT) to redirect D3D11 device and DXGI factory creation functions to custom hook implementations.
	**/
	virtual void Load() override;
	virtual void PostPostLoad() override;
	virtual void SetupResources() override;

	UpscaleMethod GetUpscaleMethod() const;
	FrameGenMethod GetFrameGenMethod() const;
	bool UsesDLSSGFrameGen() const;

	/**
	 * @brief Number of displayed frames per rendered frame while frame generation runs.
	 * @return 1 without frame generation, 2 for FSR, numFramesToGenerate + 1 for DLSS-G.
	 */
	uint32_t GetFrameGenerationMultiplier() const;

	void CheckResources(UpscaleMethod a_upscalemethod);
	void CreateUpscalingTextureResources(UpscaleMethod a_upscalemethod);
	void DestroyUpscalingTextureResources(UpscaleMethod a_upscalemethod);

	winrt::com_ptr<ID3D11ComputeShader> encodeTexturesCS[4];  // One for each UpscaleMethod (kNONE, kTAA, kFSR, kDLSS)
	ID3D11ComputeShader* GetEncodeTexturesCS();

	winrt::com_ptr<ID3D11PixelShader> depthRefractionUpscalePS;
	ID3D11PixelShader* GetDepthRefractionUpscalePS();

	winrt::com_ptr<ID3D11PixelShader> underwaterMaskUpscalePS;
	ID3D11PixelShader* GetUnderwaterMaskUpscalePS();

	winrt::com_ptr<ID3D11VertexShader> upscaleVS;
	ID3D11VertexShader* GetUpscaleVS();

	winrt::com_ptr<ID3D11DepthStencilState> upscaleDepthStencilState;
	winrt::com_ptr<ID3D11BlendState> upscaleBlendState;
	winrt::com_ptr<ID3D11RasterizerState> upscaleRasterizerState;

	// Helper: Create a Texture2D matching source format at a given size
	static eastl::unique_ptr<Texture2D> CreateTextureFromSource(ID3D11Resource* src, uint32_t width, uint32_t height,
		bool copyBindFlags = false, bool createSRV = false, bool createUAV = false, const char* name = nullptr);

	void ConfigureTAA();
	void ConfigureUpscaling(RE::BSGraphics::State* a_state);
	void Upscale();

	// D3D11 textures
	Texture2D* reactiveMaskTexture = nullptr;
	Texture2D* transparencyCompositionMaskTexture = nullptr;
	Texture2D* motionVectorCopyTexture = nullptr;
	Texture2D* sharpenerTexture = nullptr;

	virtual void ClearShaderCache() override;

	// Static instances instead of singletons
	static inline Streamline streamline;      ///< DX11 instance: DLSS + Reflex + PCL
	static inline Streamline streamlineDX12;  ///< DX12 instance: DLSS-G frame generation
	static inline FidelityFX fidelityFX;      ///< AMD FSR frame generation
	static inline DX12SwapChain dx12SwapChain;
	static inline RCAS rcas;  ///< Standalone RCAS sharpening for DLSS

	winrt::com_ptr<ID3D11PixelShader> copyDepthToSharedBufferPS;

	float projectionPosScaleX = 0.0f;
	float projectionPosScaleY = 0.0f;

	float dynamicResolutionWidthRatio = 1.0f;
	float dynamicResolutionHeightRatio = 1.0f;

	bool previousUpscalingWasActive = false;
	bool depthUpscaleUseWideKernel = false;

	/**
	 * Set by MenuOpenCloseEventHandler when LoadingMenu closes (cell/worldspace transitions,
	 * initial load). Consumed at the start of Upscale() to force a one-frame DLSS feature
	 * rebuild.
	 */
	std::atomic<bool> pendingDLSSReset{ false };

	void CopySharedD3D12Resources();
	void PostDisplay();
	void PerformUpscaling();
	void UpscaleDepth();

	/**
	 * @brief Applies RCAS sharpening to the main render target after DLSS upscaling.
	 *
	 * Runs in HDR space before tonemapping. Only called when DLSS is active and sharpness > 0.
	 */
	void ApplySharpening();

	static void TimerSleepQPC(int64_t targetQPC);

	void FrameLimiter();

	// Presented-FPS lock (settings.dlssgLock), see Upscaling.cpp
	bool FrameGenLockActive() const;
	uint32_t FrameGenLockMultiplier();
	void FrameGenLockLimiter(bool generatingThisFrame);
	void FrameGenLockReset();
	void SleepUntilQPC(int64_t targetQPC) const;

	static double GetRefreshRate(HWND a_window);

	// Unified interface methods - external code should use these instead of direct access
	void LoadUpscalingSDKs();  // Loads all SDKs at once
	HANDLE GetFrameLatencyWaitableObject() const;
	float GetFrameTime() const;

	// Backend interface methods
	bool IsBackendInitialized() const;
	void CheckBackendFeatures(IDXGIAdapter* adapter);
	void UpgradeBackendInterface(void** ppInterface);
	void SetBackendD3DDevice(ID3D11Device* device);
	void PostBackendDevice();

	// Module availability methods
	bool HasFrameGenModule() const;

	// Proxy interface methods
	void SetProxyD3D11Device(ID3D11Device* device);
	void SetProxyD3D11DeviceContext(ID3D11DeviceContext* context);
	void CreateProxySwapChain(IDXGIAdapter* adapter, DXGI_SWAP_CHAIN_DESC swapChainDesc);
	bool CreateProxySwapChainDLSSG(IDXGIAdapter* adapter, DXGI_SWAP_CHAIN_DESC swapChainDesc);
	void CreateProxyInterop();
	IDXGISwapChain* GetProxySwapChain();

	using BlurResources = DX12SwapChain::BlurResources;

	// Get all D3D11 resources needed for background blur when D3D12 swap chain is active
	BlurResources GetBlurResources() const;

private:
	struct Main_UpdateJitter
	{
		static void thunk(RE::BSGraphics::State* a_state);
		static inline REL::Relocation<decltype(thunk)> func;
	};

	struct MenuManagerDrawInterfaceStartHook
	{
		static void thunk(int64_t a1);
		static inline REL::Relocation<decltype(thunk)> func;
	};

	struct Main_PostProcessing
	{
		static void thunk(RE::ImageSpaceManager* a_this, uint32_t a3, RE::RENDER_TARGET a_target, void* a_4, bool a_5);
		static inline REL::Relocation<decltype(thunk)> func;
	};

	struct Main_RenderPrecipitation
	{
		static void thunk();
		static inline REL::Relocation<decltype(thunk)> func;
	};

	struct BSFaceGenManager_UpdatePendingCustomizationTextures
	{
		static void thunk();
		static inline REL::Relocation<decltype(thunk)> func;
	};

	class MenuOpenCloseEventHandler : public RE::BSTEventSink<RE::MenuOpenCloseEvent>
	{
	public:
		virtual RE::BSEventNotifyControl ProcessEvent(const RE::MenuOpenCloseEvent* a_event, RE::BSTEventSource<RE::MenuOpenCloseEvent>*) override;
		static bool Register();
	};
};
