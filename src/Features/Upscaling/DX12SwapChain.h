#pragma once

#include <Windows.Foundation.h>
#include <stdio.h>
#include <winrt/base.h>
#include <wrl\client.h>
#include <wrl\wrappers\corewrappers.h>

#include <d3d11_4.h>
#include <d3d12.h>

#include <directx/d3dx12.h>

class WrappedResource
{
public:
	WrappedResource(D3D11_TEXTURE2D_DESC a_texDesc, ID3D11Device5* a_d3d11Device, ID3D12Device* a_d3d12Device);
	~WrappedResource();

	ID3D11Texture2D* resource11 = nullptr;
	ID3D11ShaderResourceView* srv = nullptr;
	ID3D11UnorderedAccessView* uav = nullptr;
	ID3D11RenderTargetView* rtv = nullptr;
	winrt::com_ptr<ID3D12Resource> resource;
};

struct DXGISwapChainProxy : IDXGISwapChain
{
public:
	DXGISwapChainProxy(IDXGISwapChain4* a_swapChain);

	IDXGISwapChain4* swapChain;

	/****IUnknown****/
	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObj) override;
	virtual ULONG STDMETHODCALLTYPE AddRef() override;
	virtual ULONG STDMETHODCALLTYPE Release() override;

	/****IDXGIObject****/
	virtual HRESULT STDMETHODCALLTYPE SetPrivateData(_In_ REFGUID Name, UINT DataSize, _In_reads_bytes_(DataSize) const void* pData) override;
	virtual HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(_In_ REFGUID Name, _In_opt_ const IUnknown* pUnknown) override;
	virtual HRESULT STDMETHODCALLTYPE GetPrivateData(_In_ REFGUID Name, _Inout_ UINT* pDataSize, _Out_writes_bytes_(*pDataSize) void* pData) override;
	virtual HRESULT STDMETHODCALLTYPE GetParent(_In_ REFIID riid, _COM_Outptr_ void** ppParent) override;

	/****IDXGIDeviceSubObject****/
	virtual HRESULT STDMETHODCALLTYPE GetDevice(_In_ REFIID riid, _COM_Outptr_ void** ppDevice) override;

	/****IDXGISwapChain****/
	virtual HRESULT STDMETHODCALLTYPE Present(UINT SyncInterval, UINT Flags);
	virtual HRESULT STDMETHODCALLTYPE GetBuffer(UINT Buffer, _In_ REFIID riid, _COM_Outptr_ void** ppSurface);
	virtual HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL Fullscreen, _In_opt_ IDXGIOutput* pTarget);
	virtual HRESULT STDMETHODCALLTYPE GetFullscreenState(_Out_opt_ BOOL* pFullscreen, _COM_Outptr_opt_result_maybenull_ IDXGIOutput** ppTarget);
	virtual HRESULT STDMETHODCALLTYPE GetDesc(_Out_ DXGI_SWAP_CHAIN_DESC* pDesc);
	virtual HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT NewFormat, UINT SwapChainFlags);
	virtual HRESULT STDMETHODCALLTYPE ResizeTarget(_In_ const DXGI_MODE_DESC* pNewTargetParameters);
	virtual HRESULT STDMETHODCALLTYPE GetContainingOutput(_COM_Outptr_ IDXGIOutput** ppOutput);
	virtual HRESULT STDMETHODCALLTYPE GetFrameStatistics(_Out_ DXGI_FRAME_STATISTICS* pStats);
	virtual HRESULT STDMETHODCALLTYPE GetLastPresentCount(_Out_ UINT* pLastPresentCount);
};

class DX12SwapChain
{
public:
	static constexpr UINT kLegacyBufferCount = 2;     ///< Proxy swap-chain buffers on the serialized hand-off
	static constexpr UINT kPipelinedBufferCount = 3;  ///< Proxy swap-chain buffers on the pipelined hand-off
	static constexpr UINT kMaxBufferCount = 3;        ///< Size of every array indexed by frameIndex
	static constexpr UINT kFrameRingSize = 3;         ///< Depth of the D3D12-private frame-input ring

	winrt::com_ptr<ID3D12Device> d3d12Device;
	winrt::com_ptr<ID3D12CommandQueue> commandQueue;
	winrt::com_ptr<ID3D12CommandAllocator> commandAllocators[kMaxBufferCount];
	winrt::com_ptr<ID3D12GraphicsCommandList4> commandLists[kMaxBufferCount];

	IDXGISwapChain4* swapChain;

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc;

	WrappedResource* swapChainBufferWrapped;
	WrappedResource* uiBufferWrapped;

	// D3D12 interop resources for frame generation
	WrappedResource* depthBufferShared12 = nullptr;
	WrappedResource* motionVectorBufferShared12 = nullptr;

	winrt::com_ptr<ID3D11Device5> d3d11Device;
	winrt::com_ptr<ID3D11DeviceContext4> d3d11Context;

	// D3D12 -> D3D11/CPU fence: only the D3D12 queue signals it (fenceValue), D3D11 and the CPU wait.
	winrt::com_ptr<ID3D11Fence> d3d11Fence;
	winrt::com_ptr<ID3D12Fence> d3d12Fence;

	// D3D11 -> D3D12 fence: D3D11 signals "frame inputs written" (inputReadyValue), D3D12 waits.
	// Separate from d3d12Fence so the two devices never signal the same fence out of order once
	// D3D11 no longer waits for the frame-generation present.
	winrt::com_ptr<ID3D11Fence> inputReadyFence11;
	winrt::com_ptr<ID3D12Fence> inputReadyFence12;
	UINT64 inputReadyValue = 0;

	winrt::com_ptr<ID3D12Resource> swapChainBuffers[kMaxBufferCount];

	UINT frameIndex = 0;
	UINT64 fenceValue = 0;

	UINT64 frameFenceValues[kMaxBufferCount] = {};

	// Swap-chain buffer count chosen at creation (Upscaling.frameGenerationPipelined at launch).
	UINT proxyBufferCount = kLegacyBufferCount;

	// Pipelined hand-off: each frame the D3D12 queue copies the D3D11-written inputs into a private
	// ring slot and signals D3D11 right after the copy; frame generation then reads only the slot.
	struct FrameRingSlot
	{
		winrt::com_ptr<ID3D12Resource> scene;
		winrt::com_ptr<ID3D12Resource> ui;
		winrt::com_ptr<ID3D12Resource> hudless;
		winrt::com_ptr<ID3D12Resource> depth;
		winrt::com_ptr<ID3D12Resource> motionVectors;
	};
	FrameRingSlot frameRing[kFrameRingSize];
	winrt::com_ptr<ID3D12CommandAllocator> ringCopyAllocators[kFrameRingSize];
	winrt::com_ptr<ID3D12GraphicsCommandList> ringCopyLists[kFrameRingSize];
	UINT64 ringCopyFenceValues[kFrameRingSize] = {};
	ID3D12Resource* frameRingSources[5] = {};  ///< Shared resources the ring was built from (rebuild on change)
	bool frameRingReady = false;
	bool frameRingFailed = false;
	UINT64 presentCounter = 0;
	int loggedHandOffMode = -1;  ///< -1 not logged yet, 0 serialized, 1 pipelined

	LARGE_INTEGER qpf;

	double refreshRate = 0;

	DXGISwapChainProxy* swapChainProxy = nullptr;

	// Whether DLSS-G is handling frame generation (vs FSR FG)
	bool useDLSSG = false;

	// DLSS-G only: the Streamline proxy of the presenting queue. The swap chain is created through
	// it so DLSS-G's queue/swap-chain hooks run; CS itself submits through the native commandQueue.
	winrt::com_ptr<ID3D12CommandQueue> commandQueueSL;

	// DLSS-G only: copy of the scene before the UI is composited on top (kBufferTypeHUDLessColor).
	WrappedResource* hudlessBufferWrapped = nullptr;

	// DLSS-G only: D3D11 compute pass that composites the premultiplied UI buffer onto the scene.
	winrt::com_ptr<ID3D11ComputeShader> composeUICS;
	bool composeUICSFailed = false;

	// Set by HDRDisplay::SetUIBuffer when this frame's UI went to uiBufferWrapped because frame
	// generation composites it; consumed (and cleared) by Present.
	bool uiRoutedForFGComposite = false;

	// Returns the current frame time (in seconds) for accurate FPS calculation when frame generation is active
	float GetFrameTime() const;

	void CreateD3D12Device(IDXGIAdapter* a_adapter);
	void CreateSwapChain(IDXGIAdapter* adapter, DXGI_SWAP_CHAIN_DESC swapChainDesc);

	/**
	 * @brief Creates the D3D12 device, queue and swap chain for DLSS-G through Streamline's manual-hooking proxies.
	 *
	 * Order follows the Streamline manual-hooking guide: native device -> slSetD3DDevice (initializes
	 * the plugins and registers DLSS-G's hooks) -> command queue via the proxy device -> swap chain via
	 * the proxy factory. On failure all partially created objects are dropped so the FSR path can start clean.
	 * @return True if the DLSS-G swap chain is ready.
	 */
	bool CreateSwapChainDLSSG(IDXGIAdapter* adapter, DXGI_SWAP_CHAIN_DESC swapChainDesc);

	/** @brief True when DLSS-G can receive the UI separately (compose shader and HUD-less buffer ready). */
	bool CanSeparateUIForDLSSG() const;

	/** @brief Compiles the DLSS-G UI composition shader (no-op if already compiled or previously failed). */
	void SetupDLSSGUIComposition();

	void CreateInterop();
	void RecreateWrappedResources(const DXGI_SWAP_CHAIN_DESC1& desc);

	DXGISwapChainProxy* GetSwapChainProxy();
	void SetD3D11Device(ID3D11Device* a_d3d11Device);
	void SetD3D11DeviceContext(ID3D11DeviceContext* a_d3d11Context);

	HRESULT GetBuffer(UINT buffer, REFIID riid, void** ppSurface);
	HRESULT ResizeBuffers(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT format, UINT flags);
	HRESULT Present(UINT SyncInterval, UINT Flags);
	HRESULT GetDevice(_In_ REFIID riid, _COM_Outptr_ void** ppDevice);
	HANDLE GetFrameLatencyWaitableObject();

	void SetColorSpace(bool enableHDR);

	// Resources needed by BackgroundBlur when D3D12 swap chain is active
	struct BlurResources
	{
		ID3D11Texture2D* backbufferTex = nullptr;
		ID3D11RenderTargetView* backbufferRTV = nullptr;
		ID3D11ShaderResourceView* backbufferSRV = nullptr;
		ID3D11ShaderResourceView* uiBufferSRV = nullptr;
		ID3D11RenderTargetView* uiBufferRTV = nullptr;
	};

	// Get all resources needed for background blur in one call
	BlurResources GetBlurResources() const;

	// D3D12 interop resource management
	void CreateSharedResources();

	/** @brief Swap-chain buffer count for a new proxy swap chain (3 when the hand-off is pipelined at launch). */
	static UINT GetLaunchBufferCount();

private:
	/**
	 * @brief Creates (or rebuilds after a resize) the D3D12-private frame-input ring.
	 * @return True if the ring matches the current shared resources and can be used this frame.
	 */
	bool EnsureFrameRing();

	/** @brief Drops the frame-input ring (callers make sure the GPU no longer uses it). */
	void ReleaseFrameRing();

	/** @brief Blocks the CPU until everything queued on the D3D12 queue has executed. */
	void WaitForGPUIdle();

	/**
	 * @brief DLSS-G: keeps a HUD-less copy of the scene and composites the UI onto the backbuffer on D3D11.
	 * @return True if the backbuffer now contains scene + UI and the HUD-less copy is valid.
	 */
	bool ComposeUIForDLSSG();
};
