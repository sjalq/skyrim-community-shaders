#include "DX12SwapChain.h"

#include <FidelityFX/api/include/dx12/ffx_api_dx12.hpp>
#include <algorithm>
#include <dxgi1_6.h>
#include <format>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "../HDRDisplay.h"
#include "../Upscaling.h"
#include "FidelityFX.h"
#include "Streamline.h"

void DX12SwapChain::CreateD3D12Device(IDXGIAdapter* a_adapter)
{
	DX::ThrowIfFailed(D3D12CreateDevice(a_adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d12Device)));

	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
	queueDesc.NodeMask = 0;

	DX::ThrowIfFailed(d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

	for (UINT i = 0; i < kMaxBufferCount; i++) {
		DX::ThrowIfFailed(d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocators[i])));
		DX::ThrowIfFailed(d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[i].get(), nullptr, IID_PPV_ARGS(&commandLists[i])));
		commandLists[i]->Close();
	}
}

UINT DX12SwapChain::GetLaunchBufferCount()
{
	return globals::features::upscaling.settings.frameGenerationPipelined ? kPipelinedBufferCount : kLegacyBufferCount;
}

void DX12SwapChain::CreateSwapChain(IDXGIAdapter* adapter, DXGI_SWAP_CHAIN_DESC a_swapChainDesc)
{
	CreateD3D12Device(adapter);

	IDXGIFactory4* dxgiFactory;
	DX::ThrowIfFailed(adapter->GetParent(IID_PPV_ARGS(&dxgiFactory)));

	// Runtime format negotiation for swap chain
	DXGI_FORMAT attemptedFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
	DXGI_FORMAT negotiatedFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
	bool fallbackUsed = false;

	// Test R10G10B10A2 support for HDR capability
	D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport = { DXGI_FORMAT_R10G10B10A2_UNORM, D3D12_FORMAT_SUPPORT1_RENDER_TARGET, D3D12_FORMAT_SUPPORT2_NONE };
	if (SUCCEEDED(d3d12Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &formatSupport, sizeof(formatSupport)))) {
		if ((formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) == 0) {
			logger::warn("[DX12SwapChain] R10G10B10A2_UNORM not supported as render target, falling back to R8G8B8A8_UNORM");
			negotiatedFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
			fallbackUsed = true;
		}
	} else {
		logger::warn("[DX12SwapChain] CheckFeatureSupport failed for R10G10B10A2_UNORM, falling back to R8G8B8A8_UNORM");
		negotiatedFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
		fallbackUsed = true;
	}

	logger::info("[DX12SwapChain] Swap chain format negotiation: attempted={}, negotiated={}, fallback={}",
		static_cast<uint32_t>(attemptedFormat),
		static_cast<uint32_t>(negotiatedFormat),
		fallbackUsed ? "true" : "false");

	swapChainDesc = {};
	swapChainDesc.Width = a_swapChainDesc.BufferDesc.Width;
	swapChainDesc.Height = a_swapChainDesc.BufferDesc.Height;
	swapChainDesc.Format = negotiatedFormat;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	proxyBufferCount = GetLaunchBufferCount();
	swapChainDesc.BufferCount = proxyBufferCount;
	swapChainDesc.SwapEffect = a_swapChainDesc.SwapEffect;
	swapChainDesc.Flags = a_swapChainDesc.Flags;

	ffx::CreateContextDescFrameGenerationSwapChainForHwndDX12 ffxSwapChainDesc{};

	ffxSwapChainDesc.desc = &swapChainDesc;
	ffxSwapChainDesc.dxgiFactory = dxgiFactory;
	ffxSwapChainDesc.fullscreenDesc = nullptr;
	ffxSwapChainDesc.gameQueue = commandQueue.get();
	ffxSwapChainDesc.hwnd = a_swapChainDesc.OutputWindow;
	ffxSwapChainDesc.swapchain = &swapChain;

	auto& fidelityFX = globals::features::upscaling.fidelityFX;

	if (ffx::CreateContext(fidelityFX.swapChainContext, nullptr, ffxSwapChainDesc) != ffx::ReturnCode::Ok) {
		logger::critical("[FidelityFX] Failed to create swap chain context!");
	} else {
		// Hybrid spin + WaitForSingleObject on fences so the FG present/pacing threads stop
		// busy-spinning; safety margin and variance come from the settings (live-adjustable).
		const auto& settings = globals::features::upscaling.settings;
		fidelityFX.ApplyFramePacingTuning(settings.fsrFGPacingSafetyMarginMs, settings.fsrFGPacingVarianceFactor);
	}

	for (UINT i = 0; i < proxyBufferCount; i++)
		DX::ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(&swapChainBuffers[i])));

	frameIndex = swapChain->GetCurrentBackBufferIndex();

	// Set color space based on HDR Display feature state and negotiated format
	auto* hdr = globals::features::hdrDisplay.loaded ? &globals::features::hdrDisplay : nullptr;
	bool enableHDR = hdr && hdr->settings.enableHDR;
	// Only set HDR color space if not falling back to SDR format
	SetColorSpace(enableHDR && !fallbackUsed);

	fidelityFX.SetupFrameGeneration();
}

bool DX12SwapChain::CreateSwapChainDLSSG(IDXGIAdapter* adapter, DXGI_SWAP_CHAIN_DESC a_swapChainDesc)
{
	auto& slDX12 = globals::features::upscaling.streamlineDX12;

	const auto resetPartialState = [&]() {
		if (swapChain) {
			swapChain->Release();
			swapChain = nullptr;
		}
		for (UINT i = 0; i < kMaxBufferCount; i++) {
			swapChainBuffers[i] = nullptr;
			commandLists[i] = nullptr;
			commandAllocators[i] = nullptr;
		}
		commandQueue = nullptr;
		commandQueueSL = nullptr;
		// Streamline was already bound to this device; keep it alive so the dormant DX12 SL
		// instance never holds a dangling device pointer.
		static winrt::com_ptr<ID3D12Device> retiredDevice;
		if (d3d12Device)
			retiredDevice = d3d12Device;
		d3d12Device = nullptr;
		useDLSSG = false;
	};

	try {
		if (!slDX12.initialized || !slDX12.slUpgradeInterface || !slDX12.slGetNativeInterface)
			throw std::runtime_error("Streamline DX12 interposer is not initialized");

		// 1. Native device.
		DX::ThrowIfFailed(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(d3d12Device.put())));

		// 2. Bind it to Streamline. This initializes the plugins and registers DLSS-G's hooks, which
		//    must exist before the queue and swap chain are created.
		if (!slDX12.SetD3DDevice12(d3d12Device.get()))
			throw std::runtime_error("slSetD3DDevice failed");
		slDX12.PostDevice();
		if (!slDX12.featureDLSSG)
			throw std::runtime_error("DLSS-G entry points are unavailable after device bind");

		// 3. Command queue through the proxy device (DLSS-G hooks ID3D12Device::CreateCommandQueue).
		// The proxy (which holds its own reference to the native device) is intentionally never
		// released: the proxy queue created below keeps pointing at it for the process lifetime.
		ID3D12Device* proxyDevice = d3d12Device.get();
		if (SL_FAILED(upgradeResult, slDX12.slUpgradeInterface(reinterpret_cast<void**>(&proxyDevice))))
			throw std::runtime_error(std::format("slUpgradeInterface(ID3D12Device) failed: {}", magic_enum::enum_name(upgradeResult)));

		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
		queueDesc.NodeMask = 0;
		DX::ThrowIfFailed(proxyDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(commandQueueSL.put())));

		// CS submits its own work through the native queue (Streamline guide: native interfaces
		// everywhere except the hooked entry points).
		void* nativeQueue = nullptr;
		if (SL_FAILED(nativeResult, slDX12.slGetNativeInterface(commandQueueSL.get(), &nativeQueue)) || !nativeQueue)
			throw std::runtime_error(std::format("slGetNativeInterface(ID3D12CommandQueue) failed: {}", magic_enum::enum_name(nativeResult)));
		commandQueue.attach(static_cast<ID3D12CommandQueue*>(nativeQueue));

		for (UINT i = 0; i < kMaxBufferCount; i++) {
			DX::ThrowIfFailed(d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocators[i].put())));
			DX::ThrowIfFailed(d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[i].get(), nullptr, IID_PPV_ARGS(commandLists[i].put())));
			commandLists[i]->Close();
		}

		// 4. Swap chain through the proxy factory (DLSS-G hooks CreateSwapChainForHwnd, GetBuffer,
		//    GetCurrentBackBufferIndex, Present and ResizeBuffers on it).
		winrt::com_ptr<IDXGIFactory4> nativeFactory;
		DX::ThrowIfFailed(adapter->GetParent(IID_PPV_ARGS(nativeFactory.put())));
		IDXGIFactory* proxyFactory = nativeFactory.get();
		if (SL_FAILED(factoryResult, slDX12.slUpgradeInterface(reinterpret_cast<void**>(&proxyFactory))))
			throw std::runtime_error(std::format("slUpgradeInterface(IDXGIFactory) failed: {}", magic_enum::enum_name(factoryResult)));
		winrt::com_ptr<IDXGIFactory2> proxyFactory2;
		const HRESULT factoryQI = proxyFactory->QueryInterface(IID_PPV_ARGS(proxyFactory2.put()));
		proxyFactory->Release();  // proxyFactory2 holds the reference that matters
		DX::ThrowIfFailed(factoryQI);

		DXGI_FORMAT attemptedFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
		DXGI_FORMAT negotiatedFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
		bool fallbackUsed = false;

		D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport = { DXGI_FORMAT_R10G10B10A2_UNORM, D3D12_FORMAT_SUPPORT1_RENDER_TARGET, D3D12_FORMAT_SUPPORT2_NONE };
		if (SUCCEEDED(d3d12Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &formatSupport, sizeof(formatSupport)))) {
			if ((formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) == 0) {
				logger::warn("[DX12SwapChain] R10G10B10A2_UNORM not supported as render target, falling back to R8G8B8A8_UNORM");
				negotiatedFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
				fallbackUsed = true;
			}
		} else {
			logger::warn("[DX12SwapChain] CheckFeatureSupport failed for R10G10B10A2_UNORM, falling back to R8G8B8A8_UNORM");
			negotiatedFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
			fallbackUsed = true;
		}

		logger::info("[DX12SwapChain] DLSS-G swap chain format negotiation: attempted={}, negotiated={}, fallback={}",
			static_cast<uint32_t>(attemptedFormat),
			static_cast<uint32_t>(negotiatedFormat),
			fallbackUsed ? "true" : "false");

		swapChainDesc = {};
		swapChainDesc.Width = a_swapChainDesc.BufferDesc.Width;
		swapChainDesc.Height = a_swapChainDesc.BufferDesc.Height;
		swapChainDesc.Format = negotiatedFormat;
		swapChainDesc.SampleDesc.Count = 1;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		proxyBufferCount = GetLaunchBufferCount();
		swapChainDesc.BufferCount = proxyBufferCount;
		swapChainDesc.SwapEffect = a_swapChainDesc.SwapEffect;
		// No frame-latency waitable object: DLSS-G presents from its own pacing thread and Reflex
		// handles latency; waiting on the native swap chain's object would fight that pacer.
		swapChainDesc.Flags = a_swapChainDesc.Flags & ~static_cast<UINT>(DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);

		winrt::com_ptr<IDXGISwapChain1> swapChain1;
		DX::ThrowIfFailed(proxyFactory2->CreateSwapChainForHwnd(
			commandQueueSL.get(),
			a_swapChainDesc.OutputWindow,
			&swapChainDesc,
			nullptr,
			nullptr,
			swapChain1.put()));

		DX::ThrowIfFailed(swapChain1->QueryInterface(IID_PPV_ARGS(&swapChain)));

		for (UINT i = 0; i < proxyBufferCount; i++)
			DX::ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(swapChainBuffers[i].put())));

		frameIndex = swapChain->GetCurrentBackBufferIndex();

		auto* hdr = globals::features::hdrDisplay.loaded ? &globals::features::hdrDisplay : nullptr;
		bool enableHDR = hdr && hdr->settings.enableHDR;
		SetColorSpace(enableHDR && !fallbackUsed);

		useDLSSG = true;

		// Query the supported multi-frame range up front so the first frames already use it.
		slDX12.QueryDLSSGState(nullptr, false);

		logger::info("[DX12SwapChain] Created DLSS-G swap chain through Streamline proxies ({}x{}, format {})",
			swapChainDesc.Width, swapChainDesc.Height, static_cast<uint32_t>(swapChainDesc.Format));
		return true;
	} catch (const std::exception& e) {
		logger::error("[DX12SwapChain] DLSS-G swap chain creation failed: {}. Falling back to FSR frame generation if available.", e.what());
		resetPartialState();
		slDX12.featureDLSSG = false;
		return false;
	}
}

bool DX12SwapChain::CanSeparateUIForDLSSG() const
{
	return useDLSSG && composeUICS && hudlessBufferWrapped && hudlessBufferWrapped->srv &&
	       uiBufferWrapped && uiBufferWrapped->srv && swapChainBufferWrapped && swapChainBufferWrapped->uav;
}

void DX12SwapChain::SetupDLSSGUIComposition()
{
	if (!useDLSSG || composeUICS || composeUICSFailed)
		return;

	std::vector<std::pair<const char*, const char*>> defines;
	composeUICS.attach(static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\Upscaling\\ComposeUICS.hlsl", defines, "cs_5_0")));
	if (composeUICS) {
		logger::info("[DX12SwapChain] DLSS-G UI composition ready (HUD-less + UI color/alpha inputs enabled)");
	} else {
		composeUICSFailed = true;
		logger::warn("[DX12SwapChain] ComposeUICS.hlsl failed to compile; DLSS-G will run without HUD-less/UI inputs");
	}
}

bool DX12SwapChain::ComposeUIForDLSSG()
{
	if (!CanSeparateUIForDLSSG())
		return false;

	auto context = d3d11Context.get();
	auto state = globals::state;

	state->BeginPerfEvent("DLSS-G UI Composite");

	// The UI buffer is usually still bound as the render target here (kFRAMEBUFFER redirect), and
	// D3D11 nulls an SRV whose resource is bound for output, so unbind OM first and restore after.
	ID3D11RenderTargetView* savedRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
	ID3D11DepthStencilView* savedDSV = nullptr;
	context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, savedRTVs, &savedDSV);
	context->OMSetRenderTargets(0, nullptr, nullptr);

	// HUD-less copy of the scene, then backbuffer = UI + (1 - UI.a) * scene (premultiplied UI,
	// the same blend FidelityFX applies with FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA).
	context->CopyResource(hudlessBufferWrapped->resource11, swapChainBufferWrapped->resource11);

	ID3D11ShaderResourceView* views[2] = { hudlessBufferWrapped->srv, uiBufferWrapped->srv };
	context->CSSetShaderResources(0, ARRAYSIZE(views), views);
	ID3D11UnorderedAccessView* uavs[1] = { swapChainBufferWrapped->uav };
	context->CSSetUnorderedAccessViews(0, ARRAYSIZE(uavs), uavs, nullptr);
	context->CSSetShader(composeUICS.get(), nullptr, 0);

	globals::profiler->BeginPass("DX12SwapChain::DLSSGUIComposite");
	context->Dispatch((swapChainDesc.Width + 7) / 8, (swapChainDesc.Height + 7) / 8, 1);
	globals::profiler->EndPass();

	views[0] = nullptr;
	views[1] = nullptr;
	context->CSSetShaderResources(0, ARRAYSIZE(views), views);
	uavs[0] = nullptr;
	context->CSSetUnorderedAccessViews(0, ARRAYSIZE(uavs), uavs, nullptr);
	context->CSSetShader(nullptr, nullptr, 0);

	context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, savedRTVs, savedDSV);
	for (auto*& rtv : savedRTVs) {
		if (rtv) {
			rtv->Release();
			rtv = nullptr;
		}
	}
	if (savedDSV)
		savedDSV->Release();

	state->EndPerfEvent();
	return true;
}

void DX12SwapChain::CreateInterop()
{
	HANDLE sharedFenceHandle;
	DX::ThrowIfFailed(d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3d12Fence)));
	DX::ThrowIfFailed(d3d12Device->CreateSharedHandle(d3d12Fence.get(), nullptr, GENERIC_ALL, nullptr, &sharedFenceHandle));
	DX::ThrowIfFailed(d3d11Device->OpenSharedFence(sharedFenceHandle, IID_PPV_ARGS(&d3d11Fence)));
	CloseHandle(sharedFenceHandle);

	HANDLE inputReadyFenceHandle;
	DX::ThrowIfFailed(d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(inputReadyFence12.put())));
	DX::ThrowIfFailed(d3d12Device->CreateSharedHandle(inputReadyFence12.get(), nullptr, GENERIC_ALL, nullptr, &inputReadyFenceHandle));
	DX::ThrowIfFailed(d3d11Device->OpenSharedFence(inputReadyFenceHandle, IID_PPV_ARGS(inputReadyFence11.put())));
	CloseHandle(inputReadyFenceHandle);

	swapChainProxy = new DXGISwapChainProxy(swapChain);

	RecreateWrappedResources(swapChainDesc);

	const bool pipelined = globals::features::upscaling.settings.frameGenerationPipelined;
	logger::info("[DX12SwapChain] Frame-generation hand-off at startup: {} ({} swap-chain buffers, {} frame generation)",
		pipelined ? "pipelined (3-deep D3D12 input ring, D3D11 released right after the input copy)" :
					"serialized (D3D11 waits for the frame-generation present)",
		proxyBufferCount, useDLSSG ? "DLSS" : "FSR");
}

void DX12SwapChain::RecreateWrappedResources(const DXGI_SWAP_CHAIN_DESC1& desc)
{
	D3D11_TEXTURE2D_DESC texDesc11{};
	texDesc11.Width = desc.Width;
	texDesc11.Height = desc.Height;
	texDesc11.MipLevels = 1;
	texDesc11.ArraySize = 1;
	texDesc11.Format = desc.Format;
	texDesc11.SampleDesc.Count = 1;
	texDesc11.SampleDesc.Quality = 0;
	texDesc11.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;

	// Build both replacements before releasing the active resources so a failed
	// allocation cannot leave the proxy with only half of its interop textures.
	auto newSwapChainBuffer = std::make_unique<WrappedResource>(texDesc11, d3d11Device.get(), d3d12Device.get());

	// DLSS-G HUD-less input: same size and format as the backbuffer (required by DLSS-G).
	std::unique_ptr<WrappedResource> newHudlessBuffer;
	if (useDLSSG)
		newHudlessBuffer = std::make_unique<WrappedResource>(texDesc11, d3d11Device.get(), d3d12Device.get());

	// UI buffer uses R8G8B8A8_UNORM - vanilla UI is SDR and 8-bit precision
	texDesc11.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	auto newUiBuffer = std::make_unique<WrappedResource>(texDesc11, d3d11Device.get(), d3d12Device.get());

	delete swapChainBufferWrapped;
	delete uiBufferWrapped;
	delete hudlessBufferWrapped;
	swapChainBufferWrapped = newSwapChainBuffer.release();
	uiBufferWrapped = newUiBuffer.release();
	hudlessBufferWrapped = newHudlessBuffer.release();

	const float clearColor[4]{};
	d3d11Context->ClearRenderTargetView(swapChainBufferWrapped->rtv, clearColor);
	d3d11Context->ClearRenderTargetView(uiBufferWrapped->rtv, clearColor);
	if (hudlessBufferWrapped)
		d3d11Context->ClearRenderTargetView(hudlessBufferWrapped->rtv, clearColor);
}

DXGISwapChainProxy* DX12SwapChain::GetSwapChainProxy()
{
	return swapChainProxy;
}

void DX12SwapChain::SetD3D11Device(ID3D11Device* a_d3d11Device)
{
	DX::ThrowIfFailed(a_d3d11Device->QueryInterface(IID_PPV_ARGS(&d3d11Device)));
}

void DX12SwapChain::SetD3D11DeviceContext(ID3D11DeviceContext* a_d3d11Context)
{
	DX::ThrowIfFailed(a_d3d11Context->QueryInterface(IID_PPV_ARGS(&d3d11Context)));
}

HRESULT DX12SwapChain::GetBuffer(UINT buffer, REFIID riid, void** ppSurface)
{
	if (!ppSurface)
		return E_POINTER;

	*ppSurface = nullptr;
	if (buffer != 0 || !swapChainBufferWrapped || !swapChainBufferWrapped->resource11)
		return DXGI_ERROR_INVALID_CALL;

	// IDXGISwapChain::GetBuffer returns an owned COM reference. Returning the raw
	// pointer here let the caller's Release destroy the shared texture while the
	// D3D12 side still retained and submitted its corresponding resource.
	return swapChainBufferWrapped->resource11->QueryInterface(riid, ppSurface);
}

HRESULT DX12SwapChain::ResizeBuffers(UINT bufferCount, UINT width, UINT height, DXGI_FORMAT format, UINT flags)
{
	if (!swapChain)
		return DXGI_ERROR_INVALID_CALL;

	// DXGI defines zero as "preserve the current buffer count". FidelityFX's
	// frame-generation swap-chain stores the supplied value verbatim and uses it
	// as its replacement-buffer count, so forwarding zero leaves it with no valid
	// source resource at the next Present.
	UINT effectiveBufferCount = bufferCount ? bufferCount : swapChainDesc.BufferCount;
	if (!bufferCount)
		logger::warn("[FidelityFX] Normalized ResizeBuffers count from 0 to {} to preserve replacement buffers", effectiveBufferCount);
	if (effectiveBufferCount != proxyBufferCount) {
		if (proxyBufferCount == kLegacyBufferCount) {
			logger::error("[DX12SwapChain] Rejected unsupported resize buffer count {} (CS requires 2)", effectiveBufferCount);
			return DXGI_ERROR_UNSUPPORTED;
		}
		// The game only ever sees the wrapped D3D11 buffer; keep the proxy's own count.
		static bool countOverrideLogged = false;
		if (!countOverrideLogged) {
			countOverrideLogged = true;
			logger::info("[DX12SwapChain] ResizeBuffers asked for {} buffers; keeping {} for the pipelined hand-off", effectiveBufferCount, proxyBufferCount);
		}
		effectiveBufferCount = proxyBufferCount;
	}

	// Nothing queued may still reference the old buffers or the input ring built from them.
	WaitForGPUIdle();

	// These references are to FidelityFX replacement buffers. They must not keep
	// the old generation alive across the provider's resize, and must be refreshed
	// before CS records another copy.
	for (auto& buffer : swapChainBuffers)
		buffer = nullptr;
	const HRESULT result = swapChain->ResizeBuffers(effectiveBufferCount, width, height, format, flags);
	if (FAILED(result))
		return result;

	DXGI_SWAP_CHAIN_DESC1 resizedDesc{};
	const HRESULT descResult = swapChain->GetDesc1(&resizedDesc);
	if (FAILED(descResult))
		return descResult;

	const bool wrappedResourcesChanged = resizedDesc.Width != swapChainDesc.Width ||
	                                     resizedDesc.Height != swapChainDesc.Height ||
	                                     resizedDesc.Format != swapChainDesc.Format;
	if (wrappedResourcesChanged)
		RecreateWrappedResources(resizedDesc);  // the input ring follows on the next pipelined present
	swapChainDesc = resizedDesc;

	for (UINT i = 0; i < proxyBufferCount; i++)
		DX::ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(swapChainBuffers[i].put())));
	frameIndex = swapChain->GetCurrentBackBufferIndex();
	return S_OK;
}

void DX12SwapChain::WaitForGPUIdle()
{
	if (!commandQueue || !d3d12Fence)
		return;
	fenceValue++;
	if (SUCCEEDED(commandQueue->Signal(d3d12Fence.get(), fenceValue)))
		DX::ThrowIfFailed(d3d12Fence->SetEventOnCompletion(fenceValue, nullptr));
}

void DX12SwapChain::ReleaseFrameRing()
{
	for (auto& slot : frameRing)
		slot = FrameRingSlot{};
	std::fill(std::begin(frameRingSources), std::end(frameRingSources), nullptr);
	frameRingReady = false;
}

bool DX12SwapChain::EnsureFrameRing()
{
	if (frameRingFailed || !d3d12Device || !swapChainBufferWrapped || !uiBufferWrapped)
		return false;

	ID3D12Resource* const sources[5] = {
		swapChainBufferWrapped->resource.get(),
		uiBufferWrapped->resource.get(),
		hudlessBufferWrapped ? hudlessBufferWrapped->resource.get() : nullptr,
		depthBufferShared12 ? depthBufferShared12->resource.get() : nullptr,
		motionVectorBufferShared12 ? motionVectorBufferShared12->resource.get() : nullptr,
	};
	if (frameRingReady && std::equal(std::begin(sources), std::end(sources), std::begin(frameRingSources)))
		return true;

	// First use, or the shared inputs were recreated (resize, shared depth/motion created later):
	// nothing queued may still read the old slots.
	WaitForGPUIdle();
	ReleaseFrameRing();

	try {
		const auto makeTwin = [&](ID3D12Resource* source, const wchar_t* name, UINT slot) {
			winrt::com_ptr<ID3D12Resource> twin;
			if (!source)
				return twin;
			D3D12_RESOURCE_DESC desc = source->GetDesc();
			desc.Alignment = 0;
			desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			desc.Flags &= ~D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
			const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);
			DX::ThrowIfFailed(d3d12Device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &desc,
				D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(twin.put())));
			twin->SetName(std::format(L"CS FG input ring {} {}", name, slot).c_str());
			return twin;
		};

		for (UINT slot = 0; slot < kFrameRingSize; slot++) {
			auto& ring = frameRing[slot];
			ring.scene = makeTwin(sources[0], L"scene", slot);
			ring.ui = makeTwin(sources[1], L"UI", slot);
			ring.hudless = makeTwin(sources[2], L"HUD-less", slot);
			ring.depth = makeTwin(sources[3], L"depth", slot);
			ring.motionVectors = makeTwin(sources[4], L"motion vectors", slot);

			if (!ringCopyAllocators[slot]) {
				DX::ThrowIfFailed(d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(ringCopyAllocators[slot].put())));
				DX::ThrowIfFailed(d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ringCopyAllocators[slot].get(), nullptr, IID_PPV_ARGS(ringCopyLists[slot].put())));
				DX::ThrowIfFailed(ringCopyLists[slot]->Close());
			}
		}
	} catch (const std::exception& e) {
		logger::error("[DX12SwapChain] Could not create the frame-generation input ring ({}); falling back to the serialized hand-off", e.what());
		ReleaseFrameRing();
		frameRingFailed = true;
		return false;
	}

	std::copy(std::begin(sources), std::end(sources), std::begin(frameRingSources));
	frameRingReady = true;
	logger::info("[DX12SwapChain] Frame-generation input ring ready: {} slots of {}x{} (scene, UI{}{})",
		kFrameRingSize, swapChainDesc.Width, swapChainDesc.Height,
		sources[2] ? ", HUD-less" : "", (sources[3] && sources[4]) ? ", depth, motion vectors" : "");
	return true;
}

HRESULT DX12SwapChain::Present(UINT SyncInterval, UINT Flags)
{
	auto& upscaling = globals::features::upscaling;

	// Scale UI brightness BEFORE fence sync so the D3D11 UIBrightnessCS dispatch
	// is covered by the D3D11→D3D12 fence. Without this, FidelityFX may read
	// uiBufferWrapped on D3D12 before the PQ encoding completes on D3D11.
	// Only runs when HDR Display feature is loaded (UIBrightnessCS may not exist otherwise)
	auto* hdr = globals::features::hdrDisplay.loaded ? &globals::features::hdrDisplay : nullptr;
	if (hdr)
		hdr->ScaleUIBrightnessForFG();

	bool isHDR = hdr && hdr->settings.enableHDR;

	const bool useFrameGeneration = upscaling.ShouldUseFrameGenerationThisFrame();

	// DLSS-G does not composite UI itself. When frame generation routed this frame's UI into
	// uiBufferWrapped (HDRDisplay::SetUIBuffer), composite it here on D3D11 and keep a HUD-less
	// copy of the scene, so DLSS-G gets HUD-less color + UI color/alpha and leaves the HUD out of
	// the interpolation. Runs before the fence signal so D3D12 sees the finished images.
	bool dlssgUISeparated = false;
	if (useDLSSG && uiRoutedForFGComposite)
		dlssgUISeparated = ComposeUIForDLSSG();
	uiRoutedForFGComposite = false;

	// Pipelined hand-off (Upscaling.frameGenerationPipelined, switchable live): copy this frame's
	// D3D11 inputs into a private ring slot and release D3D11 right after that copy, instead of
	// after the frame-generation present (which waits on the presentation of the previous frame).
	const bool pipelined = upscaling.settings.frameGenerationPipelined && EnsureFrameRing();
	if (static_cast<int>(pipelined) != loggedHandOffMode) {
		logger::info("[DX12SwapChain] Frame-generation hand-off: {} ({} swap-chain buffers)",
			pipelined ? "pipelined, D3D11 released after the input copy" : "serialized, D3D11 waits for the frame-generation present",
			proxyBufferCount);
		loggedHandOffMode = static_cast<int>(pipelined);
	}

	// D3D11 -> D3D12: the D3D12 queue waits until D3D11 has written this frame's inputs
	// (includes ApplyHDR scene encoding AND UIBrightnessCS)
	inputReadyValue++;
	DX::ThrowIfFailed(d3d11Context->Signal(inputReadyFence11.get(), inputReadyValue));
	DX::ThrowIfFailed(commandQueue->Wait(inputReadyFence12.get(), inputReadyValue));

	// What frame generation reads this frame: the shared D3D11 textures (serialized) or the ring slot.
	ID3D12Resource* sceneInput = swapChainBufferWrapped->resource.get();
	ID3D12Resource* uiInput = uiBufferWrapped->resource.get();
	ID3D12Resource* hudlessInput = hudlessBufferWrapped ? hudlessBufferWrapped->resource.get() : nullptr;
	ID3D12Resource* depthInput = depthBufferShared12 ? depthBufferShared12->resource.get() : nullptr;
	ID3D12Resource* motionVectorInput = motionVectorBufferShared12 ? motionVectorBufferShared12->resource.get() : nullptr;

	if (pipelined) {
		const UINT slot = static_cast<UINT>(presentCounter % kFrameRingSize);
		auto& ring = frameRing[slot];

		// The slot's copy list was last submitted kFrameRingSize frames ago.
		if (ringCopyFenceValues[slot])
			DX::ThrowIfFailed(d3d12Fence->SetEventOnCompletion(ringCopyFenceValues[slot], nullptr));
		DX::ThrowIfFailed(ringCopyAllocators[slot]->Reset());
		auto* copyList = ringCopyLists[slot].get();
		DX::ThrowIfFailed(copyList->Reset(ringCopyAllocators[slot].get(), nullptr));

		// Only copy what is consumed this frame: depth/motion/UI are written (and read) on frame
		// generation frames only, the HUD-less copy only when the DLSS-G UI composite ran.
		struct InputCopy
		{
			ID3D12Resource* source;
			ID3D12Resource* destination;
		};
		InputCopy copies[5]{};
		UINT copyCount = 0;
		const auto addCopy = [&](ID3D12Resource* source, ID3D12Resource* destination) {
			if (source && destination)
				copies[copyCount++] = { source, destination };
		};
		addCopy(sceneInput, ring.scene.get());
		if (useFrameGeneration) {
			addCopy(uiInput, ring.ui.get());
			addCopy(depthInput, ring.depth.get());
			addCopy(motionVectorInput, ring.motionVectors.get());
		}
		if (dlssgUISeparated)
			addCopy(hudlessInput, ring.hudless.get());

		D3D12_RESOURCE_BARRIER barriers[10]{};
		for (UINT i = 0; i < copyCount; i++) {
			barriers[2 * i] = CD3DX12_RESOURCE_BARRIER::Transition(copies[i].source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
			barriers[2 * i + 1] = CD3DX12_RESOURCE_BARRIER::Transition(copies[i].destination, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		}
		copyList->ResourceBarrier(2 * copyCount, barriers);
		for (UINT i = 0; i < copyCount; i++)
			copyList->CopyResource(copies[i].destination, copies[i].source);
		for (UINT i = 0; i < copyCount; i++) {
			barriers[2 * i] = CD3DX12_RESOURCE_BARRIER::Transition(copies[i].source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
			barriers[2 * i + 1] = CD3DX12_RESOURCE_BARRIER::Transition(copies[i].destination, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
		}
		copyList->ResourceBarrier(2 * copyCount, barriers);
		DX::ThrowIfFailed(copyList->Close());

		ID3D12CommandList* copyLists[] = { copyList };
		commandQueue->ExecuteCommandLists(1, copyLists);

		// D3D11 may overwrite the shared inputs as soon as the copies have executed; it no longer
		// waits for the frame-generation present of this frame.
		fenceValue++;
		DX::ThrowIfFailed(commandQueue->Signal(d3d12Fence.get(), fenceValue));
		ringCopyFenceValues[slot] = fenceValue;
		DX::ThrowIfFailed(d3d11Context->Wait(d3d11Fence.get(), fenceValue));

		sceneInput = ring.scene.get();
		uiInput = ring.ui.get();
		hudlessInput = ring.hudless.get();
		depthInput = ring.depth.get();
		motionVectorInput = ring.motionVectors.get();
	}

	// New frame, reset
	if (frameFenceValues[frameIndex])
		DX::ThrowIfFailed(d3d12Fence->SetEventOnCompletion(frameFenceValues[frameIndex], nullptr));
	DX::ThrowIfFailed(commandAllocators[frameIndex]->Reset());
	DX::ThrowIfFailed(commandLists[frameIndex]->Reset(commandAllocators[frameIndex].get(), nullptr));

	// Copy the frame to the swap chain buffer
	{
		auto fakeSwapChain = sceneInput;
		auto realSwapChain = swapChainBuffers[frameIndex].get();
		{
			std::vector<D3D12_RESOURCE_BARRIER> barriers;
			barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(fakeSwapChain, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE));
			barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(realSwapChain, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST));
			commandLists[frameIndex]->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
		}

		commandLists[frameIndex]->CopyResource(realSwapChain, fakeSwapChain);

		{
			std::vector<D3D12_RESOURCE_BARRIER> barriers;
			barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(fakeSwapChain, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
			barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(realSwapChain, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT));
			commandLists[frameIndex]->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
		}
	}

	// The frame being presented: State::Reset() already advanced frameCount for the next frame
	// before this present ran, so the frame whose Reflex sleep/markers used frameCount is - 1.
	const uint32_t presentFrameIndex = globals::state ? globals::state->frameCount - 1 : 0;

	if (useDLSSG) {
		auto& slDX12 = upscaling.streamlineDX12;
		const bool dlssgEnabled = useFrameGeneration && slDX12.featureDLSSG && depthInput && motionVectorInput;

		if (auto* token = slDX12.GetFrameToken(presentFrameIndex)) {
			slDX12.SetFrameConstants(*token, slDX12.viewport);

			if (dlssgEnabled) {
				// Depth and motion vectors are copied before upscaling (render resolution, top-left
				// sub-rect when an upscaler is active); HUD-less and UI match the backbuffer.
				const float2 screenSize{ (float)globals::game::graphicsState->screenWidth, (float)globals::game::graphicsState->screenHeight };
				const float2 renderSize = screenSize * upscaling.resolutionScale;
				const sl::Extent renderExtent{ 0, 0,
					std::clamp(static_cast<uint32_t>(renderSize.x), 1u, swapChainDesc.Width),
					std::clamp(static_cast<uint32_t>(renderSize.y), 1u, swapChainDesc.Height) };
				const sl::Extent backbufferExtent{ 0, 0, swapChainDesc.Width, swapChainDesc.Height };

				slDX12.TagDLSSGResources(*token, commandLists[frameIndex].get(),
					depthInput,
					motionVectorInput,
					renderExtent,
					dlssgUISeparated ? hudlessInput : nullptr,
					dlssgUISeparated ? uiInput : nullptr,
					backbufferExtent);
			}
		}

		// Keep DLSS-G's resources while frame generation is only paused (menus); release them when
		// frame generation is switched off in the settings.
		const auto& settings = upscaling.settings;
		slDX12.ConfigureDLSSG(dlssgEnabled, settings.frameGenerationMode != 0, settings.dlssgFramesToGenerate,
			settings.dlssgDynamic, settings.dlssgDynamicTargetFps, settings.dlssgUIRecomposition && CanSeparateUIForDLSSG());
	} else {
		const auto& settings = upscaling.settings;
		upscaling.fidelityFX.ApplyFramePacingTuning(settings.fsrFGPacingSafetyMarginMs, settings.fsrFGPacingVarianceFactor);
		upscaling.fidelityFX.Present(useFrameGeneration, isHDR, depthInput, motionVectorInput, uiInput);
	}

	DX::ThrowIfFailed(commandLists[frameIndex]->Close());

	ID3D12CommandList* commandListsToExecute[] = { commandLists[frameIndex].get() };
	commandQueue->ExecuteCommandLists(1, commandListsToExecute);

	// Present the frame. DLSS-G matches the presented frame to its constants/tags through the
	// ePresentStart marker, so the markers carry the same frame index.
	if (useDLSSG) {
		auto& slDX12 = upscaling.streamlineDX12;
		slDX12.SetPCLMarker(sl::PCLMarker::eRenderSubmitEnd, presentFrameIndex);
		slDX12.SetPCLMarker(sl::PCLMarker::ePresentStart, presentFrameIndex);
	}
	const HRESULT presentResult = swapChain->Present(SyncInterval, Flags);
	if (useDLSSG) {
		auto& slDX12 = upscaling.streamlineDX12;
		slDX12.SetPCLMarker(sl::PCLMarker::ePresentEnd, presentFrameIndex);
		slDX12.QueryDLSSGState(commandQueue.get(), slDX12.dlssgEnabledLastFrame);
	}
	DX::ThrowIfFailed(presentResult);

	// Everything queued for this frame, including the frame-generation present work
	fenceValue++;
	DX::ThrowIfFailed(commandQueue->Signal(d3d12Fence.get(), fenceValue));
	frameFenceValues[frameIndex] = fenceValue;
	// Serialized hand-off: D3D11 waits for the frame-generation present of this frame.
	if (!pipelined)
		DX::ThrowIfFailed(d3d11Context->Wait(d3d11Fence.get(), fenceValue));

	// Update the frame index
	frameIndex = swapChain->GetCurrentBackBufferIndex();
	presentCounter++;

	float clearColor[4]{ 0, 0, 0, 0 };
	d3d11Context->ClearRenderTargetView(uiBufferWrapped->rtv, clearColor);

	// If VSync is disabled, use frame limiter to prevent tearing and optimise pacing
	if (SyncInterval == 0)
		upscaling.FrameLimiter();

	return S_OK;
}

HRESULT DX12SwapChain::GetDevice(REFIID uuid, void** ppDevice)
{
	if (uuid == __uuidof(ID3D11Device) || uuid == __uuidof(ID3D11Device1) || uuid == __uuidof(ID3D11Device2) || uuid == __uuidof(ID3D11Device3) || uuid == __uuidof(ID3D11Device4) || uuid == __uuidof(ID3D11Device5)) {
		*ppDevice = d3d11Device.get();
		return S_OK;
	}

	return swapChain->GetDevice(uuid, ppDevice);
}

HANDLE DX12SwapChain::GetFrameLatencyWaitableObject()
{
	return swapChain->GetFrameLatencyWaitableObject();
}

float DX12SwapChain::GetFrameTime() const
{
	// Calculate frame time based on swap chain presentation
	static float lastPresentTime = 0.0f;
	static float frameTime = 1.0f / 60.0f;  // Default to 60 fps
	static LARGE_INTEGER frequency = {};
	static LARGE_INTEGER currentTime = {};

	if (frequency.QuadPart == 0) {
		QueryPerformanceFrequency(&frequency);
	}

	QueryPerformanceCounter(&currentTime);
	float time = static_cast<float>(currentTime.QuadPart) / static_cast<float>(frequency.QuadPart);

	if (lastPresentTime > 0.0f) {
		frameTime = time - lastPresentTime;
	}
	lastPresentTime = time;

	return frameTime;
}

WrappedResource::WrappedResource(D3D11_TEXTURE2D_DESC a_texDesc, ID3D11Device5* a_d3d11Device, ID3D12Device* a_d3d12Device)
{
	// Create D3D11 shared texture directly instead of wrapping D3D12 resource
	a_texDesc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
	DX::ThrowIfFailed(a_d3d11Device->CreateTexture2D(&a_texDesc, nullptr, &resource11));

	// Get shared handle from D3D11 texture to enable D3D12 access
	winrt::com_ptr<IDXGIResource1> dxgiResource;
	DX::ThrowIfFailed(resource11->QueryInterface(IID_PPV_ARGS(dxgiResource.put())));
	HANDLE sharedHandle = nullptr;
	DX::ThrowIfFailed(dxgiResource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &sharedHandle));

	// Open the shared D3D11 texture as D3D12 resource
	DX::ThrowIfFailed(a_d3d12Device->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(resource.put())));
	CloseHandle(sharedHandle);

	if (a_texDesc.BindFlags & D3D11_BIND_SHADER_RESOURCE) {
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Format = a_texDesc.Format;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MostDetailedMip = 0;
		srvDesc.Texture2D.MipLevels = 1;

		DX::ThrowIfFailed(a_d3d11Device->CreateShaderResourceView(resource11, &srvDesc, &srv));
	}

	if (a_texDesc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) {
		if (a_texDesc.ArraySize > 1) {
			D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = a_texDesc.Format;
			uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
			uavDesc.Texture2DArray.FirstArraySlice = 0;
			uavDesc.Texture2DArray.ArraySize = a_texDesc.ArraySize;

			DX::ThrowIfFailed(a_d3d11Device->CreateUnorderedAccessView(resource11, &uavDesc, &uav));
		} else {
			D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = a_texDesc.Format;
			uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
			uavDesc.Texture2D.MipSlice = 0;

			DX::ThrowIfFailed(a_d3d11Device->CreateUnorderedAccessView(resource11, &uavDesc, &uav));
		}
	}

	if (a_texDesc.BindFlags & D3D11_BIND_RENDER_TARGET) {
		D3D11_RENDER_TARGET_VIEW_DESC rtvDesc = {};
		rtvDesc.Format = a_texDesc.Format;
		rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
		rtvDesc.Texture2D.MipSlice = 0;
		DX::ThrowIfFailed(a_d3d11Device->CreateRenderTargetView(resource11, &rtvDesc, &rtv));
	}
}

WrappedResource::~WrappedResource()
{
	if (resource11) {
		resource11->Release();
		resource11 = nullptr;
	}
	if (srv) {
		srv->Release();
		srv = nullptr;
	}
	if (uav) {
		uav->Release();
		uav = nullptr;
	}
	if (rtv) {
		rtv->Release();
		rtv = nullptr;
	}
	// resource (winrt::com_ptr) will be automatically released
}

DXGISwapChainProxy::DXGISwapChainProxy(IDXGISwapChain4* a_swapChain)
{
	swapChain = a_swapChain;
}

/****IUknown****/
HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::QueryInterface(REFIID riid, void** ppvObj)
{
	auto ret = swapChain->QueryInterface(riid, ppvObj);
	if (*ppvObj)
		*ppvObj = this;
	return ret;
}

ULONG STDMETHODCALLTYPE DXGISwapChainProxy::AddRef()
{
	return swapChain->AddRef();
}

ULONG STDMETHODCALLTYPE DXGISwapChainProxy::Release()
{
	return swapChain->Release();
}

/****IDXGIObject****/
HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::SetPrivateData(_In_ REFGUID Name, UINT DataSize, _In_reads_bytes_(DataSize) const void* pData)
{
	return swapChain->SetPrivateData(Name, DataSize, pData);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::SetPrivateDataInterface(_In_ REFGUID Name, _In_opt_ const IUnknown* pUnknown)
{
	return swapChain->SetPrivateDataInterface(Name, pUnknown);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetPrivateData(_In_ REFGUID Name, _Inout_ UINT* pDataSize, _Out_writes_bytes_(*pDataSize) void* pData)
{
	return swapChain->GetPrivateData(Name, pDataSize, pData);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetParent(_In_ REFIID riid, _COM_Outptr_ void** ppParent)
{
	return swapChain->GetParent(riid, ppParent);
}

/****IDXGIDeviceSubObject****/
HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetDevice(_In_ REFIID riid, _COM_Outptr_ void** ppDevice)
{
	return globals::features::upscaling.dx12SwapChain.GetDevice(riid, ppDevice);
}

/****IDXGISwapChain****/
HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::Present(UINT SyncInterval, UINT Flags)
{
	return globals::features::upscaling.dx12SwapChain.Present(SyncInterval, Flags);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetBuffer(UINT buffer, _In_ REFIID riid, _COM_Outptr_ void** ppSurface)
{
	return globals::features::upscaling.dx12SwapChain.GetBuffer(buffer, riid, ppSurface);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::SetFullscreenState(BOOL Fullscreen, _In_opt_ IDXGIOutput* pTarget)
{
	return swapChain->SetFullscreenState(Fullscreen, pTarget);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetFullscreenState(_Out_opt_ BOOL* pFullscreen, _COM_Outptr_opt_result_maybenull_ IDXGIOutput** ppTarget)
{
	return swapChain->GetFullscreenState(pFullscreen, ppTarget);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetDesc(_Out_ DXGI_SWAP_CHAIN_DESC* pDesc)
{
	return swapChain->GetDesc(pDesc);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::ResizeBuffers(UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT NewFormat, UINT SwapChainFlags)
{
	return globals::features::upscaling.dx12SwapChain.ResizeBuffers(BufferCount, Width, Height, NewFormat, SwapChainFlags);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::ResizeTarget(_In_ const DXGI_MODE_DESC* pNewTargetParameters)
{
	return swapChain->ResizeTarget(pNewTargetParameters);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetContainingOutput(_COM_Outptr_ IDXGIOutput** ppOutput)
{
	return swapChain->GetContainingOutput(ppOutput);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetFrameStatistics(_Out_ DXGI_FRAME_STATISTICS* pStats)
{
	return swapChain->GetFrameStatistics(pStats);
}

HRESULT STDMETHODCALLTYPE DXGISwapChainProxy::GetLastPresentCount(_Out_ UINT* pLastPresentCount)
{
	return swapChain->GetLastPresentCount(pLastPresentCount);
}

void DX12SwapChain::SetColorSpace(bool enableHDR)
{
	if (!swapChain)
		return;

	if (enableHDR) {
		swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
		logger::info("[DX12SwapChain] Set color space to HDR10 (PQ/BT.2020)");
	} else {
		swapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
		logger::info("[DX12SwapChain] Set color space to SDR (sRGB)");
	}
}

DX12SwapChain::BlurResources DX12SwapChain::GetBlurResources() const
{
	BlurResources res;
	if (swapChainBufferWrapped) {
		res.backbufferTex = swapChainBufferWrapped->resource11;
		res.backbufferRTV = swapChainBufferWrapped->rtv;
		res.backbufferSRV = swapChainBufferWrapped->srv;
	}
	if (uiBufferWrapped) {
		res.uiBufferSRV = uiBufferWrapped->srv;
		res.uiBufferRTV = uiBufferWrapped->rtv;
	}
	return res;
}

void DX12SwapChain::CreateSharedResources()
{
	auto renderer = globals::game::renderer;

	// Create depth buffer
	auto& main = renderer->GetRuntimeData().renderTargets[RE::RENDER_TARGETS::kMAIN];
	D3D11_TEXTURE2D_DESC texDesc{};
	main.texture->GetDesc(&texDesc);
	texDesc.Format = DXGI_FORMAT_R32_FLOAT;
	depthBufferShared12 = new WrappedResource(texDesc, d3d11Device.get(), d3d12Device.get());

	// Create motion vector buffer
	auto& motionVector = renderer->GetRuntimeData().renderTargets[RE::RENDER_TARGETS::kMOTION_VECTOR];
	motionVector.texture->GetDesc(&texDesc);
	motionVectorBufferShared12 = new WrappedResource(texDesc, d3d11Device.get(), d3d12Device.get());
}
