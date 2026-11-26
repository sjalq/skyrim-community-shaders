#include "ShadowmapCascadeRasterizerFix.h"

#include "Globals.h"
#include "State.h"

namespace FarShadowCache
{
	namespace
	{
		constexpr uint32_t kWarmupFrames = 30;
		constexpr uint32_t kStatsFrames = 600;
		constexpr float kMaxMoveSq = 256.0f * 256.0f;  // ~3.6m before forced refresh
		constexpr float kMinForwardDot = 0.985f;       // ~10 deg turn before forced refresh
		constexpr float kMaxSunDeltaSq = 1e-5f;        // ~0.18 deg sun change
		constexpr float kMaxSplitDelta = 16.0f;

		struct CacheState
		{
			std::atomic<int> interval{ 1 };
			int appliedInterval = 1;
			uint32_t warmupUntilFrame = 0;
			const RE::TESObjectCELL* lastCell = nullptr;
			const RE::BSShadowDirectionalLight* cachedLight = nullptr;
			bool valid = false;
			uint32_t lastRefreshFrame = 0;
			uint32_t skipFrame = 0xFFFFFFFFu;

			winrt::com_ptr<ID3D11Texture2D> cachedTex;
			D3D11_TEXTURE2D_DESC cachedDesc{};
			REX::W32::XMFLOAT4X4 cachedLightTransform{};
			RE::NiPoint3 cachedSunVec{};
			RE::NiPoint3 cachedCamPos{};
			RE::NiPoint3 cachedCamForward{};
			float cachedStartSplit1 = 0.0f;
			float cachedEndSplit1 = 0.0f;

			bool statsPending = false;
			uint32_t statsUntilFrame = 0;
			uint32_t statsRefreshed = 0;
			uint32_t statsSkipped = 0;
			int32_t lastRT = -1;
			uint32_t lastSlice = 0;
		};

		CacheState& Get()
		{
			static CacheState s;
			return s;
		}

		bool FrameBefore(uint32_t a_frame, uint32_t a_limit)
		{
			return static_cast<int32_t>(a_frame - a_limit) < 0;
		}

		bool GetCameraPose(RE::NiPoint3& a_pos, RE::NiPoint3& a_forward)
		{
			const auto* cam = RE::PlayerCamera::GetSingleton();
			if (!cam || !cam->cameraRoot)
				return false;
			a_pos = cam->cameraRoot->world.translate;
			a_forward = cam->cameraRoot->world.rotate.GetVectorY();
			a_forward.Unitize();
			return true;
		}

		bool IsSkipFrame(RE::BSShadowDirectionalLight* a_light, uint32_t a_frame)
		{
			auto& s = Get();
			return s.valid && s.cachedTex && s.cachedLight == a_light && s.skipFrame == a_frame;
		}

		void CaptureCascade1(RE::BSShadowDirectionalLight* a_light, uint32_t a_frame)
		{
			if (REL::Module::IsVR() || !a_light)
				return;
			auto& s = Get();
			auto& lightData = a_light->GetRuntimeData();
			if (lightData.shadowmapDescriptors.size() < 2)
				return;

			auto& desc1 = lightData.shadowmapDescriptors[1];
			auto* renderer = globals::game::renderer;
			auto* device = globals::d3d::device;
			auto* context = globals::d3d::context;
			if (!renderer || !device || !context)
				return;

			const auto rtIdx = static_cast<int32_t>(desc1.renderTarget);
			if (rtIdx < 0 || rtIdx >= RE::RENDER_TARGETS_DEPTHSTENCIL::kTOTAL)
				return;

			auto& ds = renderer->GetDepthStencilData().depthStencils[rtIdx];
			if (!ds.texture)
				return;

			D3D11_TEXTURE2D_DESC srcDesc{};
			ds.texture->GetDesc(&srcDesc);
			if (desc1.shadowmapIndex >= srcDesc.ArraySize)
				return;

			if (!s.cachedTex || s.cachedDesc.Width != srcDesc.Width || s.cachedDesc.Height != srcDesc.Height || s.cachedDesc.Format != srcDesc.Format) {
				s.cachedTex = nullptr;
				D3D11_TEXTURE2D_DESC cDesc = srcDesc;
				cDesc.ArraySize = 1;
				cDesc.MipLevels = 1;
				cDesc.MiscFlags = 0;
				if (FAILED(device->CreateTexture2D(&cDesc, nullptr, s.cachedTex.put()))) {
					s.valid = false;
					return;
				}
				s.cachedDesc = cDesc;
			}

			const UINT srcSubres = D3D11CalcSubresource(0, desc1.shadowmapIndex, srcDesc.MipLevels);
			context->CopySubresourceRegion(s.cachedTex.get(), 0, 0, 0, 0, ds.texture, srcSubres, nullptr);

			auto& dirData = a_light->GetShadowDirectionalLightRuntimeData();
			s.cachedLight = a_light;
			s.cachedLightTransform = desc1.lightTransform;
			s.cachedSunVec = dirData.sunVector;
			s.cachedStartSplit1 = dirData.startSplitDistances[1];
			s.cachedEndSplit1 = dirData.endSplitDistances[1];
			GetCameraPose(s.cachedCamPos, s.cachedCamForward);
			s.lastRefreshFrame = a_frame;
			s.valid = true;
			s.lastRT = rtIdx;
			s.lastSlice = desc1.shadowmapIndex;
			s.statsRefreshed++;
		}

		void RestoreCascade1(RE::BSShadowDirectionalLight* a_light)
		{
			if (REL::Module::IsVR() || !a_light)
				return;
			auto& s = Get();
			if (!s.valid || !s.cachedTex)
				return;
			auto& lightData = a_light->GetRuntimeData();
			if (lightData.shadowmapDescriptors.size() < 2)
				return;

			auto& desc1 = lightData.shadowmapDescriptors[1];
			auto* renderer = globals::game::renderer;
			auto* context = globals::d3d::context;
			if (!renderer || !context)
				return;

			const auto rtIdx = static_cast<int32_t>(desc1.renderTarget);
			if (rtIdx < 0 || rtIdx >= RE::RENDER_TARGETS_DEPTHSTENCIL::kTOTAL)
				return;

			auto& ds = renderer->GetDepthStencilData().depthStencils[rtIdx];
			if (!ds.texture)
				return;

			D3D11_TEXTURE2D_DESC dstDesc{};
			ds.texture->GetDesc(&dstDesc);
			if (desc1.shadowmapIndex >= dstDesc.ArraySize || dstDesc.Width != s.cachedDesc.Width || dstDesc.Height != s.cachedDesc.Height || dstDesc.Format != s.cachedDesc.Format)
				return;

			const UINT dstSubres = D3D11CalcSubresource(0, desc1.shadowmapIndex, dstDesc.MipLevels);
			context->CopySubresourceRegion(ds.texture, dstSubres, 0, 0, 0, s.cachedTex.get(), 0, nullptr);
			desc1.lightTransform = s.cachedLightTransform;
			s.statsSkipped++;
		}
	}

	int GetInterval()
	{
		return Get().interval.load(std::memory_order_relaxed);
	}

	void SetInterval(int a_interval)
	{
		Get().interval.store(std::clamp(a_interval, kMinInterval, kMaxInterval), std::memory_order_relaxed);
	}

	CullingGuard::CullingGuard(RE::BSShadowDirectionalLight* a_light, const RE::TESObjectCELL* a_interiorCell)
	{
		if (REL::Module::IsVR() || !a_light)
			return;

		auto& s = Get();
		const int interval = s.interval.load(std::memory_order_relaxed);
		const uint32_t frame = globals::state->frameCount;

		if (interval != s.appliedInterval) {
			logger::info("[FarShadowCache] Interval {} -> {} (1 renders Cascade 1 every frame)", s.appliedInterval, interval);
			s.appliedInterval = interval;
			s.warmupUntilFrame = frame + kWarmupFrames;
			s.valid = false;
			s.skipFrame = 0xFFFFFFFFu;
			s.statsPending = interval > 1;
			s.statsUntilFrame = frame + kStatsFrames;
			s.statsRefreshed = 0;
			s.statsSkipped = 0;
		}

		if (interval <= 1) {
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}

		if (s.statsPending && !FrameBefore(frame, s.statsUntilFrame)) {
			logger::info("[FarShadowCache] Interval {}: skipped {} / {} Cascade 1 renders over {} frames (rt={}, slice={}, {}x{} fmt={})",
				interval, s.statsSkipped, s.statsSkipped + s.statsRefreshed, kStatsFrames,
				s.lastRT, s.lastSlice, s.cachedDesc.Width, s.cachedDesc.Height, static_cast<uint32_t>(s.cachedDesc.Format));
			s.statsPending = false;
		}

		const auto* player = RE::PlayerCharacter::GetSingleton();
		const RE::TESObjectCELL* cell = player ? player->GetParentCell() : nullptr;
		if (a_interiorCell || cell != s.lastCell || globals::state->IsMainOrLoadingMenuOpen()) {
			s.lastCell = cell;
			s.warmupUntilFrame = frame + kWarmupFrames;
			s.valid = false;
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}

		if (FrameBefore(frame, s.warmupUntilFrame) || !s.valid || !s.cachedTex || s.cachedLight != a_light) {
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}

		auto& lightData = a_light->GetRuntimeData();
		if (lightData.shadowmapDescriptors.size() < 2) {
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}

		if ((frame - s.lastRefreshFrame) >= static_cast<uint32_t>(interval)) {
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}

		const auto& dirData = a_light->GetShadowDirectionalLightRuntimeData();
		const auto sunDelta = dirData.sunVector - s.cachedSunVec;
		if (sunDelta.SqrLength() > kMaxSunDeltaSq ||
			std::abs(dirData.startSplitDistances[1] - s.cachedStartSplit1) > kMaxSplitDelta ||
			std::abs(dirData.endSplitDistances[1] - s.cachedEndSplit1) > kMaxSplitDelta) {
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}

		RE::NiPoint3 camPos{}, camForward{};
		if (!GetCameraPose(camPos, camForward)) {
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}
		if ((camPos - s.cachedCamPos).SqrLength() > kMaxMoveSq || camForward.Dot(s.cachedCamForward) < kMinForwardDot) {
			s.skipFrame = 0xFFFFFFFFu;
			return;
		}

		// All validity checks passed: reject Cascade 1 culling at scene root via impossible Plane 0.
		auto& desc1 = lightData.shadowmapDescriptors[1];
		light = a_light;
		savedPlanes = desc1.clipPlanes;
		desc1.clipPlanes.cullingPlanes[0].normal = { 0.0f, 0.0f, 0.0f };
		desc1.clipPlanes.cullingPlanes[0].constant = 1e9f;
		desc1.clipPlanes.activePlanes = RE::NiFrustumPlanes::ActivePlane::kNear;
		s.skipFrame = frame;
		clipped = true;
	}

	CullingGuard::~CullingGuard()
	{
		if (clipped && light && !REL::Module::IsVR()) {
			auto& lightData = light->GetRuntimeData();
			if (lightData.shadowmapDescriptors.size() >= 2) {
				lightData.shadowmapDescriptors[1].clipPlanes = savedPlanes;
			}
		}
	}
}

void ShadowmapRasterizerFix::Install()
{
	// This function is called once per cascade to begin the updating and rendering process
	stl::write_thunk_call<BSShadowDirectionalLight_RenderShadowmaps_RenderCascade>(REL::RelocationID(101495, 108489).address() + REL::Relocate(0xC6, 0xC6));

	gRasterStates = reinterpret_cast<RasterStateArray*>(REL::RelocationID(524748, 411363).address());

	numCascades = static_cast<uint>(Util::GetGameSettingValue<std::int32_t>("iNumSplits:Display", Settings.at("iNumSplits:Display")));
}

void ShadowmapRasterizerFix::BSShadowDirectionalLight_RenderShadowmaps_RenderCascade::thunk(RE::BSShadowDirectionalLight* light, void* arg1, void* arg2, uint32_t flags)
{
	static uint cascade = 0;

	static bool initialized = false;
	if (!initialized) {
		//Backup
		if (cascade == 0) {
			std::memcpy(backupGameRasterStates, *gRasterStates, sizeof(RasterStateArray));
			numCascades = std::min(numCascades, maxCascades);
		}

		//Clone
		CloneRasterStates(gRasterStates, cascade);

		initialized = cascade == numCascades - 1;
	}

	//Emplace
	std::memcpy(*gRasterStates, shadowmapRasterStates[cascade], sizeof(RasterStateArray));

	const bool manageCascade1 = (cascade == 1) && !REL::Module::IsVR() && light && (FarShadowCache::GetInterval() > 1) &&
	                            light->GetRuntimeData().shadowmapDescriptors.size() >= 2;
	const uint32_t frame = manageCascade1 ? globals::state->frameCount : 0u;
	if (manageCascade1 && FarShadowCache::IsSkipFrame(light, frame)) {
		// CullingGuard rejected all objects for Cascade 1, so the accumulator has 0 passes.
		// Run func with clearRenderTarget=false to keep engine accumulator state clean, then restore cached slice + matrix.
		auto& desc1 = light->GetRuntimeData().shadowmapDescriptors[1];
		const bool prevClear = desc1.clearRenderTarget;
		desc1.clearRenderTarget = false;
		func(light, arg1, arg2, flags);
		desc1.clearRenderTarget = prevClear;
		FarShadowCache::RestoreCascade1(light);
	} else {
		func(light, arg1, arg2, flags);
		if (manageCascade1)
			FarShadowCache::CaptureCascade1(light, frame);
	}

	//Restore
	if (cascade == numCascades - 1)
		std::memcpy(*gRasterStates, backupGameRasterStates, sizeof(RasterStateArray));

	cascade = ++cascade < numCascades ? cascade : 0;
}

void ShadowmapRasterizerFix::GetUpdatedRasterDesc(D3D11_RASTERIZER_DESC& outputDesc, ShadowMapRasterizerDescriptor shadowmapDesc)
{
	outputDesc.DepthBias = shadowmapDesc.rasterDepthBias;
	outputDesc.DepthBiasClamp = shadowmapDesc.rasterDepthBiasClamp;
	outputDesc.SlopeScaledDepthBias = shadowmapDesc.rasterSlopeScaleBias;
}

// Since state objects are shared globally across the pipeline we make duplicate arrays that cover the same range of states the game does
void ShadowmapRasterizerFix::CloneRasterStates(RasterStateArray* inputArray, int cascade)
{
	for (int fill = 0; fill < 2; fill++) {
		for (int cull = 0; cull < 3; cull++) {
			for (int depth = 0; depth < 12; depth++) {
				for (int scissor = 0; scissor < 2; scissor++) {
					if (auto* gRasterizer = (*inputArray)[fill][cull][depth][scissor]) {
						D3D11_RASTERIZER_DESC desc{};
						gRasterizer->GetDesc(&desc);

						GetUpdatedRasterDesc(desc, cascadeDescriptors[cascade]);

						DX::ThrowIfFailed(globals::d3d::device->CreateRasterizerState(&desc, &shadowmapRasterStates[cascade][fill][cull][depth][scissor]));
					}
				}
			}
		}
	}
}