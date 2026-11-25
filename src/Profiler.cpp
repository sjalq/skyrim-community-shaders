#include "Profiler.h"

#include <algorithm>
#include <unordered_map>

float Profiler::RollingHistory::GetAverage() const
{
	if (count == 0)
		return lastMs;
	float sum = 0.0f;
	for (uint32_t i = 0; i < count; i++)
		sum += history[i];
	return sum / static_cast<float>(count);
}

float Profiler::RollingHistory::GetPercentile(float p) const
{
	if (count == 0)
		return lastMs;

	thread_local std::vector<float> sorted;
	sorted.resize(count);
	for (uint32_t i = 0; i < count; i++)
		sorted[i] = history[i];
	std::sort(sorted.begin(), sorted.end());

	float idx = (p / 100.0f) * static_cast<float>(count - 1);
	uint32_t lo = static_cast<uint32_t>(idx);
	uint32_t hi = std::min(lo + 1, count - 1);
	float frac = idx - static_cast<float>(lo);
	return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

void Profiler::RollingHistory::GetPercentiles(float pA, float pB, float& outA, float& outB) const
{
	if (count == 0) {
		outA = lastMs;
		outB = lastMs;
		return;
	}

	thread_local std::vector<float> sorted;
	sorted.assign(history, history + count);
	std::sort(sorted.begin(), sorted.end());

	const auto at = [this](float p) {
		float idx = (p / 100.0f) * static_cast<float>(count - 1);
		uint32_t lo = static_cast<uint32_t>(idx);
		uint32_t hi = std::min(lo + 1, count - 1);
		float frac = idx - static_cast<float>(lo);
		return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
	};
	outA = at(pA);
	outB = at(pB);
}

void Profiler::Initialize(ID3D11Device* device, ID3D11DeviceContext* a_context)
{
	Release();

	context = a_context;

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	cpuTicksToMs = 1000.0 / static_cast<double>(freq.QuadPart);

	for (auto& frame : frames) {
		D3D11_QUERY_DESC disjointDesc{};
		disjointDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
		device->CreateQuery(&disjointDesc, frame.disjoint.put());

		frame.timers.resize(kMaxTimers);
		for (auto& timer : frame.timers) {
			D3D11_QUERY_DESC tsDesc{};
			tsDesc.Query = D3D11_QUERY_TIMESTAMP;
			device->CreateQuery(&tsDesc, timer.begin.put());
			device->CreateQuery(&tsDesc, timer.end.put());
		}
		frame.activeCount = 0;
		frame.inFlight = false;
	}

	writeFrame = 0;
	readFrame = 0;
	framesSinceInit = 0;
	initialized = true;
}

void Profiler::Release()
{
	for (auto& frame : frames) {
		frame.disjoint = nullptr;
		frame.timers.clear();
		frame.activeCount = 0;
		frame.inFlight = false;
	}
	results.clear();
	resultsDirty = false;
	knownTimers.clear();
	knownTimerIndex.clear();
	collectedFrames = 0;
	totalTimeMs = 0.0f;
	cpuTotalTimeMs = 0.0f;
	initialized = false;
	context = nullptr;
}

void Profiler::BeginFrame()
{
	if (!initialized || !context || frameActive)
		return;

	CollectResults();

	auto& frame = frames[writeFrame];
	frame.activeCount = 0;
	frame.inFlight = true;
	frameActive = true;
	context->Begin(frame.disjoint.get());
}

void Profiler::BeginPass(const std::string& name)
{
	if (!initialized || !context)
		return;

	if (!frameActive)
		BeginFrame();

	auto& frame = frames[writeFrame];
	if (frame.activeCount >= kMaxTimers)
		return;

	auto& timer = frame.timers[frame.activeCount];
	timer.name = name;
	context->End(timer.begin.get());
	QueryPerformanceCounter(&timer.cpuBegin);

	if (beginPerfEvent)
		beginPerfEvent(name);
}

void Profiler::EndPass()
{
	if (!initialized || !context || !frameActive)
		return;

	auto& frame = frames[writeFrame];
	if (frame.activeCount >= kMaxTimers)
		return;

	auto& timer = frame.timers[frame.activeCount];

	LARGE_INTEGER cpuEnd;
	QueryPerformanceCounter(&cpuEnd);
	timer.cpuMs = static_cast<float>(static_cast<double>(cpuEnd.QuadPart - timer.cpuBegin.QuadPart) * cpuTicksToMs);

	context->End(timer.end.get());
	frame.activeCount++;

	if (endPerfEvent)
		endPerfEvent({});
}

void Profiler::EndFrame()
{
	if (!initialized || !context || !frameActive)
		return;

	frameActive = false;
	context->End(frames[writeFrame].disjoint.get());
	writeFrame = (writeFrame + 1) % kFrameLatency;
	framesSinceInit++;
}

void Profiler::CollectResults()
{
	if (framesSinceInit < kFrameLatency)
		return;

	readFrame = writeFrame;
	auto& frame = frames[readFrame];
	if (!frame.inFlight)
		return;

	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjointData{};
	HRESULT hr = context->GetData(frame.disjoint.get(), &disjointData, sizeof(disjointData), D3D11_ASYNC_GETDATA_DONOTFLUSH);
	if (hr != S_OK)
		return;

	frame.inFlight = false;
	collectedFrames++;

	float activeTotalMs = 0.0f;
	float activeCpuTotalMs = 0.0f;

	if (!disjointData.Disjoint) {
		double ticksToMs = 1000.0 / static_cast<double>(disjointData.Frequency);

		for (uint32_t i = 0; i < frame.activeCount; i++) {
			auto& timer = frame.timers[i];
			UINT64 tsBegin = 0, tsEnd = 0;

			if (context->GetData(timer.begin.get(), &tsBegin, sizeof(tsBegin), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK)
				continue;
			if (context->GetData(timer.end.get(), &tsEnd, sizeof(tsEnd), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK)
				continue;

			float ms = static_cast<float>(static_cast<double>(tsEnd - tsBegin) * ticksToMs);
			activeTotalMs += ms;
			activeCpuTotalMs += timer.cpuMs;

			auto [it, inserted] = knownTimerIndex.try_emplace(timer.name, knownTimers.size());
			if (inserted) {
				KnownTimer kt;
				kt.name = timer.name;
				knownTimers.push_back(std::move(kt));
			}
			auto& known = knownTimers[it->second];
			// First sample of this pass in this collected frame: restart the per-frame sums.
			// (Same aggregation the old per-frame activeTimers map did for repeated passes.)
			if (known.lastSampleFrame != collectedFrames) {
				known.frameGpuMs = 0.0f;
				known.frameCpuMs = 0.0f;
			}
			known.frameGpuMs += ms;
			known.frameCpuMs += timer.cpuMs;
			known.gpu.PushSample(ms);
			known.cpu.PushSample(timer.cpuMs);
			known.lastSampleFrame = collectedFrames;
		}
	}

	RetireStaleTimers();

	totalTimeMs = activeTotalMs;
	cpuTotalTimeMs = activeCpuTotalMs;

	// Per-pass results and their percentiles are only read by the profiler UI and the
	// devbench bridge; build them on demand in GetResults() instead of every frame.
	resultsDirty = true;
}

void Profiler::RebuildResults() const
{
	resultsDirty = false;
	results.clear();
	results.reserve(knownTimers.size());
	for (const auto& known : knownTimers) {
		TimerResult result;
		result.name = known.name;
		if (known.lastSampleFrame == collectedFrames) {
			result.gpuTimeMs = known.frameGpuMs;
			result.cpuTimeMs = known.frameCpuMs;
		} else {
			result.gpuTimeMs = known.gpu.lastMs;
			result.cpuTimeMs = known.cpu.lastMs;
		}
		result.avgMs = known.gpu.GetAverage();
		known.gpu.GetPercentiles(95.0f, 99.0f, result.p95Ms, result.p99Ms);
		result.cpuAvgMs = known.cpu.GetAverage();
		known.cpu.GetPercentiles(95.0f, 99.0f, result.cpuP95Ms, result.cpuP99Ms);
		result.valid = true;
		result.historyBuffer = known.gpu.history;
		result.historyHead = known.gpu.head;
		result.historyCount = known.gpu.count;
		results.push_back(std::move(result));
	}
}
void Profiler::RetireStaleTimers()
{
	const size_t before = knownTimers.size();
	std::erase_if(knownTimers, [this](const KnownTimer& known) {
		return collectedFrames - known.lastSampleFrame >= kTimerRetireFrames;
	});
	if (knownTimers.size() != before)
		RebuildTimerIndex();
}

void Profiler::RebuildTimerIndex()
{
	knownTimerIndex.clear();
	for (size_t i = 0; i < knownTimers.size(); i++)
		knownTimerIndex[knownTimers[i].name] = i;
}
