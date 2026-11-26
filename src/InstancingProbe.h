#pragma once

#include <d3d11.h>

/**
 * @brief Instancing slice S0 (plan/instancing-design.txt, sections 1, 2 and 5): counters only, no draw changes.
 *
 * When the Advanced setting "Instancing Probe" is on, every BSBatchRenderer::RenderPassImmediately call inside
 * a measured window (sun shadow cascades 0-2 and 3+, opaque depth prepass, G-buffer main view, reflection
 * cubemap) is classified against the design's Eligible() rules (4.1) and grouped by the design's bucket key
 * (2.3) for the RenderBatches call it belongs to. When that RenderBatches call returns, the buckets are
 * counted: distinct keys, bucket-size histogram (1, 2-3, 4-7, 8-15, 16+), distinct geometries vs renderer
 * data vs materials, and whether each gathered pass still points at the same geometry and shader property
 * (pass lifetime, design 2.2). A DrawIndexed detour counts draws per pass, compares VertexShader::id with
 * the cache descriptor of the current technique (design 3.2) and, while the constant buffer ring is active,
 * compares every bucket member's VS/PS PerGeometry (b2) registers with its bucket's first member (design 3.3,
 * TreeParams variance for TREE_ANIM). Every pass is still drawn by the engine exactly as before.
 *
 * Every 600 frames while enabled, one "[InstancingProbe]" line per window (per-frame averages) and one frame
 * line go to the log. Off, each hook costs one branch.
 */
namespace InstancingProbe
{
	/** @brief Requests the probe on or off (live toggle); applied at the next frame boundary. */
	void SetEnabled(bool a_enabled);

	/** @brief The requested state, as saved in the settings. */
	[[nodiscard]] bool IsEnabled();

	/** @brief Installs the engine hooks. Call at PostPostLoad after every feature installed its own, so ours are outermost. */
	void Install();

	/** @brief Detours ID3D11DeviceContext::DrawIndexed (vtable slot 12). Call with the immediate context. */
	void InstallD3DHooks(ID3D11DeviceContext* a_context);

	/** @brief Latches the toggle, advances the frame counters and logs every 600 frames. Call once per frame at Present. */
	void OnFrameBoundary();
}
