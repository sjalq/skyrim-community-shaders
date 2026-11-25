// DLSS Frame Generation UI composition.
//
// DLSS-G does not composite the UI itself (unlike FSR frame generation), so with DLSS-G the UI
// stays in its own premultiplied buffer and is blended onto the scene here, right before present.
// The scene is read from the HUD-less copy that is also handed to DLSS-G, and the result is the
// final backbuffer: Final.rgb = UI.rgb + (1 - UI.a) * HudLess.rgb, the blend DLSS-G expects for
// kBufferTypeUIColorAndAlpha and the one FidelityFX applies with premultiplied UI.

Texture2D<float4> HudLessTexture : register(t0);
Texture2D<float4> UITexture : register(t1);

RWTexture2D<float4> OutputTexture : register(u0);

[numthreads(8, 8, 1)] void main(uint3 dispatchID : SV_DispatchThreadID) {
	uint width, height;
	OutputTexture.GetDimensions(width, height);
	if (dispatchID.x >= width || dispatchID.y >= height)
		return;

	float4 scene = HudLessTexture[dispatchID.xy];
	float4 ui = UITexture[dispatchID.xy];

	OutputTexture[dispatchID.xy] = float4(ui.rgb + (1.0 - ui.a) * scene.rgb, scene.a);
}
