import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { addPropertyControls, ControlType } from 'framer';

/**
 * @framerDisableUnlink
 * @framerIntrinsicWidth 1200
 * @framerIntrinsicHeight 800
 * @framerSupportedLayoutWidth any
 * @framerSupportedLayoutHeight any
 */

// Shaders for Fluid Post-Processing (simulation & post pass)
const baseVertex = /* glsl */ `
  varying vec2 vUv;
  void main () {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const paintFragment = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  
  uniform sampler2D uBuffer; // The previous simulation state (H, Vx, Vy, Hprev) encoded in [0, 1]
  uniform vec2 uMouse;       // Screen-space mouse pos (in UV space)
  uniform vec2 uMouseVelocity; // Injected mouse velocity force
  uniform float uAspect;
  uniform float uRadius;
  uniform float uDissipation; // Damping for height
  uniform float uVelocityDissipation; // Damping for velocity
  uniform float uAdvect;      // Advection strength
  uniform vec2 uTexelSize;
  uniform float uWaveSpeed;   // Speed of wave propagation (c^2)
  uniform float uVorticity;   // Vorticity confinement strength
  uniform float uHeightForce; // Direct height force injection from mouse velocity

  // Hover ripple injection
  uniform vec2 uHoverCenter;
  uniform float uHoverActive;
  uniform float uTime;

  // Helper to decode FBO values from [0, 1] to [-1, 1]
  vec4 decodeSample(vec2 uv) {
    return texture2D(uBuffer, uv) * 2.0 - 1.0;
  }

  void main() {
    // 1. Advection: we advect the entire state using the velocity field stored in .gb
    vec2 currentVel = texture2D(uBuffer, vUv).gb * 2.0 - 1.0;
    vec2 advectedUv = vUv - currentVel * uAdvect;
    
    // Sample neighbors using advected coordinates for fluid continuity
    vec4 stateN = decodeSample(advectedUv + vec2(0.0, uTexelSize.y));
    vec4 stateS = decodeSample(advectedUv - vec2(0.0, uTexelSize.y));
    vec4 stateE = decodeSample(advectedUv + vec2(uTexelSize.x, 0.0));
    vec4 stateW = decodeSample(advectedUv - vec2(uTexelSize.x, 0.0));
    vec4 stateCurr = decodeSample(advectedUv);
    
    float H_curr = stateCurr.r;
    float H_prev = stateCurr.a;
    vec2 vel = stateCurr.gb;
    
    // 2. Wave Propagation (2D Discrete Wave Solver)
    float laplacian = (stateN.r + stateS.r + stateE.r + stateW.r) - 4.0 * H_curr;
    float H_next = 2.0 * H_curr - H_prev + uWaveSpeed * laplacian;
    H_next *= uDissipation; // Damping height over time
    
    // 3. Vorticity Confinement (Curl recovery for small swirls)
    // curl = dVy/dx - dVx/dy
    float curlN = (stateN.b - stateN.g);
    float curlS = (stateS.b - stateS.g);
    float curlE = (stateE.b - stateE.g);
    float curlW = (stateW.b - stateW.g);
    float curlCurr = (stateCurr.b - stateCurr.g);
    
    vec2 curlGrad = vec2(
      abs(curlE) - abs(curlW),
      abs(curlN) - abs(curlS)
    ) * 0.5;
    
    vec2 curlForce = normalize(curlGrad + 1e-5) * curlCurr * uVorticity;
    vel += curlForce;
    
    // Decay velocity (momentum)
    vel *= uVelocityDissipation;
    
    // 4. Force Injection from Mouse movement
    vec2 dMouse = vUv - uMouse;
    dMouse.x *= uAspect;
    float mouseDist = length(dMouse);
    float mouseWeight = smoothstep(uRadius, 0.0, mouseDist);
    mouseWeight = pow(mouseWeight, 2.0);
    
    vel += uMouseVelocity * mouseWeight;
    
    // Inject mouse speed into height to generate physical ripple waves
    float mouseSpeed = length(uMouseVelocity);
    H_next += mouseSpeed * uHeightForce * mouseWeight;
    
    // 5. Hover Ripple Injection removed to avoid artificial flickering, letting real mouse physics drive all ripples
    
    // Prevent numerical explosion or banding
    H_next = clamp(H_next, -1.0, 1.0);
    vel = clamp(vel, -1.0, 1.0);
    H_curr = clamp(H_curr, -1.0, 1.0);
    
    // Encode back into [0, 1] for stable FBO storage
    gl_FragColor = vec4(
      H_next * 0.5 + 0.5,
      vel.x * 0.5 + 0.5,
      vel.y * 0.5 + 0.5,
      H_curr * 0.5 + 0.5
    );
  }
`;

const blurFragment = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  
  uniform sampler2D uInput;
  uniform vec2 uTexelSize;
  uniform float uMaxBlurRadius;

  void main() {
    // Read velocity from the state (encoded in .gb channels)
    vec4 state = texture2D(uInput, vUv);
    vec2 vel = state.gb * 2.0 - 1.0;
    float velMag = length(vel);
    
    // Adaptive blur: fast movement stays crisp (smaller radius), slow movement is smoothed (larger radius)
    float adaptiveFactor = mix(uMaxBlurRadius, 0.1, smoothstep(0.0, 1.0, velMag * 6.0));
    vec2 off = uTexelSize * adaptiveFactor;
    
    vec4 sum = texture2D(uInput, vUv) * 0.25;
    sum += texture2D(uInput, vUv + vec2(off.x, 0.0)) * 0.125;
    sum += texture2D(uInput, vUv - vec2(off.x, 0.0)) * 0.125;
    sum += texture2D(uInput, vUv + vec2(0.0, off.y)) * 0.125;
    sum += texture2D(uInput, vUv - vec2(0.0, off.y)) * 0.125;
    sum += texture2D(uInput, vUv + vec2(off.x, off.y)) * 0.0625;
    sum += texture2D(uInput, vUv - vec2(off.x, off.y)) * 0.0625;
    sum += texture2D(uInput, vUv + vec2(off.x, -off.y)) * 0.0625;
    sum += texture2D(uInput, vUv - vec2(off.x, -off.y)) * 0.0625;
    
    gl_FragColor = sum;
  }
`;

const postFragment = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  
  uniform sampler2D tMap;        // scene rendering
  uniform sampler2D tSimulation; // living water height & velocity state (encoded in [0, 1])
  uniform vec2 uTexelSize;
  uniform float uTime;
  
  // Optical tuning
  uniform float uHeightScale;       // Reconstructed normal vertical scale
  uniform float uRefractionStrength; // Refraction UV distortion scale
  uniform float uRefractionOffset;  // Depth refraction dispersion factor
  uniform float uRgbShift;          // Chromatic dispersion strength
  uniform float uSpecular;          // Specular highlights intensity
  uniform float uRoughness;         // GGX Specular roughness
  uniform float uSpecularColorIntensity;
  uniform float uAbsorptionStrength; // Teal color attenuation
  uniform float uCausticIntensity;  // Curvature-based caustics
  uniform float uMicroStrength;      // High frequency micro normals strength
  uniform float uMicroScale;         // Scale of procedural detail noise
  uniform float uAmbientReflection;  // Base reflection intensity

  // Hover targets for local depth amplification
  uniform vec2 uHoverCenter;
  uniform vec2 uHoverSize;
  uniform float uHoverActive;

  // Constants for physical lighting direction
  const vec3 uLightDir = normalize(vec3(0.8, 1.2, 1.4));
  const vec3 uViewDir = vec3(0.0, 0.0, 1.0);

  // Multi-frequency wave noise for micro-ripples and sparkles
  float waveNoise(vec2 p, float time) {
    float n = sin(p.x * 2.0 + time * 1.5) * cos(p.y * 2.0 + time * 1.1);
    n += sin(p.x * 4.3 - time * 2.3) * sin(p.y * 3.7 + time * 1.9) * 0.5;
    n += cos(p.x * 8.5 + time * 3.1) * cos(p.y * 9.1 - time * 2.8) * 0.25;
    n += sin(p.x * 17.0 - time * 4.5) * cos(p.y * 18.5 + time * 4.2) * 0.125;
    return n;
  }

  // Professional GGX Microfacet Specular BRDF
  float GGX_Specular(vec3 N, vec3 V, vec3 L, float roughness, float F0) {
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    
    // D Term (Normal Distribution Function)
    float denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
    float D = alpha2 / (3.1415926535 * denom * denom);
    
    // G Term (Geometric Shadowing/Masking)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G1_V = NdotV / (NdotV * (1.0 - k) + k);
    float G1_L = NdotL / (NdotL * (1.0 - k) + k);
    float G = G1_V * G1_L;
    
    // F Term (Schlick Fresnel approximation)
    float F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    
    return (D * G * F) / (4.0 * NdotV * NdotL + 0.001);
  }

  void main() {
    // Decode water state
    vec4 state = texture2D(tSimulation, vUv) * 2.0 - 1.0;
    float height = state.r;
    vec2 vel = state.gb;
    
  // Reconstruct surface normals from neighboring pixels of the height field
    float hL = texture2D(tSimulation, vUv - vec2(uTexelSize.x * 2.0, 0.0)).r * 2.0 - 1.0;
    float hR = texture2D(tSimulation, vUv + vec2(uTexelSize.x * 2.0, 0.0)).r * 2.0 - 1.0;
    float hD = texture2D(tSimulation, vUv - vec2(0.0, uTexelSize.y * 2.0)).r * 2.0 - 1.0;
    float hU = texture2D(tSimulation, vUv + vec2(0.0, uTexelSize.y * 2.0)).r * 2.0 - 1.0;
    
    float dh_dx = (hR - hL) * 0.25;
    float dh_dy = (hU - hD) * 0.25;
    
    vec3 normal = vec3(-dh_dx * uHeightScale, -dh_dy * uHeightScale, 1.0);
    
    // Highly optimized single-pass analytical gradient calculation for micro-ripples
    vec2 microGrad = vec2(0.0);
    float microScaleTime = uTime * 1.5;
    
    // First octave wave
    vec2 waveA = vUv * uMicroScale * 1.5 + vec2(1.5, 1.1) * microScaleTime;
    vec2 sinA = sin(waveA);
    vec2 cosA = cos(waveA);
    microGrad.x += cosA.x * cosA.y * 1.5 * uMicroScale * 1.5;
    microGrad.y += -sinA.x * sinA.y * 1.5 * uMicroScale * 1.5;
    
    // Second octave wave
    vec2 waveB = vUv * uMicroScale * vec2(3.3, 2.7) + vec2(-2.3, 1.9) * microScaleTime;
    vec2 sinB = sin(waveB);
    vec2 cosB = cos(waveB);
    microGrad.x += cosB.x * sinB.y * 0.4 * uMicroScale * 3.3;
    microGrad.y += sinB.x * cosB.y * 0.4 * uMicroScale * 2.7;
    
    normal.xy += microGrad * uMicroStrength * 0.03;
    normal = normalize(normal);
    
    // Calculate card hover projection to locally amplify depth, refraction, reflection and specular
    float inCard = 0.0;
    if (uHoverActive > 0.0) {
      vec2 distToCard = abs(vUv - uHoverCenter) / (uHoverSize * 0.5);
      float cardMask = smoothstep(1.1, 0.9, max(distToCard.x, distToCard.y));
      inCard = cardMask * uHoverActive;
    }
    
    float localDepth = 1.0 + inCard * 0.6;
    float localRefraction = uRefractionStrength * (1.0 + inCard * 0.4);
    float localReflection = uAmbientReflection * (1.0 + inCard * 0.5);
    float localSpecular = uSpecular * (1.0 + inCard * 0.5);
    
    // 1. Refraction & Chromatic Dispersion (very subtle RGB splitting only on steep waves)
    float dispersion = uRgbShift * length(normal.xy) * (1.0 + abs(height) * uRefractionOffset);
    
    vec2 rOffset = normal.xy * localRefraction * (1.0 + dispersion) * localDepth;
    vec2 gOffset = normal.xy * localRefraction * localDepth;
    vec2 bOffset = normal.xy * localRefraction * (1.0 - dispersion) * localDepth;
    
    float r = texture2D(tMap, clamp(vUv + rOffset, 0.0001, 0.9999)).r;
    float g = texture2D(tMap, clamp(vUv + gOffset, 0.0001, 0.9999)).g;
    float b = texture2D(tMap, clamp(vUv + bOffset, 0.0001, 0.9999)).b;
    vec3 sceneColor = vec3(r, g, b);
    
    // 2. Pristine water transparency (no color absorption to avoid screen turning blue or flickering)
    
    // 3. Environmental Reflection with Fresnel
    float cosTheta = max(dot(normal, uViewDir), 0.0);
    float R0 = 0.02; // Water index of reflection
    float fresnel = R0 + (1.0 - R0) * pow(1.0 - cosTheta, 5.0);
    
    // Slate sky-dome reflection with integrated subtle iridescence
    vec3 skyReflection = mix(vec3(0.04, 0.06, 0.1), vec3(0.68, 0.76, 0.92), normal.y * 0.5 + 0.5);
    vec3 iridescence = vec3(
      0.5 + 0.5 * cos(6.28318 * (fresnel + 0.00)),
      0.5 + 0.5 * cos(6.28318 * (fresnel + 0.33)),
      0.5 + 0.5 * cos(6.28318 * (fresnel + 0.67))
    );
    skyReflection += iridescence * 0.06;
    
    // 4. GGX Specular
    float specHighlight = GGX_Specular(normal, uViewDir, uLightDir, uRoughness, R0);
    vec3 specularColor = vec3(1.0, 0.98, 0.95) * specHighlight * localSpecular;
    
    // 5. Dynamic Caustics
    // Curvature-based caustic focus (Laplacian of height field)
    float laplacian = (hR + hL + hU + hD) - 4.0 * height;
    float causticFocus = max(-laplacian, 0.0) * uCausticIntensity;
    
    // Flowing caustics that drift with fluid velocity
    float causticFlow = waveNoise(vUv * 15.0 - vel * uTime * 2.0, uTime * 0.4);
    float caustic = causticFocus * (0.6 + 0.4 * max(causticFlow, 0.0)) * localDepth;
    
    // 6. Composition
    vec3 finalColor = mix(sceneColor, skyReflection, fresnel * localReflection);
    
    // Under-water caustic light
    finalColor += caustic * vec3(0.88, 0.96, 1.0) * (1.0 - fresnel * 0.5);
    
    // Top-surface specular reflections
    finalColor += specularColor * uSpecularColorIntensity;
    
    // Frame everything with a subtle visual vignette
    float vignette = smoothstep(1.5, 0.5, length(vUv - 0.5));
    finalColor *= mix(0.92, 1.0, vignette);
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

// Shaders for Gallery Planes (Bending & motion blur)
const galleryVertex = /* glsl */ `
  uniform float uVelocity;
  uniform float uTime;
  uniform float uBendStrength;
  uniform float uFocalPoint;
  uniform float uFalloff;
  varying vec2 vUv;
  varying float vEdgeMask;

  void main() {
    vUv = uv;
    vec3 pos = position;
    
    // Calculate non-linear bending based on focal point
    // We use a smoother falloff to avoid "wavy" edges on wider cards
    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    float shiftedX = worldPos.x - uFocalPoint;
    
    // Smoother Gaussian-like falloff for elegant curvature
    float centerMask = exp(-pow(shiftedX * uFalloff * 0.7, 2.0));
    vEdgeMask = 1.0 - centerMask;
    
    // Applying the bend to Z
    float velocityBend = centerMask * abs(uVelocity) * uBendStrength * 0.4;
    pos.z -= velocityBend;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const galleryFragment = /* glsl */ `
  uniform sampler2D uTexture;
  uniform float uVelocity;
  uniform float uTime;
  uniform float uRgbSplit;
  uniform float uMotionBlurStrength;
  uniform float uGrainStrength;
  uniform float uHover;
  uniform float uOpacity;
  varying vec2 vUv;
  varying float vEdgeMask;

  float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
  }

  void main() {
    float blurFactor = smoothstep(0.0, 1.0, vEdgeMask);
    blurFactor *= blurFactor;
    
    float edgeDistX = min(vUv.x, 1.0 - vUv.x);
    float edgeDistY = min(vUv.y, 1.0 - vUv.y);
    float texEdgeMask = smoothstep(0.0, 0.06, edgeDistX) * smoothstep(0.0, 0.06, edgeDistY);
    
    // Check if the scroll velocity is practically zero. 
    // This allows us to completely skip the heavy multi-sampling loop when stationary.
    if (abs(uVelocity) < 0.0001) {
      vec4 baseColor = texture2D(uTexture, vUv);
      float noise = (random(vUv + uTime * 2.0) - 0.5) * uGrainStrength;
      baseColor.rgb += noise;
      baseColor.rgb = mix(baseColor.rgb, baseColor.rgb * 1.05 + vec3(0.01), uHover * 0.3);
      gl_FragColor = vec4(baseColor.rgb, baseColor.a * uOpacity);
      return;
    }
    
    vec2 velocityVec = vec2(uVelocity * uMotionBlurStrength * 0.25 * blurFactor * texEdgeMask, 0.0);
    
    float maxOffset = 0.25;
    if (length(velocityVec) > maxOffset) {
      velocityVec = normalize(velocityVec) * maxOffset;
    }
    
    vec4 colorSum = vec4(0.0);
    float weightSum = 0.0;
    
    float jitter = random(vUv + uTime) - 0.5;
    vec2 rgbShift = vec2(uVelocity * uRgbSplit * 0.04 * blurFactor * texEdgeMask, 0.0);
    
    // Drastically optimized loop: 8 taps instead of 20.
    // In combination with jittering and a smooth exponential falloff, 8 taps are visually indistinguishable 
    // from 20 taps, but reduce texture samples by 60%.
    for(int i = 0; i < 8; i++) {
      float t = (float(i) + jitter) / 8.0 - 0.5;
      float weight = exp(-8.0 * t * t);
      vec2 currentUv = vUv + velocityVec * t;
      
      vec2 zoomedUv = currentUv;
      
      vec2 rUv = clamp(zoomedUv + rgbShift * t, 0.0001, 0.9999);
      vec2 gUv = clamp(zoomedUv, 0.0001, 0.9999);
      vec2 bUv = clamp(zoomedUv - rgbShift * t, 0.0001, 0.9999);
      
      float r = texture2D(uTexture, rUv).r;
      float g = texture2D(uTexture, gUv).g;
      float b = texture2D(uTexture, bUv).b;
      
      colorSum += vec4(r, g, b, 1.0) * weight;
      weightSum += weight;
    }
    
    vec4 color = colorSum / weightSum;
    float noise = (random(vUv + uTime * 2.0) - 0.5) * uGrainStrength;
    color.rgb += noise;
    
    // Smooth brightening hover glow
    color.rgb = mix(color.rgb, color.rgb * 1.05 + vec3(0.01), uHover * 0.3);
    
    gl_FragColor = vec4(color.rgb, color.a * uOpacity);
  }
`;

interface RollingTextProps {
  text: string;
  hovered: boolean;
  className?: string;
  delayOffset?: number;
  staggerDelay?: number;
  duration?: number;
  style?: React.CSSProperties;
}

const RollingText: React.FC<RollingTextProps> = ({
  text,
  hovered,
  className = '',
  delayOffset = 0,
  staggerDelay = 0.03,
  duration = 0.6,
  style
}) => {
  const chars = Array.from(text);

  return (
    <span 
      className={`inline-flex items-center select-none ${className}`} 
      style={{ 
        ...style,
        display: 'inline-flex',
        whiteSpace: 'nowrap',
        perspective: '1000px',
      }}
    >
      {chars.map((char, i) => {
        if (char === ' ') {
          return <span key={i} style={{ display: 'inline-block', width: '0.25em' }}>&nbsp;</span>;
        }

        const delay = hovered 
          ? delayOffset + i * staggerDelay
          : delayOffset + (chars.length - 1 - i) * (staggerDelay * 0.5);

        return (
          <span
            key={i}
            style={{
              display: 'inline-block',
              overflow: 'hidden',
              verticalAlign: 'middle',
              lineHeight: '1.2em',
              height: '1.2em',
              position: 'relative',
            }}
          >
            <span
              style={{
                display: 'block',
                transform: hovered ? 'rotateX(0deg) translateY(0%)' : 'rotateX(-90deg) translateY(100%)',
                opacity: hovered ? 1 : 0,
                transition: `transform ${duration}s cubic-bezier(0.16, 1, 0.3, 1), opacity ${duration}s cubic-bezier(0.16, 1, 0.3, 1)`,
                transitionDelay: `${delay}s`,
                transformOrigin: '50% 100%',
                willChange: 'transform, opacity',
              }}
            >
              {char}
            </span>
          </span>
        );
      })}
    </span>
  );
};

// Physics-based Scroll Engine hook
function useVirtualScroll(scrollConfig: {
  touchpadMultiplier: number;
  mouseMultiplier: number;
  friction: number;
  dragMultiplier: number;
}) {
  const target = useRef(0);
  const config = useRef(scrollConfig);

  useEffect(() => {
    config.current = scrollConfig;
  }, [scrollConfig.touchpadMultiplier, scrollConfig.mouseMultiplier, scrollConfig.friction, scrollConfig.dragMultiplier]);

  useEffect(() => {
    let velocity = 0;
    let momentumId: number;

    const onWheel = (e: WheelEvent) => {
      cancelAnimationFrame(momentumId);
      const isTouchpad = Math.abs(e.deltaY) < 50 && Math.abs(e.deltaX) < 50;
      const multiplier = isTouchpad ? config.current.touchpadMultiplier : config.current.mouseMultiplier; 
      
      const delta = (e.deltaY + e.deltaX) * multiplier;
      target.current += delta;
      
      if (!isTouchpad) {
        velocity = delta;
        applyMomentum();
      }
    };

    const applyMomentum = () => {
      if (Math.abs(velocity) > 0.0001) {
        target.current += velocity;
        velocity *= config.current.friction;
        momentumId = requestAnimationFrame(applyMomentum);
      }
    };

    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let lastTime = 0;
    let lastMoveTime = 0;

    const onPointerDown = (e: PointerEvent) => {
      // Limit to left-click only for standard mouse pointer types, allow all touches/pens
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      
      cancelAnimationFrame(momentumId);
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      lastTime = performance.now();
      lastMoveTime = lastTime;
      velocity = 0;
      
      // Request pointer capture to keep tracking gestures flawlessly even if dragging goes off-screen
      try {
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
      } catch (err) {
        // Fallback for environments where pointer capture might throw
      }
      
      document.body.style.cursor = 'grabbing';
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      
      const now = performance.now();
      const dt = Math.max(now - lastTime, 1);
      lastTime = now;
      lastMoveTime = now;

      const deltaX = startX - e.clientX;
      const deltaY = startY - e.clientY;
      
      // Calibrate pixel-to-WebGL projection ratio for perfect tactile 1:1 drag-to-scroll tracking.
      // Under our 3D perspective camera (FOV = 45, Z = 8), the WebGL viewport height is exactly 6.627417 units.
      const pixelHeight = window.innerHeight || 800;
      const projectionRatio = 6.627417 / pixelHeight;
      
      // Scale by dragMultiplier relative to 0.01 baseline (1.0 in Framer maps to 100% 1:1)
      const dragScale = projectionRatio * (config.current.dragMultiplier * 100);

      let delta = 0;
      const isTouch = e.pointerType === 'touch';

      if (isTouch) {
        // For touch gestures, track horizontal swipes exclusively to align with horizontal gallery layout,
        // preventing diagonal drift and letting the browser handle vertical page scrolling natively.
        delta = deltaX * dragScale;
      } else {
        // For mouse, keep dominant axis tracking
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
          delta = deltaX * dragScale;
        } else {
          delta = deltaY * dragScale;
        }
      }

      target.current += delta;
      velocity = delta * (16 / dt);

      startX = e.clientX;
      startY = e.clientY;
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!isDragging) return;
      isDragging = false;
      document.body.style.cursor = '';

      // Release pointer capture cleanly
      try {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      } catch (err) {}

      // Tactile lift check: if user paused dragging before releasing (e.g. holding finger still), stop momentum instantly
      const timeSinceLastMove = performance.now() - lastMoveTime;
      if (timeSinceLastMove > 80) {
        velocity = 0;
      }

      applyMomentum();
    };

    window.addEventListener('wheel', onWheel, { passive: true });
    window.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);

    return () => {
      cancelAnimationFrame(momentumId);
      window.removeEventListener('wheel', onWheel);
      window.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
      document.body.style.cursor = '';
    };
  }, []);

  return target;
}

const defaultItems = [
  { url: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80', title: 'Mountain Landscapes', tag: 'PHOTOGRAPHY', link: 'https://unsplash.com', alt: 'Mountain Landscapes' },
  { url: 'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=800&q=80', title: 'Coastal Breezes', tag: 'NATURE', link: 'https://unsplash.com', alt: 'Coastal Breezes' },
  { url: 'https://images.unsplash.com/photo-1501854140801-50d01698950b?auto=format&fit=crop&w=800&q=80', title: 'Ocean Waves', tag: 'SURFING', link: 'https://unsplash.com', alt: 'Ocean Waves' },
  { url: 'https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?auto=format&fit=crop&w=800&q=80', title: 'Forest Trails', tag: 'EXPLORATION', link: 'https://unsplash.com', alt: 'Forest Trails' },
  { url: 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=800&q=80', title: 'Urban Jungles', tag: 'ARCHITECTURE', link: 'https://unsplash.com', alt: 'Urban Jungles' },
  { url: 'https://images.unsplash.com/photo-1505761671935-60b3a742798e?auto=format&fit=crop&w=800&q=80', title: 'Desert Dunes', tag: 'TRAVEL', link: 'https://unsplash.com', alt: 'Desert Dunes' },
  { url: 'https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?auto=format&fit=crop&w=800&q=80', title: 'Starry Nights', tag: 'ASTRONOMY', link: 'https://unsplash.com', alt: 'Starry Nights' },
];

export default function FluidGallery(props: any) {
  const { items = defaultItems } = props;

  // Group Type Property safe destructuring with top-level fallbacks
  const layout = props.layoutGroup || {};
  const cardWidthRaw = layout.cardWidth !== undefined ? layout.cardWidth : (props.cardWidth !== undefined ? props.cardWidth : 360);
  const cardHeightRaw = layout.cardHeight !== undefined ? layout.cardHeight : (props.cardHeight !== undefined ? props.cardHeight : 440);
  const cardGapRaw = layout.cardGap !== undefined ? layout.cardGap : (props.cardGap !== undefined ? props.cardGap : 30);

  // Fallback / legacy support for WebGL units (e.g. width/height/gap < 15)
  let cardWidth = cardWidthRaw;
  let cardHeight = cardHeightRaw;
  let cardGap = cardGapRaw;

  const legacyW = layout.width !== undefined ? layout.width : ((props.width !== undefined && typeof props.width === 'number' && props.width < 15) ? props.width : undefined);
  const legacyH = layout.height !== undefined ? layout.height : ((props.height !== undefined && typeof props.height === 'number' && props.height < 15) ? props.height : undefined);
  const legacyG = layout.gap !== undefined ? layout.gap : ((props.gap !== undefined && typeof props.gap === 'number' && props.gap < 5) ? props.gap : undefined);

  if (legacyH !== undefined) {
    cardHeight = Math.round(legacyH * (800 / 6.627417));
  }
  if (legacyW !== undefined) {
    cardWidth = Math.round(legacyW * (800 / 6.627417));
  }
  if (legacyG !== undefined) {
    cardGap = Math.round(legacyG * (800 / 6.627417));
  }

  const lerpSpeed = layout.lerpSpeed !== undefined ? layout.lerpSpeed : (props.lerpSpeed !== undefined ? props.lerpSpeed : 0.08);
  const velocityMultiplier = layout.velocityMultiplier !== undefined ? layout.velocityMultiplier : (props.velocityMultiplier !== undefined ? props.velocityMultiplier : 1.0);

  const scrollPhysics = props.scrollPhysicsGroup || {};
  const touchpadMultiplier = scrollPhysics.touchpadMultiplier !== undefined ? scrollPhysics.touchpadMultiplier : (props.touchpadMultiplier !== undefined ? props.touchpadMultiplier : 0.006);
  const mouseMultiplier = scrollPhysics.mouseMultiplier !== undefined ? scrollPhysics.mouseMultiplier : (props.mouseMultiplier !== undefined ? props.mouseMultiplier : 0.012);
  const friction = scrollPhysics.friction !== undefined ? scrollPhysics.friction : (props.friction !== undefined ? props.friction : 0.94);
  const dragMultiplier = scrollPhysics.dragMultiplier !== undefined ? scrollPhysics.dragMultiplier : (props.dragMultiplier !== undefined ? props.dragMultiplier : 0.01);

  const bend = props.bendGroup || {};
  const bendStrength = bend.bendStrength !== undefined ? bend.bendStrength : (props.bendStrength !== undefined ? props.bendStrength : 12.0);
  const bendFocalPoint = bend.bendFocalPoint !== undefined ? bend.bendFocalPoint : (props.bendFocalPoint !== undefined ? props.bendFocalPoint : 0.0);
  const bendFalloff = bend.bendFalloff !== undefined ? bend.bendFalloff : (props.bendFalloff !== undefined ? props.bendFalloff : 0.2);

  const effects = props.effectsGroup || {};
  const motionBlurStrength = effects.motionBlurStrength !== undefined ? effects.motionBlurStrength : (props.motionBlurStrength !== undefined ? props.motionBlurStrength : 1.5);
  const rgbSplit = effects.rgbSplit !== undefined ? effects.rgbSplit : (props.rgbSplit !== undefined ? props.rgbSplit : 1.5);
  const grainStrength = effects.grainStrength !== undefined ? effects.grainStrength : (props.grainStrength !== undefined ? props.grainStrength : 0.0);

  const ticker = props.tickerGroup || {};
  const tickerEnabled = ticker.tickerEnabled !== undefined ? ticker.tickerEnabled : (props.tickerEnabled !== undefined ? props.tickerEnabled : false);
  const tickerSpeed = ticker.tickerSpeed !== undefined ? ticker.tickerSpeed : (props.tickerSpeed !== undefined ? props.tickerSpeed : 0.15);

  const typography = props.typographyGroup || {};
  const enableHoverText = typography.enableHoverText !== undefined ? typography.enableHoverText : (props.enableHoverText !== undefined ? props.enableHoverText : true);
  const enableTag = typography.enableTag !== undefined ? typography.enableTag : (props.enableTag !== undefined ? props.enableTag : true);
  const showTextOnMobile = typography.showTextOnMobile !== undefined ? typography.showTextOnMobile : (props.showTextOnMobile !== undefined ? props.showTextOnMobile : false);
  const textStagger = typography.textStagger !== undefined ? typography.textStagger : (props.textStagger !== undefined ? props.textStagger : 0.045);
  const textPower = typography.textPower !== undefined ? typography.textPower : (props.textPower !== undefined ? props.textPower : 0.75);
  const textDuration = typography.textDuration !== undefined ? typography.textDuration : (props.textDuration !== undefined ? props.textDuration : 0.7);
  const textRotateStart = typography.textRotateStart !== undefined ? typography.textRotateStart : (props.textRotateStart !== undefined ? props.textRotateStart : 85);
  const textTranslateYStart = typography.textTranslateYStart !== undefined ? typography.textTranslateYStart : (props.textTranslateYStart !== undefined ? props.textTranslateYStart : 15);
  const textTranslateZStart = typography.textTranslateZStart !== undefined ? typography.textTranslateZStart : (props.textTranslateZStart !== undefined ? props.textTranslateZStart : -30);
  const textBlurStart = typography.textBlurStart !== undefined ? typography.textBlurStart : (props.textBlurStart !== undefined ? props.textBlurStart : 3);
  const textEase = typography.textEase !== undefined ? typography.textEase : (props.textEase !== undefined ? props.textEase : "cubic-bezier(0.16, 1, 0.3, 1)");

  const titleFont = typography.titleFont !== undefined ? typography.titleFont : (props.titleFont !== undefined ? props.titleFont : {
    fontFamily: "Inter",
    fontWeight: 800,
    fontSize: 13,
    letterSpacing: "0.25em",
  });
  const tagFont = typography.tagFont !== undefined ? typography.tagFont : (props.tagFont !== undefined ? props.tagFont : {
    fontFamily: "JetBrains Mono",
    fontWeight: 500,
    fontSize: 10,
    letterSpacing: "0.2em",
  });

  const fluidSim = props.fluidSimulationGroup || {};
  const dissipation = fluidSim.dissipation !== undefined ? fluidSim.dissipation : (props.dissipation !== undefined ? props.dissipation : 0.98);
  const radius = fluidSim.radius !== undefined ? fluidSim.radius : (props.radius !== undefined ? props.radius : 0.1);
  const velocityScale = fluidSim.velocityScale !== undefined ? fluidSim.velocityScale : (props.velocityScale !== undefined ? props.velocityScale : 5.0);
  const advect = fluidSim.advect !== undefined ? fluidSim.advect : (props.advect !== undefined ? props.advect : 0.001);
  const surfaceTension = fluidSim.surfaceTension !== undefined ? fluidSim.surfaceTension : (props.surfaceTension !== undefined ? props.surfaceTension : 0.2);
  const distort = fluidSim.distort !== undefined ? fluidSim.distort : (props.distort !== undefined ? props.distort : 0.01);
  const rgbShift = fluidSim.rgbShift !== undefined ? fluidSim.rgbShift : (props.rgbShift !== undefined ? props.rgbShift : 0.003);
  const jitter = fluidSim.jitter !== undefined ? fluidSim.jitter : (props.jitter !== undefined ? props.jitter : 0.1);
  const specular = fluidSim.specular !== undefined ? fluidSim.specular : (props.specular !== undefined ? props.specular : 0.6);
  const lighting = fluidSim.lighting !== undefined ? fluidSim.lighting : (props.lighting !== undefined ? props.lighting : 0.4);
  const surfaceZ = fluidSim.surfaceZ !== undefined ? fluidSim.surfaceZ : (props.surfaceZ !== undefined ? props.surfaceZ : 0.1);
  const causticIntensity = fluidSim.causticIntensity !== undefined ? fluidSim.causticIntensity : (props.causticIntensity !== undefined ? props.causticIntensity : 0.2);

  const introGroup = props.startingAnimationGroup || {};
  const enableIntro = introGroup.enableIntro !== undefined ? introGroup.enableIntro : (props.enableIntro !== undefined ? props.enableIntro : true);
  const introStyle = introGroup.introStyle !== undefined ? introGroup.introStyle : (props.introStyle !== undefined ? props.introStyle : "calm-reveal");
  const introDuration = introGroup.introDuration !== undefined ? introGroup.introDuration : (props.introDuration !== undefined ? props.introDuration : 1.8);
  const introStagger = introGroup.introStagger !== undefined ? introGroup.introStagger : (props.introStagger !== undefined ? props.introStagger : 0.08);
  const introEaseCurve = introGroup.introEaseCurve !== undefined ? introGroup.introEaseCurve : (props.introEaseCurve !== undefined ? props.introEaseCurve : "expo-out");
  const introStartScale = introGroup.introStartScale !== undefined ? introGroup.introStartScale : (props.introStartScale !== undefined ? props.introStartScale : 0.95);
  const introYOffset = introGroup.introYOffset !== undefined ? introGroup.introYOffset : (props.introYOffset !== undefined ? props.introYOffset : -0.4);
  const introRotation = introGroup.introRotation !== undefined ? introGroup.introRotation : (props.introRotation !== undefined ? props.introRotation : 0);

  const [isClient, setIsClient] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [viewportWidth, setViewportWidth] = useState<number>(typeof window !== 'undefined' ? window.innerWidth : 1200);
  const [viewportHeight, setViewportHeight] = useState<number>(typeof window !== 'undefined' ? window.innerHeight : 800);
  const [webglError, setWebglError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textRefs = useRef<(HTMLDivElement | null)[]>([]);
  const tagRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Robust mobile & tablet view detection that triggers correctly on real touch screens,
  // responsive simulation viewports (<1024px width), and Framer mobile canvas frames.
  const isMobileOrTablet = isMobile || viewportWidth < 1024 || (typeof window !== 'undefined' && ('ontouchstart' in window || navigator.maxTouchPoints > 0));

  // Parse the new autoResponsiveScale parameter (default: false as requested)
  const autoResponsiveScale = props.autoResponsiveScale !== undefined ? props.autoResponsiveScale : false;

  // Combine with any user-specified verticalOffset (which we also scale on mobile)
  const userOffsetRaw = layout.verticalOffset !== undefined ? layout.verticalOffset : (props.verticalOffset !== undefined ? props.verticalOffset : 0);
  const userOffset = userOffsetRaw * (viewportWidth < 768 ? 0.7 : 1.0);

  // Determine scale and vertical offset based on the selected mode
  let scale = 1.0;
  let totalVerticalOffset = userOffset;

  if (autoResponsiveScale) {
    // 1. Determine base scale based on viewport width
    if (viewportWidth < 768) {
      scale = 0.65 + (0.35 * (viewportWidth - 320) / (768 - 320));
      scale = Math.max(0.6, Math.min(1.0, scale));
    }

    // 2. Define safe boundaries (in screen pixels)
    // Header margin: room at the top for headers/menus
    const safeHeaderMargin = viewportWidth < 768 ? 95 : 120;
    // Footer margin: room at the bottom
    const safeFooterMargin = viewportWidth < 768 ? 40 : 60;

    // The maximum available height for the card
    const maxAvailableHeight = viewportHeight - safeHeaderMargin - safeFooterMargin;

    // 3. If the scaled card is too tall for the available height, scale it down further
    const currentCardHeight = cardHeight * scale;
    if (currentCardHeight > maxAvailableHeight && maxAvailableHeight > 0) {
      const heightScale = maxAvailableHeight / cardHeight;
      scale = Math.max(0.4, heightScale); // Clamp to a minimum scale of 0.4 to prevent cards from disappearing
    }

    // 4. Calculate the vertical offset to perfectly center the card in the *available* space
    // The center of the available space is:
    const availableSpaceCenter = safeHeaderMargin + (maxAvailableHeight * 0.5);
    // The default center of the viewport is:
    const viewportCenter = viewportHeight * 0.5;
    // The difference is our target auto vertical offset (positive pushes the cards down)
    const autoSafetyOffset = availableSpaceCenter - viewportCenter;

    totalVerticalOffset = userOffset + autoSafetyOffset;
  } else {
    // Completely fixed-size behavior. Absolutely no dynamic resizing or scaling when stretched!
    if (isMobileOrTablet) {
      scale = 0.85; // Slight mobile scaling for real mobile devices only
    } else {
      scale = 1.0;  // Fully fixed 1.0 scale on desktop / Framer canvas
    }
  }

  // Recalculate card dimensions with the final scale
  const finalCardHeight = cardHeight * scale;

  // Parse font sizes securely and apply dynamic scale for mobile/small viewports
  const parsedTitleFontSize = typeof titleFont.fontSize === 'number' ? titleFont.fontSize : parseInt(titleFont.fontSize) || 13;
  const parsedTagFontSize = typeof tagFont.fontSize === 'number' ? tagFont.fontSize : parseInt(tagFont.fontSize) || 10;

  const responsiveTitleStyle = {
    ...titleFont,
    fontSize: Math.max(9, Math.round(parsedTitleFontSize * scale)),
  };

  const responsiveTagStyle = {
    ...tagFont,
    fontSize: Math.max(8, Math.round(parsedTagFontSize * scale)),
  };

  // Initialize scroll physics based on configuration props
  const targetScroll = useVirtualScroll({
    touchpadMultiplier,
    mouseMultiplier,
    friction,
    dragMultiplier
  });

  // To ensure a seamless infinite scroll even with few items, we repeat items if count is small.
  const displayItems = React.useMemo(() => {
    const list = items && items.length > 0 ? items : defaultItems;
    let result = [...list];
    while (result.length > 0 && result.length < 6) {
      result = [...result, ...list];
    }
    return result;
  }, [items]);

  // Reference for updating properties on each frame dynamically without rebuilding
  const controlsRef = useRef({
    cardWidth: cardWidth * scale, cardHeight: cardHeight * scale, cardGap: cardGap * scale, lerpSpeed, velocityMultiplier,
    totalVerticalOffset,
    bendStrength, bendFocalPoint, bendFalloff, motionBlurStrength,
    rgbSplit, grainStrength, enableHoverText, enableTag, showTextOnMobile,
    tickerEnabled, tickerSpeed,
    dissipation, radius, velocityScale, advect, surfaceTension,
    distort, rgbShift, jitter, specular, lighting, surfaceZ, causticIntensity,
    enableIntro, introStyle, introDuration, introStagger,
    introEaseCurve, introStartScale, introYOffset, introRotation
  });

  useEffect(() => {
    controlsRef.current = {
      cardWidth: cardWidth * scale, cardHeight: cardHeight * scale, cardGap: cardGap * scale, lerpSpeed, velocityMultiplier,
      totalVerticalOffset,
      bendStrength, bendFocalPoint, bendFalloff, motionBlurStrength,
      rgbSplit, grainStrength, enableHoverText, enableTag, showTextOnMobile,
      tickerEnabled, tickerSpeed,
      dissipation, radius, velocityScale, advect, surfaceTension,
      distort, rgbShift, jitter, specular, lighting, surfaceZ, causticIntensity,
      enableIntro, introStyle, introDuration, introStagger,
      introEaseCurve, introStartScale, introYOffset, introRotation
    };
  }, [
    scale,
    totalVerticalOffset,
    autoResponsiveScale,
    cardWidth, cardHeight, cardGap, lerpSpeed, velocityMultiplier,
    bendStrength, bendFocalPoint, bendFalloff, motionBlurStrength,
    rgbSplit, grainStrength, enableHoverText, enableTag, showTextOnMobile,
    tickerEnabled, tickerSpeed,
    dissipation, radius, velocityScale, advect, surfaceTension,
    distort, rgbShift, jitter, specular, lighting, surfaceZ, causticIntensity,
    enableIntro, introStyle, introDuration, introStagger,
    introEaseCurve, introStartScale, introYOffset, introRotation
  ]);

  useEffect(() => {
    setIsClient(true);
    setIsMobile('ontouchstart' in window || navigator.maxTouchPoints > 0);
  }, []);

  const getImageUrl = (urlVal: any): string => {
    if (!urlVal) return '';
    if (typeof urlVal === 'string') return urlVal;
    if (typeof urlVal === 'object') {
      if (urlVal.src) return urlVal.src;
      if (urlVal.url) return urlVal.url;
      // Search for any string starting with http or data:
      for (const key in urlVal) {
        if (typeof urlVal[key] === 'string' && (urlVal[key].startsWith('http') || urlVal[key].startsWith('data:'))) {
          return urlVal[key];
        }
      }
    }
    return '';
  };

  const displayItemsSerialized = JSON.stringify(displayItems);

  useEffect(() => {
    if (!isClient) return;

    // Reset WebGL error state on mount/rebuild to allow retrying
    setWebglError(null);

    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    // 1. Setup dimensions dynamically using the actual container size
    let widthPx = container.clientWidth || window.innerWidth;
    let heightPx = container.clientHeight || window.innerHeight;
    let dpr = window.devicePixelRatio || 1;

    // 2. Main Scene & Camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    const camera = new THREE.PerspectiveCamera(45, widthPx / heightPx, 0.1, 100);
    camera.position.set(0, 0, 8);

    // Pre-flight check WebGL support to handle context loss/exhaustion gracefully
    const hasWebGL = !!(canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    if (!hasWebGL) {
      setWebglError("Your browser has exceeded the WebGL context limit, has hardware acceleration disabled, or context was lost.");
      return;
    }

    let renderer: THREE.WebGLRenderer;
    let gl: any;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
      });
      renderer.setSize(widthPx, heightPx, false);
      renderer.setPixelRatio(dpr);
      gl = renderer.getContext();
      if (!gl) {
        throw new Error("Unable to retrieve WebGL rendering context.");
      }
    } catch (e: any) {
      console.error("Error creating WebGL context:", e);
      setWebglError(e?.message || "WebGL context creation failed");
      return;
    }

    // Check device capabilities to select the most high-performance supported render target type
    const isWebGL2 = renderer.capabilities.isWebGL2;
    let type: any = THREE.UnsignedByteType;

    if (isWebGL2) {
      const extHalf = gl.getExtension('EXT_color_buffer_half_float');
      const extFloat = gl.getExtension('EXT_color_buffer_float');
      if (extHalf) {
        type = THREE.HalfFloatType;
      } else if (extFloat) {
        type = THREE.FloatType;
      }
    } else {
      const extHalf = gl.getExtension('OES_texture_half_float');
      const extFloat = gl.getExtension('OES_texture_float');
      if (extHalf) {
        type = THREE.HalfFloatType;
      } else if (extFloat) {
        type = THREE.FloatType;
      }
    }

    // Verify framebuffer support before utilizing float/half-float rendering
    let supportsHalfFloatRender = false;
    if (type !== THREE.UnsignedByteType) {
      try {
        const testFBO = new THREE.WebGLRenderTarget(16, 16, {
          type: type,
          format: THREE.RGBAFormat,
          depthBuffer: false,
          stencilBuffer: false,
        });
        renderer.setRenderTarget(testFBO);
        const fbStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        renderer.setRenderTarget(null);
        testFBO.dispose();
        if (fbStatus === gl.FRAMEBUFFER_COMPLETE) {
          supportsHalfFloatRender = true;
        }
      } catch (e) {
        supportsHalfFloatRender = false;
      }
    }

    if (!supportsHalfFloatRender) {
      type = THREE.UnsignedByteType;
    }

    // 3. Fluid simulation FBO helper functions with safe try-catch fallbacks
    const getFBO = (w: number, h: number, format: any, fboType: any, filter: any) => {
      try {
        return new THREE.WebGLRenderTarget(Math.ceil(w), Math.ceil(h), {
          type: fboType,
          format,
          minFilter: filter,
          magFilter: filter,
          depthBuffer: false,
          stencilBuffer: false,
        });
      } catch (e) {
        console.warn('FBO creation failed with type, falling back to UnsignedByteType', e);
        return new THREE.WebGLRenderTarget(Math.ceil(w), Math.ceil(h), {
          type: THREE.UnsignedByteType,
          format,
          minFilter: filter,
          magFilter: filter,
          depthBuffer: false,
          stencilBuffer: false,
        });
      }
    };

    const getDoubleFBO = (w: number, h: number, format: any, fboType: any, filter: any) => {
      return {
        read: getFBO(w, h, format, fboType, filter),
        write: getFBO(w, h, format, fboType, filter),
        swap: function () {
          const temp = this.read;
          this.read = this.write;
          this.write = temp;
        }
      };
    };

    const filter = THREE.LinearFilter;

    // Create FBOs. Note: sceneFbo is standard 3D rendering of images, so it ALWAYS uses THREE.UnsignedByteType for 100% universal compatibility.
    let paintFbo = getDoubleFBO(widthPx * dpr / 4, heightPx * dpr / 4, THREE.RGBAFormat, type, filter);
    let blurFbo = getDoubleFBO(widthPx * dpr / 8, heightPx * dpr / 8, THREE.RGBAFormat, type, filter);
    let sceneFbo = getFBO(widthPx * dpr, heightPx * dpr, THREE.RGBAFormat, THREE.UnsignedByteType, filter);

    // 4. Create fluid simulation quad & utility scene
    const simScene = new THREE.Scene();
    // Use a standard non-clipping orthographic camera for screen-space quad rendering
    const simCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    simCamera.position.set(0, 0, 1);
    const simQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    simScene.add(simQuad);

    const renderQuad = (target: THREE.WebGLRenderTarget | null, material: THREE.ShaderMaterial) => {
      simQuad.material = material;
      renderer.setRenderTarget(target);
      renderer.render(simScene, simCamera);
      renderer.setRenderTarget(null);
    };

    // 5. Build Shader Materials
    const paintMat = new THREE.ShaderMaterial({
      vertexShader: baseVertex,
      fragmentShader: paintFragment,
      uniforms: {
        uBuffer: { value: null },
        uMouse: { value: new THREE.Vector2(-1, -1) },
        uMouseVelocity: { value: new THREE.Vector2(0, 0) },
        uAspect: { value: widthPx / heightPx },
        uRadius: { value: radius },
        uDissipation: { value: dissipation },
        uVelocityDissipation: { value: 0.96 },
        uAdvect: { value: advect },
        uTexelSize: { value: new THREE.Vector2(1 / (widthPx * dpr / 4), 1 / (heightPx * dpr / 4)) },
        uWaveSpeed: { value: 0.05 + surfaceTension * 0.3 },
        uVorticity: { value: surfaceTension * 0.5 },
        uHeightForce: { value: velocityScale * 0.2 },
        uHoverCenter: { value: new THREE.Vector2(0.5, 0.5) },
        uHoverActive: { value: 0.0 },
        uTime: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    });

    const blurMat = new THREE.ShaderMaterial({
      vertexShader: baseVertex,
      fragmentShader: blurFragment,
      uniforms: {
        uInput: { value: null },
        uTexelSize: { value: new THREE.Vector2(1 / (widthPx * dpr / 8), 1 / (heightPx * dpr / 8)) },
        uMaxBlurRadius: { value: 1.5 }
      },
      depthTest: false,
      depthWrite: false,
    });

    const postMat = new THREE.ShaderMaterial({
      vertexShader: baseVertex,
      fragmentShader: postFragment,
      uniforms: {
        tMap: { value: null },
        tSimulation: { value: null },
        uTime: { value: 0 },
        uTexelSize: { value: new THREE.Vector2(1 / (widthPx * dpr), 1 / (heightPx * dpr)) },
        
        // High fidelity mappings
        uHeightScale: { value: distort * 400.0 },
        uRefractionStrength: { value: distort * 5.0 },
        uRefractionOffset: { value: surfaceZ * 10.0 },
        uRgbShift: { value: rgbShift },
        uSpecular: { value: specular * 1.5 },
        uRoughness: { value: 0.12 },
        uSpecularColorIntensity: { value: 1.5 },
        uAbsorptionStrength: { value: lighting * 1.5 },
        uCausticIntensity: { value: causticIntensity * 4.0 },
        uMicroStrength: { value: jitter * 0.1 },
        uMicroScale: { value: 25.0 },
        uAmbientReflection: { value: lighting * 0.8 },

        // Interactive card hover highlights projection
        uHoverCenter: { value: new THREE.Vector2(0.5, 0.5) },
        uHoverSize: { value: new THREE.Vector2(0.3, 0.4) },
        uHoverActive: { value: 0.0 },
      },
      depthTest: false,
      depthWrite: false,
    });

    // 6. Build Image Planes
    const textureLoader = new THREE.TextureLoader();
    textureLoader.setCrossOrigin('anonymous');
    const planeGeometry = new THREE.PlaneGeometry(3.1, 3.7, 32, 32);

    // Create a robust gradient fallback texture helper for modern, elegant look
    const createGradientTexture = (col1: string, col2: string) => {
      const canvasTex = document.createElement('canvas');
      canvasTex.width = 256;
      canvasTex.height = 256;
      const ctx = canvasTex.getContext('2d');
      if (ctx) {
        const grad = ctx.createLinearGradient(0, 0, 256, 256);
        grad.addColorStop(0, col1);
        grad.addColorStop(1, col2);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 256, 256);
      }
      const tex = new THREE.CanvasTexture(canvasTex);
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      return tex;
    };

    const premiumGradients = [
      ['#1e1e24', '#2c3e50'], // Deep Charcoal
      ['#ff7e5f', '#feb47b'], // Warm Coral/Orange
      ['#2b5876', '#4e4376'], // Slate Indigo
      ['#11998e', '#38ef7d'], // Teal Emerald
      ['#141e30', '#243b55'], // Space Blue
      ['#00c6ff', '#0072ff'], // Electric Blue
      ['#f43f5e', '#ec4899'], // Rose Horizon
    ];

    const planes: THREE.Mesh[] = [];
    const materialsList: THREE.ShaderMaterial[] = [];

    displayItems.forEach((item: any, i: number) => {
      const colors = premiumGradients[i % premiumGradients.length];
      const itemFallbackTexture = createGradientTexture(colors[0], colors[1]);

      const mat = new THREE.ShaderMaterial({
        vertexShader: galleryVertex,
        fragmentShader: galleryFragment,
        uniforms: {
          uTexture: { value: itemFallbackTexture },
          uVelocity: { value: 0 },
          uTime: { value: 0 },
          uBendStrength: { value: bendStrength },
          uFocalPoint: { value: bendFocalPoint },
          uFalloff: { value: bendFalloff },
          uRgbSplit: { value: rgbSplit },
          uMotionBlurStrength: { value: motionBlurStrength },
          uGrainStrength: { value: grainStrength },
          uHover: { value: 0.0 },
          uOpacity: { value: enableIntro ? 0.0 : 1.0 }
        },
        transparent: true,
        side: THREE.DoubleSide
      });

      const imageUrl = getImageUrl(item.url);
      if (imageUrl) {
        textureLoader.load(imageUrl, (tex) => {
          tex.minFilter = THREE.LinearFilter;
          tex.generateMipmaps = false;
          mat.uniforms.uTexture.value = tex;
        });
      }

      const mesh = new THREE.Mesh(planeGeometry, mat);
      scene.add(mesh);
      planes.push(mesh);
      materialsList.push(mat);
    });

    // 7. Raycasting & Mouse move
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2(-999, -999);
    
    const mouseData = {
      x: -1, y: -1,
      vX: 0, vY: 0,
      hasMoved: false
    };
    const lastMouse = { x: 0, y: 0, init: false };
    const mouseScreen = { x: -100, y: -100, isOver: false };

    const onPointerMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      pointer.x = (x / rect.width) * 2 - 1;
      pointer.y = -(y / rect.height) * 2 + 1;

      // Tracking mouse delta for the paint flow effect
      const clientX = e.clientX;
      const clientY = e.clientY;

      // Tracking screen coordinates for custom water cursor
      mouseScreen.x = clientX;
      mouseScreen.y = clientY;
      mouseScreen.isOver = true;
      
      if (!lastMouse.init) {
        lastMouse.init = true;
        lastMouse.x = clientX;
        lastMouse.y = clientY;
      }

      const deltaX = clientX - lastMouse.x;
      const deltaY = clientY - lastMouse.y;

      lastMouse.x = clientX;
      lastMouse.y = clientY;

      if (Math.abs(deltaX) > 0 || Math.abs(deltaY) > 0) {
        mouseData.x = x / rect.width;
        mouseData.y = 1 - (y / rect.height);
        mouseData.vX = deltaX / rect.width;
        mouseData.vY = -deltaY / rect.height;
        mouseData.hasMoved = true;
      }
    };

    const onPointerLeave = () => {
      pointer.set(-999, -999);
      mouseScreen.isOver = false;
    };

    // Pointer Down and Up click triggers for link redirection
    const pointerDownPos = { x: 0, y: 0, time: 0 };

    const onCanvasPointerDown = (e: PointerEvent) => {
      pointerDownPos.x = e.clientX;
      pointerDownPos.y = e.clientY;
      pointerDownPos.time = performance.now();
    };

    const onCanvasPointerUp = (e: PointerEvent) => {
      const now = performance.now();
      const dt = now - pointerDownPos.time;
      const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);

      // Interpret micro-gestures as a standard click instead of an active drag scroll
      if (dist < 8 && dt < 250) {
        const rect = canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        const clickPointer = new THREE.Vector2(
          (clickX / rect.width) * 2 - 1,
          -(clickY / rect.height) * 2 + 1
        );

        raycaster.setFromCamera(clickPointer, camera);
        const intersects = raycaster.intersectObjects(planes);
        if (intersects.length > 0) {
          const firstIntersect = intersects[0];
          const clickedIdx = planes.indexOf(firstIntersect.object as THREE.Mesh);
          if (clickedIdx !== -1) {
            const item = displayItems[clickedIdx];
            const itemLink = item?.link;
            if (itemLink) {
              window.open(itemLink, '_blank');
            }
          }
        }
      }
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerleave', onPointerLeave);
    container.addEventListener('pointerdown', onCanvasPointerDown);
    container.addEventListener('pointerup', onCanvasPointerUp);

    // 8. Infinite Scroll & Frame Loop variables
    const clock = new THREE.Clock();
    let currentScrollVal = 0;
    let prevScrollVal = 0;
    let scrollVelocity = 0;
    let animationId = 0;
    let lastHoveredIdx: number | null = null;
    let smoothHoverActive = 0.0;
    const hoverCenterUV = new THREE.Vector2(0.5, 0.5);
    const hoverSizeUV = new THREE.Vector2(0.3, 0.4);

    const animate = () => {
      animationId = requestAnimationFrame(animate);

      const delta = clock.getDelta();
      const elapsed = clock.getElapsedTime();
      const currentParams = controlsRef.current;

      // Update auto-scroll ticker if enabled
      if (currentParams.tickerEnabled) {
        targetScroll.current += currentParams.tickerSpeed * delta * 15.0;
      }

      // Smooth scrolling position towards target
      currentScrollVal = THREE.MathUtils.lerp(
        currentScrollVal,
        targetScroll.current,
        currentParams.lerpSpeed
      );

      const deltaScroll = -currentScrollVal - prevScrollVal;
      prevScrollVal = -currentScrollVal;
      scrollVelocity = THREE.MathUtils.lerp(scrollVelocity, deltaScroll, 0.1);

      // Convert cardWidth, cardHeight, and cardGap from px to WebGL coordinate units in real-time
      const vHeight = 2 * Math.tan((camera.fov * Math.PI) / 360) * camera.position.z;
      const webglW = currentParams.cardWidth * (vHeight / heightPx);
      const webglH = currentParams.cardHeight * (vHeight / heightPx);
      const webglG = currentParams.cardGap * (vHeight / heightPx);

      const step = webglW + webglG;
      const totalWidth = displayItems.length * step;

      const wrap = (val: number, min: number, max: number) => {
        const range = max - min;
        return ((((val - min) % range) + range) % range) + min;
      };

      // Raycast hover detection
      raycaster.setFromCamera(pointer, camera);
      const intersects = raycaster.intersectObjects(planes);
      let hoveredIdx: number | null = null;
      const isMoving = Math.abs(scrollVelocity) > 0.01;

      if (intersects.length > 0 && (currentParams.enableHoverText || currentParams.enableTag) && !isMoving) {
        const firstIntersect = intersects[0];
        hoveredIdx = planes.indexOf(firstIntersect.object as THREE.Mesh);
      }

      if (hoveredIdx !== lastHoveredIdx) {
        lastHoveredIdx = hoveredIdx;
        setHoveredIndex(hoveredIdx);
      }

      // Handle native mouse cursor pointer state
      if (hoveredIdx !== null) {
        document.body.style.cursor = 'pointer';
      } else {
        document.body.style.cursor = '';
      }

      // Position planes & update uniform states
      // Optimized single-pass loop for both WebGL meshes and DOM text overlays
      const isScrolling = Math.abs(scrollVelocity) > 0.05;
      const halfW = webglW * 0.5;
      const halfH = webglH * 0.5;
      const planeScaleX = webglW / 3.1;
      const planeScaleY = webglH / 3.7;
      const wrapThreshold = totalWidth * 0.5 - step;
      const wrapLimit = totalWidth * 0.5;
      const tempV = new THREE.Vector3();

      // Convert the total vertical offset from screen pixels to WebGL coordinate units
      // WebGL y increases upwards, but screen-space y increases downwards, so a positive screen-space offset must be negative in WebGL.
      const webglYOffset = -currentParams.totalVerticalOffset * (vHeight / heightPx);

      planes.forEach((plane, i) => {
        // Reset rotation for clean frame rendering
        plane.rotation.set(0, 0, 0);

        const baseX = (i - displayItems.length * 0.5 + 0.5) * step;
        const basePos = baseX - currentScrollVal;
        
        // Inline wrapping for speed
        let x = ((basePos + wrapLimit) % totalWidth);
        if (x < 0) x += totalWidth;
        plane.position.x = x - wrapLimit;
        plane.position.y = webglYOffset;
        
        let scaleFactor = 1.0;
        let opacityVal = 1.0;

        if (currentParams.enableIntro) {
          const introDelay = i * currentParams.introStagger;
          const introElapsed = Math.max(0, elapsed - introDelay);
          const rawProgress = Math.min(1.0, introElapsed / currentParams.introDuration);
          
          // Fully customizable or premium ease-out curves for ultra-smooth buttery starts
          let progress = 0;
          const curve = currentParams.introEaseCurve || "expo-out";
          if (curve === "expo-out") {
            progress = rawProgress === 1 ? 1 : 1 - Math.pow(2, -10 * rawProgress);
          } else if (curve === "cubic-out") {
            progress = 1 - Math.pow(1 - rawProgress, 3);
          } else if (curve === "quint-out") {
            progress = 1 - Math.pow(1 - rawProgress, 5);
          } else if (curve === "sine-out") {
            progress = Math.sin((rawProgress * Math.PI) / 2);
          } else {
            progress = rawProgress; // linear
          }

          opacityVal = progress;

          if (currentParams.introStyle === "calm-reveal") {
            plane.position.y += (progress - 1.0) * 0.35;
            scaleFactor = 0.985 + 0.015 * progress;
          } else if (currentParams.introStyle === "slide-up") {
            plane.position.y += (progress - 1.0) * 4.0;
            scaleFactor = 0.85 + 0.15 * progress;
            plane.rotation.z = (1.0 - progress) * -0.12;
          } else if (currentParams.introStyle === "scale-in") {
            scaleFactor = 0.3 + 0.7 * progress;
            plane.rotation.z = (1.0 - progress) * 0.15;
          } else if (currentParams.introStyle === "fan-out") {
            const targetX = x - wrapLimit;
            plane.position.x = THREE.MathUtils.lerp(0, targetX, progress);
            plane.position.y += (progress - 1.0) * 1.5;
            plane.rotation.z = (1.0 - progress) * (i - displayItems.length * 0.5 + 0.5) * 0.08;
            scaleFactor = 0.7 + 0.3 * progress;
          } else if (currentParams.introStyle === "fade-only") {
            // Just fade-in, keep default layout intact
          } else if (currentParams.introStyle === "custom") {
            const startScale = currentParams.introStartScale !== undefined ? currentParams.introStartScale : 0.95;
            const startYOffset = currentParams.introYOffset !== undefined ? currentParams.introYOffset : -0.4;
            const startRotDegrees = currentParams.introRotation !== undefined ? currentParams.introRotation : 0;
            const startRotRad = (startRotDegrees * Math.PI) / 180;

            plane.position.y += (progress - 1.0) * -startYOffset;
            scaleFactor = startScale + (1.0 - startScale) * progress;
            plane.rotation.z = (1.0 - progress) * startRotRad;
          }
        }

        plane.scale.set(planeScaleX * scaleFactor, planeScaleY * scaleFactor, 1);

        const mat = plane.material as THREE.ShaderMaterial;
        const isHovered = (hoveredIdx === i);
        const targetHover = isHovered ? 1.0 : 0.0;
        
        // Fast lerp
        mat.uniforms.uHover.value += (targetHover - mat.uniforms.uHover.value) * 0.15;
        mat.uniforms.uVelocity.value = scrollVelocity * currentParams.velocityMultiplier;
        mat.uniforms.uTime.value = elapsed;
        mat.uniforms.uBendStrength.value = currentParams.bendStrength;
        mat.uniforms.uFocalPoint.value = currentParams.bendFocalPoint;
        mat.uniforms.uFalloff.value = currentParams.bendFalloff;
        mat.uniforms.uMotionBlurStrength.value = currentParams.motionBlurStrength;
        mat.uniforms.uRgbSplit.value = currentParams.rgbSplit;
        mat.uniforms.uGrainStrength.value = currentParams.grainStrength;
        mat.uniforms.uOpacity.value = opacityVal;
        
        // Update overlay positions
        const textEl = textRefs.current[i];
        const tagEl = tagRefs.current[i];
        if (!textEl && !tagEl) return;

        const uVelocity = mat.uniforms.uVelocity.value;
        const uBendStrength = currentParams.bendStrength;
        const uFocalPoint = currentParams.bendFocalPoint;
        const uFalloff = currentParams.bendFalloff;

        const currentHalfW = halfW * scaleFactor;
        const currentHalfH = halfH * scaleFactor;

        // Bending math replicated for pixel-perfect tracking
        const leftWorldX = plane.position.x - currentHalfW;
        const shiftedLeftX = (leftWorldX - uFocalPoint) * uFalloff * 0.7;
        const centerMaskLeft = Math.exp(-shiftedLeftX * shiftedLeftX);
        const leftZ = -centerMaskLeft * Math.abs(uVelocity) * uBendStrength * 0.4;

        tempV.set(leftWorldX, plane.position.y + currentHalfH, leftZ);
        tempV.project(camera);
        const leftX = (tempV.x * 0.5 + 0.5) * widthPx;
        const leftY = (-tempV.y * 0.5 + 0.5) * heightPx;

        const rightWorldX = plane.position.x + currentHalfW;
        const shiftedRightX = (rightWorldX - uFocalPoint) * uFalloff * 0.7;
        const centerMaskRight = Math.exp(-shiftedRightX * shiftedRightX);
        const rightZ = -centerMaskRight * Math.abs(uVelocity) * uBendStrength * 0.4;

        tempV.set(rightWorldX, plane.position.y + currentHalfH, rightZ);
        tempV.project(camera);
        const rightX = (tempV.x * 0.5 + 0.5) * widthPx;
        const rightY = (-tempV.y * 0.5 + 0.5) * heightPx;

        const isWrappingEdge = Math.abs(plane.position.x) > wrapThreshold;
        
        // On mobile/tablet or responsive viewports, if always-on is enabled, show the text overlay for all cards
        const forceShowOnMobile = isMobileOrTablet && currentParams.showTextOnMobile;
        const showTitle = (isHovered || forceShowOnMobile) && currentParams.enableHoverText && !isScrolling && !isWrappingEdge;
        const showTag = (isHovered || forceShowOnMobile) && currentParams.enableTag && !isScrolling && !isWrappingEdge;

        if (textEl) {
          textEl.style.transform = `translate3d(${leftX}px, ${leftY}px, 0) translateY(-100%) translateY(-8px)`;
          textEl.style.opacity = showTitle ? opacityVal.toString() : '0';
        }
        if (tagEl) {
          tagEl.style.transform = `translate3d(${rightX}px, ${rightY}px, 0) translate3d(-100%, -100%, 0) translateY(-8px)`;
          tagEl.style.opacity = showTag ? opacityVal.toString() : '0';
        }
      });

      // Fluid Simulation Rendering Passes
      // 1. Calculate smooth card hover projection to locally amplify depth, refraction, reflection, and specular
      const targetHoverActiveVal = (hoveredIdx !== null) ? 1.0 : 0.0;
      smoothHoverActive = THREE.MathUtils.lerp(smoothHoverActive, targetHoverActiveVal, 0.1);

      if (hoveredIdx !== null) {
        const hoveredPlane = planes[hoveredIdx];
        if (hoveredPlane) {
          const tempV = new THREE.Vector3();
          tempV.setFromMatrixPosition(hoveredPlane.matrixWorld);
          tempV.project(camera);
          
          hoverCenterUV.set(tempV.x * 0.5 + 0.5, tempV.y * 0.5 + 0.5);
          
          const tempV2 = new THREE.Vector3();
          tempV2.setFromMatrixPosition(hoveredPlane.matrixWorld);
          tempV2.x += webglW * 0.5;
          tempV2.y += webglH * 0.5;
          tempV2.project(camera);
          
          hoverSizeUV.set(
            Math.max(0.01, Math.abs(tempV2.x - tempV.x)),
            Math.max(0.01, Math.abs(tempV2.y - tempV.y))
          );
        }
      }

      // 2. Paint FBO Pass (Solves 2D discrete wave propagation, advection & vorticity)
      paintMat.uniforms.uRadius.value = currentParams.radius;
      
      // Settle waves naturally: smooth and organic decay on exit and stationary states
      {
        // Keep the natural, configured high-fidelity fluid dissipation (e.g. 0.98)
        // across all states (hovering, leaving, moving, stationary) so waves
        // always propagate and settle with perfect, organic physical realism.
        const targetDiss = currentParams.dissipation;
        const targetVelDiss = 0.95 + (currentParams.dissipation - 0.9) * 0.2;

        // Smoothly lerp the dissipation factors to avoid harsh visual jumps
        paintMat.uniforms.uDissipation.value += (targetDiss - paintMat.uniforms.uDissipation.value) * 0.08;
        paintMat.uniforms.uVelocityDissipation.value += (targetVelDiss - paintMat.uniforms.uVelocityDissipation.value) * 0.08;
      }
      paintMat.uniforms.uAdvect.value = currentParams.advect;
      paintMat.uniforms.uWaveSpeed.value = 0.05 + currentParams.surfaceTension * 0.3;
      paintMat.uniforms.uVorticity.value = currentParams.surfaceTension * 0.5;
      paintMat.uniforms.uHeightForce.value = currentParams.velocityScale * 0.2;
      paintMat.uniforms.uTime.value = elapsed;

      paintMat.uniforms.uHoverCenter.value.copy(hoverCenterUV);
      paintMat.uniforms.uHoverActive.value = smoothHoverActive;
      paintMat.uniforms.uBuffer.value = paintFbo.read.texture;
      
      if (mouseData.hasMoved) {
        paintMat.uniforms.uMouse.value.set(mouseData.x, mouseData.y);
        paintMat.uniforms.uMouseVelocity.value.set(mouseData.vX * currentParams.velocityScale, mouseData.vY * currentParams.velocityScale);
        mouseData.hasMoved = false; // consume input
      } else {
        // Slowed decay multiplier (0.92 instead of 0.85) to let the gesture inertia
        // glide beautifully and smoothly in the fluid surface, matching real-life physics.
        paintMat.uniforms.uMouseVelocity.value.multiplyScalar(0.92); 
      }
      renderQuad(paintFbo.write, paintMat);
      paintFbo.swap();

      // 3. Adaptive Blur Pass (Cleans up aliasing of wave heights for highly smooth normals)
      blurMat.uniforms.uInput.value = paintFbo.read.texture;
      blurMat.uniforms.uMaxBlurRadius.value = 1.0 + (1.0 - currentParams.dissipation) * 5.0;
      renderQuad(blurFbo.write, blurMat);
      blurFbo.swap();

      // 4. Render 3D Scene to FBO
      renderer.setRenderTarget(sceneFbo);
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);

      // 5. Render Final Composite Post-processing Pass to Screen (Performs PBR normal shading, reflection, refraction, and absorption)
      postMat.uniforms.tMap.value = sceneFbo.texture;
      postMat.uniforms.tSimulation.value = blurFbo.read.texture;
      postMat.uniforms.uTime.value = elapsed;

      postMat.uniforms.uHeightScale.value = currentParams.distort * 400.0;
      postMat.uniforms.uRefractionStrength.value = currentParams.distort * 5.0;
      postMat.uniforms.uRefractionOffset.value = currentParams.surfaceZ * 10.0;
      postMat.uniforms.uRgbShift.value = currentParams.rgbShift;
      postMat.uniforms.uSpecular.value = currentParams.specular * 1.5;
      postMat.uniforms.uAbsorptionStrength.value = currentParams.lighting * 1.5;
      postMat.uniforms.uCausticIntensity.value = currentParams.causticIntensity * 4.0;
      postMat.uniforms.uMicroStrength.value = currentParams.jitter * 0.1;
      postMat.uniforms.uAmbientReflection.value = currentParams.lighting * 0.8;

      postMat.uniforms.uHoverCenter.value.copy(hoverCenterUV);
      postMat.uniforms.uHoverSize.value.copy(hoverSizeUV);
      postMat.uniforms.uHoverActive.value = smoothHoverActive;

      renderQuad(null, postMat);
    };

    animate();

    let resizeCount = 0;
    let lastResizeTime = 0;
    let lastHeight = 0;

    // Use a ResizeObserver on the container ref to perfectly adapt to initial sizes and responsive container scaling
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width === 0 || height === 0) continue;

        // Feedback loop protection for auto-height wrapper containers
        const now = performance.now();
        if (lastHeight > 0 && height > lastHeight && (now - lastResizeTime) < 100) {
          resizeCount++;
          if (resizeCount > 5) {
            console.warn("ResizeObserver feedback loop detected. Throttling size updates.");
            continue;
          }
        } else {
          resizeCount = 0;
        }

        lastHeight = height;
        lastResizeTime = now;

        widthPx = width;
        heightPx = height;
        dpr = window.devicePixelRatio || 1;
        setViewportWidth(width);
        setViewportHeight(height);

        camera.aspect = widthPx / heightPx;
        camera.updateProjectionMatrix();

        renderer.setSize(widthPx, heightPx, false);
        renderer.setPixelRatio(dpr);

        // Safely dispose old framebuffer memory before allocations
        paintFbo.read.dispose();
        paintFbo.write.dispose();
        blurFbo.read.dispose();
        blurFbo.write.dispose();
        sceneFbo.dispose();

        paintFbo = getDoubleFBO(widthPx * dpr / 4, heightPx * dpr / 4, THREE.RGBAFormat, type, filter);
        blurFbo = getDoubleFBO(widthPx * dpr / 8, heightPx * dpr / 8, THREE.RGBAFormat, type, filter);
        sceneFbo = getFBO(widthPx * dpr, heightPx * dpr, THREE.RGBAFormat, THREE.UnsignedByteType, filter);

        paintMat.uniforms.uAspect.value = widthPx / heightPx;
        paintMat.uniforms.uTexelSize.value.set(1 / (widthPx * dpr / 4), 1 / (heightPx * dpr / 4));
        blurMat.uniforms.uTexelSize.value.set(1 / (widthPx * dpr / 8), 1 / (heightPx * dpr / 8));
        postMat.uniforms.uTexelSize.value.set(1 / (widthPx * dpr), 1 / (heightPx * dpr));
      }
    });

    resizeObserver.observe(container);

    // Cleanup resources completely
    return () => {
      cancelAnimationFrame(animationId);
      resizeObserver.disconnect();
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerleave', onPointerLeave);
      container.removeEventListener('pointerdown', onCanvasPointerDown);
      container.removeEventListener('pointerup', onCanvasPointerUp);
      document.body.style.cursor = '';

      if (renderer) {
        renderer.dispose();
        const glContext = renderer.getContext();
        if (glContext) {
          const loseContextExt = glContext.getExtension('WEBGL_lose_context');
          if (loseContextExt) {
            loseContextExt.loseContext();
          }
        }
      }
      if (planeGeometry) planeGeometry.dispose();
      if (paintMat) paintMat.dispose();
      if (blurMat) blurMat.dispose();
      if (postMat) postMat.dispose();

      if (planes) {
        planes.forEach((plane) => {
          const mat = plane.material as THREE.ShaderMaterial;
          if (mat && mat.uniforms && mat.uniforms.uTexture.value) {
            mat.uniforms.uTexture.value.dispose();
          }
          if (mat) mat.dispose();
        });
      }

      if (paintFbo && paintFbo.read && paintFbo.write) {
        paintFbo.read.dispose();
        paintFbo.write.dispose();
      }
      if (blurFbo && blurFbo.read && blurFbo.write) {
        blurFbo.read.dispose();
        blurFbo.write.dispose();
      }
      if (sceneFbo) {
        sceneFbo.dispose();
      }
    };
  }, [isClient, displayItemsSerialized]);

  const containerStyle: React.CSSProperties = {
    width: props.width !== undefined ? (typeof props.width === 'number' ? `${props.width}px` : props.width) : '100%',
    height: props.height !== undefined ? (typeof props.height === 'number' ? `${props.height}px` : props.height) : '100vh',
  };

  if (!isClient) {
    return <div id="fluid-gallery-fallback" className="bg-white relative" style={containerStyle} />;
  }

  if (webglError) {
    return (
      <div 
        id="fluid-gallery-error" 
        className="bg-zinc-50 flex flex-col items-center justify-center p-6 text-center border border-zinc-200 rounded-lg relative"
        style={containerStyle}
      >
        <div className="max-w-md space-y-4">
          <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center mx-auto text-red-500">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-zinc-950 font-sans tracking-tight">WebGL Context Error</h3>
          <p className="text-sm text-zinc-500 font-sans leading-relaxed">
            We encountered an issue creating the WebGL context. This usually happens when too many WebGL tabs are open or hardware acceleration is disabled in your browser.
          </p>
          <div className="text-xs font-mono text-zinc-400 bg-zinc-100 p-2.5 rounded border border-zinc-200 select-all overflow-x-auto max-w-full">
            {webglError}
          </div>
          <button 
            onClick={() => window.location.reload()} 
            className="inline-flex items-center justify-center px-4 py-2 text-xs font-medium font-sans uppercase tracking-wider text-white bg-zinc-900 hover:bg-zinc-800 rounded transition-colors"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  return (
    <div id="fluid-gallery-container" ref={containerRef} className="bg-white overflow-hidden relative touch-pan-y select-none" style={containerStyle}>
      {/* Native WebGL Canvas */}
      <canvas id="threejs-canvas" ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', touchAction: 'pan-y' }} className="block touch-pan-y select-none" />

      {/* Real-time projected Typography & Accessibility Layer */}
      <div 
        id="fluid-text-layer" 
        style={{ 
          position: 'absolute', 
          top: 0, 
          left: 0, 
          right: 0, 
          bottom: 0, 
          pointerEvents: 'none', 
          zIndex: 10, 
          overflow: 'hidden' 
        }}
      >
        {displayItems.map((item: any, i: number) => {
          const isHovered = (hoveredIndex === i);
          return (
            <React.Fragment key={i}>
              {/* Left Side title */}
              <div
                ref={(el) => {
                  textRefs.current[i] = el;
                }}
                style={{ 
                  position: 'absolute', 
                  top: 0, 
                  left: 0, 
                  pointerEvents: 'none', 
                  opacity: 0, 
                  willChange: 'transform',
                  transition: 'opacity 0.2s ease'
                }}
              >
                <div 
                  style={{
                    paddingBottom: '4px',
                    paddingLeft: '4px',
                    marginLeft: '-4px',
                  }}
                >
                  <RollingText
                    text={item.title || ''}
                    hovered={enableHoverText && (isHovered || (isMobileOrTablet && showTextOnMobile))}
                    staggerDelay={textStagger}
                    duration={textDuration}
                    style={{
                      ...responsiveTitleStyle,
                      color: '#18181b',
                      textTransform: 'uppercase',
                      fontWeight: 800,
                      letterSpacing: '0.25em',
                    }}
                  />
                </div>
              </div>

              {/* Right Side tag */}
              <div
                ref={(el) => {
                  tagRefs.current[i] = el;
                }}
                style={{ 
                  position: 'absolute', 
                  top: 0, 
                  left: 0, 
                  pointerEvents: 'none', 
                  opacity: 0, 
                  willChange: 'transform',
                  transition: 'opacity 0.2s ease'
                }}
              >
                <div 
                  style={{
                    paddingBottom: '4px',
                    paddingRight: '4px',
                    marginRight: '-4px',
                  }}
                >
                  <RollingText
                    text={item.tag || ''}
                    hovered={enableTag && (isHovered || (isMobileOrTablet && showTextOnMobile))}
                    delayOffset={textStagger * 4} // Slightly longer delay for tag
                    staggerDelay={textStagger * 0.8}
                    duration={textDuration}
                    style={{
                      ...responsiveTagStyle,
                      color: '#71717a',
                      textTransform: 'uppercase',
                      fontWeight: 500,
                      letterSpacing: '0.2em',
                    }}
                  />
                </div>
              </div>
            </React.Fragment>
          );
        })}
      </div>

      {/* Accessible DOM Fallback containing genuine alt texts and anchor tags, styled to be 100% visually hidden at all times to prevent flashes/icons on load */}
      <div style={{ position: 'absolute', width: '1px', height: '1px', padding: 0, margin: '-1px', overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', border: 0, opacity: 0, pointerEvents: 'none' }}>
        {displayItems.map((item: any, i: number) => (
          <a
            key={`fallback-${i}`}
            href={item.link || '#'}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={item.alt || item.title || 'Gallery link'}
          >
            <img
              src={getImageUrl(item.url)}
              alt={item.alt || item.title || 'Gallery image'}
            />
            <span>{item.title}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

FluidGallery.defaultProps = {
  items: defaultItems,
  layoutGroup: {
    cardWidth: 360,
    cardHeight: 440,
    cardGap: 30,
    verticalOffset: 0,
    lerpSpeed: 0.08,
    velocityMultiplier: 1.0,
  },
  scrollPhysicsGroup: {
    touchpadMultiplier: 0.006,
    mouseMultiplier: 0.012,
    friction: 0.94,
    dragMultiplier: 0.01,
  },
  bendGroup: {
    bendStrength: 1.2,
    bendFocalPoint: 0.0,
    bendFalloff: 0.2,
  },
  effectsGroup: {
    motionBlurStrength: 1.5,
    rgbSplit: 1.5,
    grainStrength: 0.0,
  },
  tickerGroup: {
    tickerEnabled: false,
    tickerSpeed: 0.15,
  },
  typographyGroup: {
    enableHoverText: true,
    enableTag: true,
    showTextOnMobile: false,
    textStagger: 0.045,
    textPower: 0.75,
    textDuration: 0.7,
    textRotateStart: 85,
    textTranslateYStart: 15,
    textTranslateZStart: -30,
    textBlurStart: 3,
    textEase: "cubic-bezier(0.16, 1, 0.3, 1)",
    titleFont: {
      fontFamily: "Inter",
      fontWeight: 800,
      fontSize: 13,
      lineHeight: "1.2em",
    },
    tagFont: {
      fontFamily: "JetBrains Mono",
      fontWeight: 500,
      fontSize: 10,
      lineHeight: "1.2em",
    }
  },
  fluidSimulationGroup: {
    dissipation: 0.98,
    radius: 0.1,
    velocityScale: 5.0,
    advect: 0.001,
    surfaceTension: 0.2,
    distort: 0.01,
    rgbShift: 0.003,
    jitter: 0.1,
    specular: 0.6,
    lighting: 0.4,
    surfaceZ: 0.1,
    causticIntensity: 0.2,
  },
  startingAnimationGroup: {
    enableIntro: true,
    introStyle: "calm-reveal",
    introDuration: 1.8,
    introStagger: 0.08,
    introEaseCurve: "expo-out",
    introStartScale: 0.95,
    introYOffset: -0.4,
    introRotation: 0,
  },
  // Flat properties for older Framer versions or code-based instantiations
  enableIntro: true,
  introStyle: "calm-reveal",
  introDuration: 1.8,
  introStagger: 0.08,
  introEaseCurve: "expo-out",
  introStartScale: 0.95,
  introYOffset: -0.4,
  introRotation: 0,
  width: 1200,
  height: 800,
  cardWidth: 360,
  cardHeight: 440,
  cardGap: 30,
  verticalOffset: 0,
  lerpSpeed: 0.08,
  velocityMultiplier: 1.0,
  touchpadMultiplier: 0.006,
  mouseMultiplier: 0.012,
  friction: 0.94,
  dragMultiplier: 0.01,
  bendStrength: 12.0,
  bendFocalPoint: 0.0,
  bendFalloff: 0.2,
  motionBlurStrength: 1.5,
  rgbSplit: 1.5,
  grainStrength: 0.0,
  tickerEnabled: false,
  tickerSpeed: 0.15,
  enableHoverText: true,
  enableTag: true,
  showTextOnMobile: false,
  textStagger: 0.045,
  textPower: 0.75,
  textDuration: 0.7,
  textRotateStart: 85,
  textTranslateYStart: 15,
  textTranslateZStart: -30,
  textBlurStart: 3,
  textEase: "cubic-bezier(0.16, 1, 0.3, 1)",
  titleFont: {
    fontFamily: "Inter",
    fontWeight: 800,
    fontSize: 13,
    lineHeight: "1.2em",
  },
  tagFont: {
    fontFamily: "JetBrains Mono",
    fontWeight: 500,
    fontSize: 10,
    lineHeight: "1.2em",
  },
  dissipation: 0.98,
  radius: 0.1,
  velocityScale: 5.0,
  advect: 0.001,
  surfaceTension: 0.2,
  distort: 0.01,
  rgbShift: 0.003,
  jitter: 0.1,
  specular: 0.6,
  lighting: 0.4,
  surfaceZ: 0.1,
  causticIntensity: 0.2,
};

// Map all properties to Framer Property Controls for native panel adjustments
addPropertyControls(FluidGallery, {
  autoResponsiveScale: {
    type: ControlType.Boolean,
    title: "Auto Safety Scale",
    defaultValue: false,
  },
  width: {
    type: ControlType.Number,
    title: "Container Width",
    defaultValue: 1200,
    min: 200,
    max: 3000,
    step: 1,
  },
  height: {
    type: ControlType.Number,
    title: "Container Height",
    defaultValue: 800,
    min: 200,
    max: 3000,
    step: 1,
  },
  // Gallery Items
  items: {
    type: ControlType.Array,
    title: "Gallery Items",
    control: {
      type: ControlType.Object,
      controls: {
        url: { type: ControlType.Image, title: "Image" },
        title: { type: ControlType.String, title: "Title" },
        tag: { type: ControlType.String, title: "Tag" },
        link: { type: ControlType.String, title: "Redirect Link" },
        alt: { type: ControlType.String, title: "Alt Accessibility Text" },
      }
    }
  },

  // Layout Parameters
  layoutGroup: {
    type: ControlType.Object,
    title: "Layout & Scale",
    controls: {
      cardWidth: { type: ControlType.Number, title: "Card Width (px)", defaultValue: 360, min: 100, max: 1000, step: 1 },
      cardHeight: { type: ControlType.Number, title: "Card Height (px)", defaultValue: 440, min: 100, max: 1000, step: 1 },
      cardGap: { type: ControlType.Number, title: "Card Gap (px)", defaultValue: 30, min: 0, max: 200, step: 1 },
      verticalOffset: { type: ControlType.Number, title: "Vertical Y-Offset (px)", defaultValue: 0, min: -400, max: 400, step: 1 },
      lerpSpeed: { type: ControlType.Number, title: "Scroll Lerp", defaultValue: 0.08, min: 0.01, max: 0.3, step: 0.01 },
      velocityMultiplier: { type: ControlType.Number, title: "Bend Velocity Mult", defaultValue: 1.0, min: 0.1, max: 5.0, step: 0.1 },
    }
  },

  // Scroll Physics
  scrollPhysicsGroup: {
    type: ControlType.Object,
    title: "Scroll Physics",
    controls: {
      touchpadMultiplier: { type: ControlType.Number, title: "Scroll Touchpad", defaultValue: 0.006, min: 0.001, max: 0.05, step: 0.001 },
      mouseMultiplier: { type: ControlType.Number, title: "Scroll Mouse", defaultValue: 0.012, min: 0.001, max: 0.05, step: 0.001 },
      friction: { type: ControlType.Number, title: "Scroll Friction", defaultValue: 0.94, min: 0.8, max: 0.99, step: 0.01 },
      dragMultiplier: { type: ControlType.Number, title: "Scroll Drag", defaultValue: 0.01, min: 0.001, max: 0.05, step: 0.001 },
    }
  },

  // Starting Animation Settings
  startingAnimationGroup: {
    type: ControlType.Object,
    title: "Starting Animation",
    controls: {
      enableIntro: { type: ControlType.Boolean, title: "Enable Intro", defaultValue: true },
      introStyle: {
        type: ControlType.Enum,
        title: "Intro Style",
        options: ["calm-reveal", "slide-up", "scale-in", "fan-out", "fade-only", "custom"],
        optionTitles: ["Calm Reveal", "Slide Up", "Scale In", "Fan Out", "Fade Only", "Custom (Set Below)"],
        defaultValue: "calm-reveal"
      },
      introDuration: { type: ControlType.Number, title: "Intro Duration (s)", defaultValue: 1.8, min: 0.5, max: 4.0, step: 0.1 },
      introStagger: { type: ControlType.Number, title: "Stagger Delay (s)", defaultValue: 0.08, min: 0.0, max: 0.5, step: 0.01 },
      introEaseCurve: {
        type: ControlType.Enum,
        title: "Ease Curve",
        options: ["expo-out", "cubic-out", "quint-out", "sine-out", "linear"],
        optionTitles: ["Expo Out (Buttery)", "Cubic Out (Smooth)", "Quint Out (Snappy)", "Sine Out (Gentle)", "Linear"],
        defaultValue: "expo-out",
        hidden(props) {
          const style = props.introStyle || (props.startingAnimationGroup && props.startingAnimationGroup.introStyle);
          return style !== "custom";
        }
      },
      introStartScale: {
        type: ControlType.Number,
        title: "Start Scale",
        defaultValue: 0.95,
        min: 0.1,
        max: 2.0,
        step: 0.01,
        hidden(props) {
          const style = props.introStyle || (props.startingAnimationGroup && props.startingAnimationGroup.introStyle);
          return style !== "custom";
        }
      },
      introYOffset: {
        type: ControlType.Number,
        title: "Start Y Offset",
        defaultValue: -0.4,
        min: -6.0,
        max: 6.0,
        step: 0.05,
        hidden(props) {
          const style = props.introStyle || (props.startingAnimationGroup && props.startingAnimationGroup.introStyle);
          return style !== "custom";
        }
      },
      introRotation: {
        type: ControlType.Number,
        title: "Start Rotation (°)",
        defaultValue: 0,
        min: -90,
        max: 90,
        step: 1,
        hidden(props) {
          const style = props.introStyle || (props.startingAnimationGroup && props.startingAnimationGroup.introStyle);
          return style !== "custom";
        }
      }
    }
  },

  // Slide Bending
  bendGroup: {
    type: ControlType.Object,
    title: "Bending Distortion",
    controls: {
      bendStrength: { type: ControlType.Number, title: "Bend Strength", defaultValue: 12.0, min: 0.0, max: 30.0, step: 0.1 },
      bendFocalPoint: { type: ControlType.Number, title: "Bend Focal Point", defaultValue: 0.0, min: -10.0, max: 10.0, step: 0.1 },
      bendFalloff: { type: ControlType.Number, title: "Bend Falloff", defaultValue: 0.2, min: 0.0, max: 2.0, step: 0.01 },
    }
  },

  // Effects
  effectsGroup: {
    type: ControlType.Object,
    title: "Image Post-FX",
    controls: {
      motionBlurStrength: { type: ControlType.Number, title: "Motion Blur", defaultValue: 1.5, min: 0.0, max: 20.0, step: 0.1 },
      rgbSplit: { type: ControlType.Number, title: "RGB Split", defaultValue: 1.5, min: 0.0, max: 20.0, step: 0.05 },
      grainStrength: { type: ControlType.Number, title: "Grain Strength", defaultValue: 0.0, min: 0.0, max: 0.3, step: 0.01 },
    }
  },

  // Ticker Settings
  tickerGroup: {
    type: ControlType.Object,
    title: "Auto-Ticker",
    controls: {
      tickerEnabled: { type: ControlType.Boolean, title: "Enable Ticker", defaultValue: false },
      tickerSpeed: { type: ControlType.Number, title: "Ticker Speed", defaultValue: 0.15, min: -2.0, max: 2.0, step: 0.05 },
    }
  },

  // Typography Settings
  typographyGroup: {
    type: ControlType.Object,
    title: "Typography Details",
    controls: {
      enableHoverText: { type: ControlType.Boolean, title: "Show Text", defaultValue: true },
      enableTag: { type: ControlType.Boolean, title: "Show Tag", defaultValue: true },
      showTextOnMobile: { type: ControlType.Boolean, title: "Always on Mobile", defaultValue: false },
      titleFont: {
        type: ControlType.Font,
        title: "Title Font",
        controls: "extended",
        defaultValue: {
          fontFamily: "Inter",
          fontWeight: 800,
          fontSize: 13,
          lineHeight: "1.2em",
        } as any
      },
      tagFont: {
        type: ControlType.Font,
        title: "Tag Font",
        controls: "extended",
        defaultValue: {
          fontFamily: "JetBrains Mono",
          fontWeight: 500,
          fontSize: 10,
          lineHeight: "1.2em",
        } as any
      },
      textStagger: { type: ControlType.Number, title: "Text Stagger", defaultValue: 0.045, min: 0.005, max: 0.15, step: 0.005 },
      textPower: { type: ControlType.Number, title: "Text Power", defaultValue: 0.75, min: 0.3, max: 2.0, step: 0.05 },
      textDuration: { type: ControlType.Number, title: "Text Duration", defaultValue: 0.7, min: 0.2, max: 2.0, step: 0.05 },
      textRotateStart: { type: ControlType.Number, title: "Text Rotate Start", defaultValue: 85, min: 0, max: 180, step: 5 },
      textTranslateYStart: { type: ControlType.Number, title: "Text Translate Y Start", defaultValue: 15, min: 0, max: 100, step: 1 },
      textTranslateZStart: { type: ControlType.Number, title: "Text Translate Z Start", defaultValue: -30, min: -100, max: 100, step: 5 },
      textBlurStart: { type: ControlType.Number, title: "Text Blur Start", defaultValue: 3, min: 0, max: 10, step: 0.5 },
      textEase: { type: ControlType.String, title: "Text CSS Easing", defaultValue: "cubic-bezier(0.16, 1, 0.3, 1)" },
    }
  },

  // Fluid Distortion parameters
  fluidSimulationGroup: {
    type: ControlType.Object,
    title: "Fluid Dynamics Simulation",
    controls: {
      dissipation: { type: ControlType.Number, title: "Fluid Dissipation", defaultValue: 0.98, min: 0.9, max: 1.0, step: 0.001 },
      radius: { type: ControlType.Number, title: "Fluid Radius", defaultValue: 0.1, min: 0.01, max: 0.5, step: 0.01 },
      velocityScale: { type: ControlType.Number, title: "Fluid Velocity Scale", defaultValue: 5.0, min: 0.1, max: 50.0, step: 0.1 },
      advect: { type: ControlType.Number, title: "Fluid Advection", defaultValue: 0.001, min: 0.0, max: 0.05, step: 0.001 },
      surfaceTension: { type: ControlType.Number, title: "Fluid Surf Tension", defaultValue: 0.2, min: 0.0, max: 1.0, step: 0.01 },
      distort: { type: ControlType.Number, title: "Fluid Distort Amount", defaultValue: 0.01, min: 0.0, max: 0.2, step: 0.001 },
      rgbShift: { type: ControlType.Number, title: "Fluid RGB Shift", defaultValue: 0.003, min: 0.0, max: 0.1, step: 0.001 },
      jitter: { type: ControlType.Number, title: "Fluid Jitter", defaultValue: 0.1, min: 0.0, max: 2.0, step: 0.01 },
      specular: { type: ControlType.Number, title: "Fluid Specular", defaultValue: 0.6, min: 0.0, max: 5.0, step: 0.01 },
      lighting: { type: ControlType.Number, title: "Fluid Lighting", defaultValue: 0.4, min: 0.0, max: 2.0, step: 0.01 },
      surfaceZ: { type: ControlType.Number, title: "Fluid Surface Z", defaultValue: 0.1, min: 0.001, max: 0.5, step: 0.001 },
      causticIntensity: { type: ControlType.Number, title: "Fluid Caustic", defaultValue: 0.2, min: 0.0, max: 5.0, step: 0.01 }
    }
  }
});
