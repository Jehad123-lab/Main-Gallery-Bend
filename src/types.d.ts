import * as _THREE from 'three';

declare global {
  interface Window {
    THREE: typeof _THREE;
  }
}
