// Ambient animated backdrop: three slow-drifting blurred blobs in the brand
// colors, rendered behind all page content (styles in globals.css).
// transform-only animation, pointer-events: none, and disabled entirely for
// prefers-reduced-motion users.
export function AnimatedBackground() {
  return (
    <div aria-hidden="true" className="p2c-bg-blobs">
      <div className="p2c-blob p2c-blob-a" />
      <div className="p2c-blob p2c-blob-b" />
      <div className="p2c-blob p2c-blob-c" />
    </div>
  );
}

export default AnimatedBackground;
