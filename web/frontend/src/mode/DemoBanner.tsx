// DemoBanner.tsx — Inline notice rendered at the top of the SPA when
// the bundle was built for GitHub Pages.
//
// The banner tells the visitor that no controller is attached, points
// at the wiki for setup instructions, and reminds them that the Copy
// DSL button is the way to get a sequence onto real hardware.

export function DemoBanner(): JSX.Element {
  return (
    <div className="demo-banner" data-testid="demo-banner" role="status">
      <strong>Demo mode.</strong> No controller is attached. Build a
      sequence, then copy the DSL into the curses editor on a machine
      running the Reactive Robot controller.{" "}
      <a
        href="../"
        className="demo-banner__link"
        data-testid="demo-banner-wiki-link"
      >
        Read the wiki for setup instructions.
      </a>
    </div>
  );
}
