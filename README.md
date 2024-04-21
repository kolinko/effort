# Effort Engine

Effort engine is an example implementation of the bucketMul algorithm - you can read [about it here](http://kolinko.github.io/effort/).

With it you can smoothly adjust—in real time—the number of calculations performed during the inference of an LLM model.

At 50% effort, it performs as fast as regular matrix multiplications on Apple Silicon chips; at 25% effort, it is twice as fast while still retaining most of the quality.

You also have the option to skip loading the least important weights.

## Getting Started

### Binaries
You can quickly get started by downloading the precompiled binaries available at:
[Effort Engine v0.0.1](https://github.com/kolinko/effort/releases/download/0.0.1/effort.0.0.1.zip)

To bypass macOS Gatekeeper, hold `option` while clicking to open the downloaded application for the first time.

### Initial Setup
On the first run, you will be prompted to download the converted weights necessary for operation. Subsequently, a matrix multiplication benchmark will execute to demonstrate the capabilities of the engine.

### Source Code

The sources are in Swift & Metal.

Download and open effort.xcodeproj. It should work straight away.

## Additional Resources

- **More Information:** Visit our [project page](http://kolinko.github.io/effort/).
- **See it in Action:** Watch a demo on [Asciinema](https://asciinema.org/a/piP22yYwcaohu5cA2gyuv1W61).

## Updates
Stay tuned for updates to this README. After a busy day of website development, refinements are on the way!
