# shellcheck shell=bash
# Sourced by harness/pod/serve_variant.sh; see harness/README.md for when each hook runs.

# Stage 2 ran on torch's default allocator; an image or pod env that turns on expandable_segments would mix in the
# custom all-reduce half of #44291. Newer torch reads PYTORCH_ALLOC_CONF first, so both names go.
pre_launch() {
    unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
}
