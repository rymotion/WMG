// C interface to the wmg-features crate (built with `--features ffi`).
// The same Rust code computes detector features in training, on the server and on device.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t wmg_n_mels(void);
uint32_t wmg_frames_per_window(void);
// NUL-terminated feature-spec hash; must equal the model's feature_spec_hash.
const char *wmg_spec_hash(void);
// Upper bound on windows wmg_extract_windows can return for this input.
size_t wmg_max_windows(size_t n_samples, uint32_t sample_rate);
// Mono samples -> log-mel windows [max_windows][n_mels][frames] + start times (s).
// normalize != 0 scales the clip to the training RMS level first; input quieter than -60 dBFS
// is treated as silence and returns 0 windows. Returns windows written or -1.
int64_t wmg_extract_windows(const float *samples, size_t n_samples, uint32_t sample_rate,
                            int32_t normalize, float *out_mel, float *out_start_s,
                            size_t max_windows);
int64_t wmg_resample(const float *input, size_t n, uint32_t sr_in, uint32_t sr_out,
                     float *out, size_t out_cap);

#ifdef __cplusplus
}
#endif
