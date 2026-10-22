#ifndef GRAIN_ELEVATOR_H
#define GRAIN_ELEVATOR_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ge_encoder ge_encoder;
typedef struct ge_embedding ge_embedding;
/* image: packaged executable or library path, UTF-8; NULL uses host executable.
   threads=0 uses min(4,max(1,cpu_count/2)). GRAIN_ORT_LIB can locate the runtime.
   All returned tensor pointers remain valid until ge_embedding_free.
   Callers must provide valid buffers/handles and close only after operations finish. */
ge_encoder *ge_open(const char *image, size_t threads);
void ge_close(ge_encoder *encoder);
ge_embedding *ge_embed_file(const ge_encoder *encoder, const char *path);
ge_embedding *ge_embed_pcm(const ge_encoder *encoder, const float *mono, size_t count, uint32_t rate);
const float *ge_embedding_data(const ge_embedding *value, size_t *count);
const size_t *ge_embedding_shape(const ge_embedding *value, size_t *rank);
const bool *ge_embedding_mask(const ge_embedding *value, size_t *count);
size_t ge_embedding_identity(const ge_embedding *value, char *out, size_t capacity);
void ge_embedding_free(ge_embedding *value);
/* String accessors return required capacity including NUL; NULL/0 queries size. */
size_t ge_last_error(char *out, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
