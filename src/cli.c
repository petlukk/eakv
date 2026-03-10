#include "internal.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>

static int cmd_inspect(const char *path) {
    eakv_cache_t *cache = NULL;
    int rc = eakv_cache_load(path, &cache);
    if (rc != EAKV_OK) {
        fprintf(stderr, "error: failed to load %s (code %d)\n", path, rc);
        return 1;
    }

    struct stat st;
    long file_size = 0;
    if (stat(path, &st) == 0) file_size = st.st_size;

    int sl = eakv_cache_seq_len(cache);
    int nl = eakv_cache_n_layers(cache);
    int nh = eakv_cache_n_heads(cache);
    int hd = eakv_cache_head_dim(cache);
    long orig_bytes = (long)nl * 2 * nh * sl * hd * 4;

    printf("%s\n", path);
    printf("  layers:     %d\n", nl);
    printf("  kv_heads:   %d\n", nh);
    printf("  head_dim:   %d\n", hd);
    printf("  seq_len:    %d\n", sl);
    printf("  file_size:  %.1f MB\n", file_size / (1024.0 * 1024.0));
    printf("  orig_size:  %.1f MB\n", orig_bytes / (1024.0 * 1024.0));
    printf("  ratio:      %.1fx\n",
           file_size > 0 ? (double)orig_bytes / file_size : 0.0);

    eakv_cache_free(cache);
    return 0;
}

static int cmd_validate(const char *path) {
    eakv_cache_t *cache = NULL;
    int rc = eakv_cache_load(path, &cache);
    if (rc != EAKV_OK) {
        fprintf(stderr, "error: failed to load %s (code %d)\n", path, rc);
        return 1;
    }

    int nl = cache->n_layers;
    int n_groups = (cache->n_kv_heads * cache->head_dim * cache->seq_len) / 64;
    int32_t *scales_bits = malloc((size_t)n_groups * sizeof(int32_t));
    int32_t *biases_bits = malloc((size_t)n_groups * sizeof(int32_t));
    if (!scales_bits || !biases_bits) {
        fprintf(stderr, "error: allocation failed\n");
        free(scales_bits); free(biases_bits);
        eakv_cache_free(cache);
        return 1;
    }

    int errors = 0;
    for (int l = 0; l < nl; l++) {
        for (int kv = 0; kv < 2; kv++) {
            eakv_kv_data_t *d = &cache->kv[l * 2 + kv];
            const char *kv_name = kv == 0 ? "K" : "V";

            memcpy(scales_bits, d->scales, (size_t)n_groups * sizeof(int32_t));
            memcpy(biases_bits, d->biases, (size_t)n_groups * sizeof(int32_t));

            int32_t result = q4_validate(
                d->scales, d->biases,
                scales_bits, biases_bits, n_groups);

            if (result == 1) {
                fprintf(stderr, "  FAIL: NaN scale in layer %d %s\n", l, kv_name);
                errors++;
            } else if (result == 2) {
                fprintf(stderr, "  FAIL: NaN bias in layer %d %s\n", l, kv_name);
                errors++;
            } else if (result == 3) {
                fprintf(stderr, "  FAIL: negative scale in layer %d %s\n", l, kv_name);
                errors++;
            } else if (result != 0) {
                fprintf(stderr, "  FAIL: error %d in layer %d %s\n", result, l, kv_name);
                errors++;
            }
        }
    }

    free(scales_bits);
    free(biases_bits);

    if (errors == 0) {
        printf("%s: ok (%d layers checked)\n", path, nl);
    } else {
        fprintf(stderr, "%s: %d errors\n", path, errors);
    }

    eakv_cache_free(cache);
    return errors ? 1 : 0;
}

static void usage(void) {
    fprintf(stderr,
        "usage: eakv <command> [args]\n"
        "\n"
        "commands:\n"
        "  inspect <file.eakv>    show metadata\n"
        "  validate <file.eakv>   check integrity\n"
    );
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }

    const char *cmd = argv[1];

    if (strcmp(cmd, "inspect") == 0) {
        if (argc < 3) { fprintf(stderr, "error: missing file\n"); return 1; }
        return cmd_inspect(argv[2]);
    }

    if (strcmp(cmd, "validate") == 0) {
        if (argc < 3) { fprintf(stderr, "error: missing file\n"); return 1; }
        return cmd_validate(argv[2]);
    }

    fprintf(stderr, "error: unknown command '%s'\n", cmd);
    usage();
    return 1;
}
