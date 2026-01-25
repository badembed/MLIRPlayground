#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

// Generic view of a memref<...xf32>.
typedef struct {
    float   *data;     // pointer to "aligned" (a.k.a. data) pointer
    int64_t  offset;   // logical offset in ELEMENTS from data
    int64_t *sizes;    // array of length `rank`
    int64_t *strides;  // array of length `rank`, in ELEMENTS
    int64_t  rank;     // number of dimensions
} GenericMemRefF32;

#define MLIR_MAX_RANK 4

static int64_t
generic_memref_num_elements(const GenericMemRefF32 *m) {
    if (!m || m->rank <= 0)
        return 0;

    int64_t total = 1;
    for (int64_t d = 0; d < m->rank; ++d) {
        int64_t s = m->sizes[d];
        if (s <= 0)
            return 0;
        total *= s;
    }
    return total;
}

int64_t
generic_memref_compute_physical_index(const GenericMemRefF32 *m,
                                      int64_t linear_index) {
    int64_t rank = m->rank;

    int64_t coords[MLIR_MAX_RANK] = {0};

    for (int64_t d = rank - 1; d >= 0; --d) {
        int64_t size_d = m->sizes[d];
        int64_t coord_d = linear_index % size_d;
        coords[d] = coord_d;
        linear_index /= size_d;
    }

    int64_t idx = m->offset;
    for (int64_t d = 0; d < rank; ++d) {
        idx += coords[d] * m->strides[d];
    }

    return idx;
}


static int
generic_memref_check_shape_compat(const GenericMemRefF32 *a,
                                  const GenericMemRefF32 *b,
                                  const char *name_a,
                                  const char *name_b) {
    if (a->rank != b->rank) {
        fprintf(stderr,
                "Shape mismatch: rank(%s)=%lld != rank(%s)=%lld\n",
                name_a, (long long)a->rank,
                name_b, (long long)b->rank);
        return -1;
    }

    for (int64_t d = 0; d < a->rank; ++d) {
        if (a->sizes[d] != b->sizes[d]) {
            fprintf(stderr,
                    "Shape mismatch at dim %lld: %s.sizes[%lld]=%lld, %s.sizes[%lld]=%lld\n",
                    (long long)d,
                    name_a, (long long)d, (long long)a->sizes[d],
                    name_b, (long long)d, (long long)b->sizes[d]);
            return -1;
        }
    }

    return 0;
}

void
sim_linalg_add_f32(GenericMemRefF32 *a,
                   GenericMemRefF32 *b,
                   GenericMemRefF32 *out) {
    if (!a || !b || !out) {
        fprintf(stderr, "sim_linalg_add_f32: null argument(s)\n");
        return;
    }

    // Check shapes.
    if (generic_memref_check_shape_compat(a, b, "a", "b") != 0)
        return;
    if (generic_memref_check_shape_compat(a, out, "a", "out") != 0)
        return;

    int64_t total = generic_memref_num_elements(a);
    if (total <= 0) {
        fprintf(stderr, "sim_linalg_add_f32: non-positive number of elements\n");
        return;
    }

    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t ia = generic_memref_compute_physical_index(a, linear);
        int64_t ib = generic_memref_compute_physical_index(b, linear);
        int64_t io = generic_memref_compute_physical_index(out, linear);

        float va = a->data[ia];
        float vb = b->data[ib];
        out->data[io] = va + vb;
        printf("DBG: %g\n", out->data[io]);
    }
}
