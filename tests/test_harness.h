/* Minimal test harness — assert macros, no dependencies. */
#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int _tests_run = 0;
static int _tests_failed = 0;

#define TEST(name) static void name(void)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        _tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "  FAIL %s:%d: %s != %s (%d != %d)\n", \
                __FILE__, __LINE__, #a, #b, (int)(a), (int)(b)); \
        _tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (double)(a), _b = (double)(b); \
    if (fabs(_a - _b) > (tol)) { \
        fprintf(stderr, "  FAIL %s:%d: %s ~= %s (%.6f != %.6f, tol=%.6f)\n", \
                __FILE__, __LINE__, #a, #b, _a, _b, (double)(tol)); \
        _tests_failed++; \
        return; \
    } \
} while(0)

#define RUN(name) do { \
    _tests_run++; \
    printf("  %-50s", #name); \
    name(); \
    printf("ok\n"); \
} while(0)

#define TEST_MAIN() int main(void) { \
    printf("\n");

#define TEST_END() \
    printf("\n%d tests, %d failed\n", _tests_run, _tests_failed); \
    return _tests_failed ? 1 : 0; \
}

#endif
