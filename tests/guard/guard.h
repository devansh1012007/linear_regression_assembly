/* guard.h — guard-page buffer allocator (asm-verification §3).
 * ASan does not instrument asm; page protection does.
 * guard_end(bytes)   : buffer ends exactly at a PROT_NONE page (overread faults)
 * guard_begin(bytes) : buffer begins right after a PROT_NONE page (underread faults)
 * Both leave canary-mapped pages beyond the guard so use-after-protect shows up
 * as SIGSEGV rather than silent success. Free with guard_free(p, bytes).
 */
#ifndef GUARD_H
#define GUARD_H
#define _GNU_SOURCE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <sys/mman.h>
#include <stddef.h>
#include <stdint.h>

static size_t g_pgsize = 4096;

static size_t guard_pgsize(void) {
#ifdef _SC_PAGESIZE
    long s = sysconf(_SC_PAGESIZE);
    if (s > 0) g_pgsize = (size_t)s;
#endif
    return g_pgsize;
}

/* Buffer occupying the END of the last RW page before a PROT_NONE page. */
static void *guard_end(size_t bytes) {
    size_t pg = guard_pgsize();
    if (bytes == 0) bytes = 1;                 /* keep at least one byte live */
    size_t pages = (bytes + pg - 1) / pg;
    char *base = mmap(0, (pages + 1) * pg, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;
    mprotect(base + pages * pg, pg, PROT_NONE);
    return base + pages * pg - bytes;
}

/* Buffer starting right AFTER a PROT_NONE page (underread faults). */
static void *guard_begin(size_t bytes) {
    size_t pg = guard_pgsize();
    if (bytes == 0) bytes = 1;
    size_t pages = (bytes + pg - 1) / pg;
    char *base = mmap(0, (pages + 1) * pg, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;
    mprotect(base, pg, PROT_NONE);
    return base + pg;
}

static void guard_free(void *p, size_t bytes) {
    (void)p; (void)bytes;                       /* retained for API symmetry */
}

/* Deterministic variant that remembers the mapping base: use these in tests. */
struct guard_buf { void *base; void *ptr; size_t map_bytes; };

static struct guard_buf gbuf_end(size_t bytes) {
    struct guard_buf g = {0};
    size_t pg = guard_pgsize();
    if (bytes == 0) bytes = 1;
    size_t pages = (bytes + pg - 1) / pg;
    g.map_bytes = (pages + 1) * pg;
    g.base = mmap(0, g.map_bytes, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g.base == MAP_FAILED) return g;
    mprotect((char *)g.base + pages * pg, pg, PROT_NONE);
    g.ptr = (char *)g.base + pages * pg - bytes;
    return g;
}

static struct guard_buf gbuf_begin(size_t bytes) {
    struct guard_buf g = {0};
    size_t pg = guard_pgsize();
    if (bytes == 0) bytes = 1;
    size_t pages = (bytes + pg - 1) / pg;
    g.map_bytes = (pages + 1) * pg;
    g.base = mmap(0, g.map_bytes, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g.base == MAP_FAILED) return g;
    mprotect(g.base, pg, PROT_NONE);
    g.ptr = (char *)g.base + pg;
    return g;
}

static void gbuf_free(struct guard_buf *g) {
    if (g->base && g->base != MAP_FAILED) munmap(g->base, g->map_bytes);
    *g = (struct guard_buf){0};
}
#pragma GCC diagnostic pop
#endif
